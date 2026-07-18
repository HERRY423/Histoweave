"""First-class SOTA spatial-domain reproduction pipeline (P2).

Goals
-----
1. **Probe** which official backends are installable in the current process /
   configured interpreters.
2. **Run** DLPFC domain-detection cells with explicit success/failure status
   (never silent toy substitutes).
3. **Emit** ``sota_benchmark_long.csv`` + throughput JSON that
   :mod:`landscape_io` and the public leaderboard can merge.

Environment variables (per method interpreter isolation)
--------------------------------------------------------
``HISTOWEAVE_SPAGCN_PYTHON``, ``HISTOWEAVE_GRAPHST_PYTHON``,
``HISTOWEAVE_STAGATE_PYTHON``, ``HISTOWEAVE_BAYESSPACE_PYTHON``,
``HISTOWEAVE_R_LIB``, ``HISTOWEAVE_SOTA_DEVICE``, ``HISTOWEAVE_SOTA_TIMEOUT``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from .task_contract import AnalysisTask, GroundTruthKind

LOG = logging.getLogger("histoweave.benchmark.sota_pipeline")

SOTA_METHODS = ("spagcn", "graphst", "stagate", "bayesspace", "banksy_py")
DEFAULT_SLICES = ("151673", "151674", "151507", "151669", "151670")
DEFAULT_SEEDS = (42, 1, 2)
PROTOCOL = "histoweave.sota_dlpfc.v1"


@dataclass(frozen=True)
class BackendProbe:
    """Result of probing one SOTA backend."""

    method: str
    available: bool
    runtime: str
    detail: str
    interpreter: str | None = None
    package_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SotaCellResult:
    """One dataset × method × seed evaluation cell."""

    dataset: str
    method: str
    seed: int
    ari: float | None
    seconds: float
    status: str
    error: str | None = None
    n_domains_truth: int | None = None
    n_obs: int | None = None
    oracle_k: bool = True

    def to_row(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "ari": self.ari,
            "seconds": round(self.seconds, 4),
            "status": self.status,
            "error": self.error or "",
            "n_domains_truth": self.n_domains_truth,
            "n_obs": self.n_obs,
            "oracle_k": int(self.oracle_k),
            "family": "sota" if self.method != "banksy_py" else "spatial_aware",
            "config": self.method,
        }


@dataclass
class SotaBenchmarkReport:
    """Full probe + optional run summary."""

    protocol: str = PROTOCOL
    task: str = AnalysisTask.SPATIAL_DOMAIN.value
    ground_truth_kind: str = GroundTruthKind.SPATIAL_DOMAIN.value
    probes: list[BackendProbe] = field(default_factory=list)
    cells: list[SotaCellResult] = field(default_factory=list)
    dry_run: bool = False
    started_at: str = ""
    finished_at: str = ""
    notes: list[str] = field(default_factory=list)

    def available_methods(self) -> list[str]:
        return [p.method for p in self.probes if p.available]

    def throughput_summary(self) -> dict[str, Any]:
        ok = [c for c in self.cells if c.status == "success" and c.ari is not None]
        failed = [c for c in self.cells if c.status in {"failed", "error", "timeout", "oom"}]
        skipped = [c for c in self.cells if c.status.startswith("skipped")]
        by_method: dict[str, dict[str, Any]] = {}
        for method in {c.method for c in self.cells}:
            rows = [c for c in self.cells if c.method == method]
            success = [c for c in rows if c.status == "success" and c.ari is not None]
            by_method[method] = {
                "n_cells": len(rows),
                "n_success": len(success),
                "mean_ari": (float(np.mean([c.ari for c in success])) if success else None),
                "mean_seconds": (float(np.mean([c.seconds for c in success])) if success else None),
                "statuses": sorted({c.status for c in rows}),
            }
        return {
            "protocol": self.protocol,
            "dry_run": self.dry_run,
            "n_cells": len(self.cells),
            "n_success": len(ok),
            "n_failed": len(failed),
            "n_skipped": len(skipped),
            "available_methods": self.available_methods(),
            "by_method": by_method,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "notes": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "task": self.task,
            "ground_truth_kind": self.ground_truth_kind,
            "dry_run": self.dry_run,
            "probes": [p.to_dict() for p in self.probes],
            "throughput": self.throughput_summary(),
            "n_cells": len(self.cells),
            "notes": list(self.notes),
        }


def _python_for(method: str) -> str:
    env_key = f"HISTOWEAVE_{method.upper()}_PYTHON"
    return os.environ.get(env_key, sys.executable)


def probe_backend(method: str) -> BackendProbe:
    """Probe whether a SOTA backend can be imported or reached."""
    method = method.lower()
    if method == "banksy_py":
        try:
            from histoweave.plugins import get_method

            cls = get_method("domain_detection", "banksy_py")
            return BackendProbe(
                method=method,
                available=True,
                runtime="python",
                detail="native histoweave banksy_py plugin",
                interpreter=sys.executable,
                package_version=cls.spec.version,
            )
        except Exception as exc:  # pragma: no cover - registry always present in tests
            return BackendProbe(method, False, "python", f"plugin missing: {exc}")

    if method == "bayesspace":
        rscript = shutil.which("Rscript")
        if rscript is None:
            return BackendProbe(method, False, "r", "Rscript not on PATH")
        # Lightweight check: R can start. Full BayesSpace install is validated at run.
        try:
            proc = subprocess.run(
                [rscript, "--vanilla", "-e", "cat(R.version.string)"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if proc.returncode != 0:
                return BackendProbe(method, False, "r", proc.stderr[-200:] or "R failed")
            # Optional package check
            check = subprocess.run(
                [
                    rscript,
                    "--vanilla",
                    "-e",
                    (
                        "if (!requireNamespace('BayesSpace', quietly=TRUE)) quit(status=2); "
                        "cat(as.character(packageVersion('BayesSpace')))"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env={**os.environ, "R_LIBS_USER": os.environ.get("HISTOWEAVE_R_LIB", "")},
            )
            if check.returncode != 0:
                return BackendProbe(
                    method,
                    False,
                    "r",
                    "BayesSpace package not installed (BiocManager::install('BayesSpace'))",
                    interpreter=rscript,
                )
            return BackendProbe(
                method,
                True,
                "r",
                "Rscript + BayesSpace available",
                interpreter=rscript,
                package_version=check.stdout.strip() or None,
            )
        except Exception as exc:
            return BackendProbe(method, False, "r", str(exc)[:300])

    # Python packages in method-specific interpreters
    package_map = {
        "spagcn": ("SpaGCN", "SpaGCN"),
        "graphst": ("GraphST", "GraphST"),
        "stagate": ("STAGATE_pyG", "STAGATE_pyG"),
    }
    if method not in package_map:
        return BackendProbe(method, False, "unknown", f"unknown method {method!r}")
    mod_name, display = package_map[method]
    python = _python_for(method)
    script = (
        "import importlib, json\n"
        f"mod = importlib.import_module({mod_name!r})\n"
        "ver = getattr(mod, '__version__', None)\n"
        "print(json.dumps({'ok': True, 'version': ver}))\n"
    )
    try:
        proc = subprocess.run(
            [python, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "import failed").strip().splitlines()
            return BackendProbe(
                method,
                False,
                "python",
                err[-1][:300] if err else f"{display} not importable",
                interpreter=python,
            )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        return BackendProbe(
            method,
            True,
            "python",
            f"{display} importable",
            interpreter=python,
            package_version=payload.get("version"),
        )
    except Exception as exc:
        return BackendProbe(method, False, "python", str(exc)[:300], interpreter=python)


def probe_all(methods: Iterable[str] = SOTA_METHODS) -> list[BackendProbe]:
    return [probe_backend(m) for m in methods]


def _bundle_path(sid: str, repo_root: Path) -> Path:
    override = os.environ.get("HISTOWEAVE_DLPFC_DATA")
    if override:
        candidate = Path(override) / f"{sid}.h5ad"
        if candidate.exists():
            return candidate
        candidate = Path(override) / f"dlpfc_{sid}.h5ad"
        if candidate.exists():
            return candidate
    root = Path(os.environ.get("HISTOWEAVE_LOCAL_DATA", repo_root))
    return root / "datasets_cache" / "dlpfc" / f"dlpfc_{sid}.h5ad"


def load_dlpfc_slice(sid: str, *, repo_root: Path | None = None):
    """Load a DLPFC h5ad bundle as SpatialTable + n_domains."""
    import pandas as pd

    from ..data import SpatialTable

    root = repo_root or Path(__file__).resolve().parents[3]
    path = _bundle_path(sid, root)
    if not path.exists():
        raise FileNotFoundError(
            f"DLPFC bundle missing at {path}. Set HISTOWEAVE_LOCAL_DATA or HISTOWEAVE_DLPFC_DATA."
        )
    try:
        import anndata as ad
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("anndata is required to load DLPFC bundles") from exc

    adata = ad.read_h5ad(path)
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    truth_col = "domain_truth" if "domain_truth" in adata.obs else "spatialLIBD_layer"
    if truth_col not in adata.obs:
        raise ValueError(f"{path} lacks domain_truth / spatialLIBD_layer")
    truth = pd.Categorical(list(adata.obs[truth_col].astype(str)))
    obs = pd.DataFrame({"domain_truth": truth}, index=adata.obs_names.astype(str))
    for coord in ("array_row", "array_col"):
        if coord in adata.obs:
            obs[coord] = adata.obs[coord].to_numpy()
    layers = {"counts": counts}
    table = SpatialTable(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=adata.var_names.astype(str)),
        obsm={"spatial": np.asarray(adata.obsm["spatial"], dtype=np.float32)},
        uns={"slice_id": sid, "assay": "visium", "platform": "visium"},
        layers=layers,
    )
    return table, int(pd.Series(truth).nunique())


def _run_banksy_py(table, *, n_domains: int, seed: int) -> np.ndarray:
    from ..plugins import create_method

    # Prefer count layer semantics; banksy_py accepts expression in X.
    result = create_method(
        "domain_detection",
        "banksy_py",
        n_domains=n_domains,
        random_state=seed,
        lambda_param=0.8,
    ).run(table.copy())
    return result.obs["domain"].astype(str).to_numpy()


def _run_plugin_or_fail(method: str, table, *, n_domains: int, seed: int) -> np.ndarray:
    """Run first-class registry plugins when available."""
    from ..plugins import create_method

    params: dict[str, Any] = {"n_domains": n_domains, "random_state": seed}
    if method == "banksy_py":
        return _run_banksy_py(table, n_domains=n_domains, seed=seed)
    if method == "spagcn":
        return (
            create_method("domain_detection", "spagcn", **params)
            .run(table.copy())
            .obs["domain"]
            .astype(str)
            .to_numpy()
        )
    if method == "graphst":
        return (
            create_method("domain_detection", "graphst", **params)
            .run(table.copy())
            .obs["domain"]
            .astype(str)
            .to_numpy()
        )
    if method == "stagate":
        return (
            create_method("domain_detection", "stagate", **params)
            .run(table.copy())
            .obs["domain"]
            .astype(str)
            .to_numpy()
        )
    if method == "bayesspace":
        return (
            create_method("domain_detection", "bayesspace", **params)
            .run(table.copy())
            .obs["domain"]
            .astype(str)
            .to_numpy()
        )
    raise KeyError(method)


def run_sota_cell(
    method: str,
    sid: str,
    seed: int,
    *,
    repo_root: Path | None = None,
    checkpoint_dir: Path | None = None,
    force: bool = False,
) -> SotaCellResult:
    """Evaluate one SOTA method on one DLPFC slice / seed."""
    from sklearn.metrics import adjusted_rand_score

    root = repo_root or Path(__file__).resolve().parents[3]
    ckpt_dir = checkpoint_dir or (root / "5x15_spatial_aware" / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"sota_{method}__{sid}__seed{seed}.json"
    if ckpt.exists() and not force:
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        return SotaCellResult(
            dataset=sid,
            method=method,
            seed=seed,
            ari=payload.get("ari"),
            seconds=float(payload.get("seconds", 0.0)),
            status=payload.get("status", "success" if payload.get("ari") is not None else "failed"),
            error=payload.get("error"),
            n_domains_truth=payload.get("n_domains_truth"),
            n_obs=payload.get("n_obs"),
            oracle_k=bool(payload.get("oracle_k", True)),
        )

    probe = probe_backend(method)
    if not probe.available:
        result = SotaCellResult(
            dataset=sid,
            method=method,
            seed=seed,
            ari=None,
            seconds=0.0,
            status="skipped_missing_backend",
            error=probe.detail,
        )
        ckpt.write_text(json.dumps(result.to_row(), allow_nan=False), encoding="utf-8")
        return result

    t0 = time.perf_counter()
    try:
        table, n_domains = load_dlpfc_slice(sid, repo_root=root)
        labels = _run_plugin_or_fail(method, table, n_domains=n_domains, seed=seed)
        truth = table.obs["domain_truth"].astype(str).to_numpy()
        ari = float(adjusted_rand_score(truth, labels))
        elapsed = time.perf_counter() - t0
        result = SotaCellResult(
            dataset=sid,
            method=method,
            seed=seed,
            ari=ari if np.isfinite(ari) else None,
            seconds=elapsed,
            status="success" if np.isfinite(ari) else "failed",
            n_domains_truth=n_domains,
            n_obs=table.n_obs,
            oracle_k=True,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        result = SotaCellResult(
            dataset=sid,
            method=method,
            seed=seed,
            ari=None,
            seconds=elapsed,
            status="failed",
            error=f"{type(exc).__name__}: {exc}"[:400],
        )
    # JSON cannot encode NaN; store nulls via None.
    row = result.to_row()
    temporary = ckpt.with_name(f".{ckpt.name}.tmp-{uuid4().hex}")
    temporary.write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
    temporary.replace(ckpt)
    return result


def run_sota_benchmark(
    *,
    methods: Iterable[str] | None = None,
    slices: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    repo_root: Path | None = None,
    out_dir: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    only_available: bool = True,
) -> SotaBenchmarkReport:
    """Probe backends and optionally evaluate the full SOTA DLPFC grid."""
    from datetime import UTC, datetime

    root = repo_root or Path(__file__).resolve().parents[3]
    out = Path(out_dir or root / "5x15_spatial_aware")
    out.mkdir(parents=True, exist_ok=True)
    method_list = list(methods or SOTA_METHODS)
    slice_list = list(slices or DEFAULT_SLICES)
    seed_list = list(seeds or DEFAULT_SEEDS)

    started = datetime.now(UTC).isoformat(timespec="seconds")
    probes = probe_all(method_list)
    report = SotaBenchmarkReport(
        probes=probes,
        dry_run=dry_run,
        started_at=started,
    )
    available = {p.method for p in probes if p.available}
    report.notes.append(
        f"available={sorted(available)}; missing={sorted(set(method_list) - available)}"
    )

    if dry_run:
        for method in method_list:
            for sid in slice_list:
                for seed in seed_list:
                    status = "skipped_dry_run" if method in available else "skipped_missing_backend"
                    detail = next(p.detail for p in probes if p.method == method)
                    report.cells.append(
                        SotaCellResult(
                            dataset=sid,
                            method=method,
                            seed=seed,
                            ari=None,
                            seconds=0.0,
                            status=status,
                            error=None if method in available else detail,
                        )
                    )
    else:
        run_methods = [m for m in method_list if m in available] if only_available else method_list
        for method in method_list:
            if method not in run_methods:
                for sid in slice_list:
                    for seed in seed_list:
                        detail = next(p.detail for p in probes if p.method == method)
                        report.cells.append(
                            SotaCellResult(
                                dataset=sid,
                                method=method,
                                seed=seed,
                                ari=None,
                                seconds=0.0,
                                status="skipped_missing_backend",
                                error=detail,
                            )
                        )
                continue
            for sid in slice_list:
                for seed in seed_list:
                    LOG.info("SOTA cell method=%s slice=%s seed=%s", method, sid, seed)
                    report.cells.append(
                        run_sota_cell(
                            method,
                            sid,
                            seed,
                            repo_root=root,
                            checkpoint_dir=out / "checkpoints",
                            force=force,
                        )
                    )

    report.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    write_sota_artifacts(report, out)
    return report


def write_sota_artifacts(report: SotaBenchmarkReport, out_dir: str | Path) -> dict[str, Path]:
    """Write long CSV + throughput JSON for leaderboard / landscape merge."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sota_benchmark_long.csv"
    json_path = out / "sota_throughput.json"
    probe_path = out / "sota_probe.json"

    fieldnames = [
        "dataset",
        "method",
        "seed",
        "ari",
        "seconds",
        "status",
        "error",
        "n_domains_truth",
        "n_obs",
        "oracle_k",
        "family",
        "config",
    ]
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{uuid4().hex}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in report.cells:
            row = cell.to_row()
            # CSV empty string for missing ARI
            if row["ari"] is None:
                row["ari"] = ""
            writer.writerow(row)
    temporary.replace(csv_path)

    probe_path.write_text(
        json.dumps(
            {
                "protocol": report.protocol,
                "probes": [p.to_dict() for p in report.probes],
                "generated_at": report.finished_at or report.started_at,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(report.throughput_summary(), indent=2), encoding="utf-8")
    return {"csv": csv_path, "throughput": json_path, "probe": probe_path}


def env_contract() -> dict[str, Any]:
    """Machine-readable environment contract for docs and CI."""
    return {
        "protocol": PROTOCOL,
        "task": AnalysisTask.SPATIAL_DOMAIN.value,
        "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
        "methods": {
            "spagcn": {
                "python_package": "SpaGCN==1.2.7",
                "env_python": "HISTOWEAVE_SPAGCN_PYTHON",
                "notes": "Isolated env recommended (NumPy/Scanpy pins conflict with GraphST).",
            },
            "graphst": {
                "python_package": "JinmiaoChenLab/GraphST",
                "env_python": "HISTOWEAVE_GRAPHST_PYTHON",
                "device": "HISTOWEAVE_SOTA_DEVICE=cpu|cuda",
            },
            "stagate": {
                "python_package": "QIFEIDKN/STAGATE_pyG",
                "env_python": "HISTOWEAVE_STAGATE_PYTHON",
                "device": "HISTOWEAVE_SOTA_DEVICE=cpu|cuda",
            },
            "bayesspace": {
                "r_package": "Bioconductor::BayesSpace",
                "env_r_lib": "HISTOWEAVE_R_LIB",
                "requires": ["Rscript", "zellkonverter"],
            },
            "banksy_py": {
                "python_package": "histoweave (native)",
                "notes": "Always available when histoweave is installed.",
            },
        },
        "timeout_seconds_env": "HISTOWEAVE_SOTA_TIMEOUT",
        "default_timeout_seconds": 7200,
        "slices": list(DEFAULT_SLICES),
        "seeds": list(DEFAULT_SEEDS),
        "outputs": [
            "sota_benchmark_long.csv",
            "sota_throughput.json",
            "sota_probe.json",
        ],
    }
