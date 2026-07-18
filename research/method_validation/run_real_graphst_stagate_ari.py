"""Real multi-slice ARI for GraphST and STAGATE on DLPFC.

Uses official backends via histoweave plugins (no mocks). Writes:

* ``results/graphst_stagate_real_ari.json``
* appends rows to ``5x15_spatial_aware/sota_benchmark_long.csv`` (best-effort)
* refreshes validation reports when compile is re-run

Environment
-----------
``KMP_DUPLICATE_LIB_OK=TRUE`` recommended on Windows OpenMP dual-load.
``HISTOWEAVE_SOTA_DEVICE=cpu|cuda`` (default cpu).
``HISTOWEAVE_REAL_ARI_MAX_OBS`` optional subsample (default full slice).
``HISTOWEAVE_GRAPHST_EPOCHS`` default 200 (paper often 600; reduced for runtime).
``HISTOWEAVE_STAGATE_EPOCHS`` default 200 (paper often 1000).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.sota_pipeline import (  # noqa: E402
    load_dlpfc_slice,
    probe_backend,
)
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

logger = logging.getLogger("real_gs_ari")
OUT = Path(__file__).resolve().parent / "results"
CKPT = ROOT / "5x15_spatial_aware" / "checkpoints"
SLICES = ("151673", "151674", "151507", "151669", "151670")
SEEDS = (42, 1, 2)
METHODS = ("graphst", "stagate")


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _subsample(table, max_obs: int | None, seed: int):
    if max_obs is None or table.n_obs <= max_obs:
        return table
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(table.n_obs, max_obs, replace=False))
    mask = np.zeros(table.n_obs, dtype=bool)
    mask[idx] = True
    return table.subset_obs(mask)


def _run_one(method: str, sid: str, seed: int, *, force: bool = False) -> dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score

    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT / f"sota_{method}__{sid}__seed{seed}.json"
    if ckpt.is_file() and not force:
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        if payload.get("ari") is not None and payload.get("status") == "success":
            # Only reuse if it was a real run (not structural mock marker)
            if payload.get("backend") != "mock_official_api":
                logger.info(
                    "reuse cache %s %s seed=%s ari=%s", method, sid, seed, payload.get("ari")
                )
                return payload

    probe = probe_backend(method)
    if not probe.available:
        row = {
            "dataset": sid,
            "method": method,
            "seed": seed,
            "ari": None,
            "seconds": 0.0,
            "status": "skipped_missing_backend",
            "error": probe.detail,
            "backend": "missing",
        }
        ckpt.write_text(json.dumps(row), encoding="utf-8")
        return row

    max_obs_env = os.environ.get("HISTOWEAVE_REAL_ARI_MAX_OBS")
    max_obs = int(max_obs_env) if max_obs_env else None
    graphst_epochs = int(os.environ.get("HISTOWEAVE_GRAPHST_EPOCHS", "200"))
    stagate_epochs = int(os.environ.get("HISTOWEAVE_STAGATE_EPOCHS", "200"))

    t0 = time.perf_counter()
    try:
        table, n_domains = load_dlpfc_slice(sid, repo_root=ROOT)
        table = _subsample(table, max_obs, seed)
        # keep n_domains from full truth; recompute if subsampled
        if "domain_truth" in table.obs:
            n_domains = int(table.obs["domain_truth"].nunique())
        params: dict[str, Any] = {"n_domains": n_domains, "random_state": seed}
        if method == "graphst":
            params["epochs"] = graphst_epochs
        elif method == "stagate":
            params["n_epochs"] = stagate_epochs
        out = create_method(MethodCategory.DOMAIN_DETECTION, method, **params).run(table.copy())
        pred = out.obs["domain"].astype(str).to_numpy()
        truth = table.obs["domain_truth"].astype(str).to_numpy()
        ari = float(adjusted_rand_score(truth, pred))
        elapsed = time.perf_counter() - t0
        row = {
            "dataset": sid,
            "method": method,
            "seed": seed,
            "ari": ari if np.isfinite(ari) else None,
            "seconds": round(elapsed, 3),
            "status": "success" if np.isfinite(ari) else "failed",
            "error": None,
            "n_domains_truth": n_domains,
            "n_obs": int(table.n_obs),
            "oracle_k": True,
            "family": "sota",
            "config": method,
            "backend": f"official_{method}",
            "epochs": params.get("epochs") or params.get("n_epochs"),
        }
        logger.info(
            "%s %s seed=%s ARI=%.4f (%.1fs n=%s)", method, sid, seed, ari, elapsed, table.n_obs
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        row = {
            "dataset": sid,
            "method": method,
            "seed": seed,
            "ari": None,
            "seconds": round(elapsed, 3),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "backend": f"official_{method}",
        }
        logger.exception("%s %s seed=%s FAILED: %s", method, sid, seed, exc)

    ckpt.write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
    return row


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for method in METHODS:
        mrows = [r for r in rows if r["method"] == method]
        ok = [r for r in mrows if r.get("status") == "success" and r.get("ari") is not None]
        per: dict[str, list[float]] = {}
        for r in ok:
            per.setdefault(str(r["dataset"]), []).append(float(r["ari"]))
        per_mean = {k: float(np.mean(v)) for k, v in per.items()}
        by_method[method] = {
            "n_cells": len(mrows),
            "n_success": len(ok),
            "mean_ari": float(np.mean([r["ari"] for r in ok])) if ok else None,
            "std_ari": float(np.std([r["ari"] for r in ok], ddof=0)) if ok else None,
            "per_dataset_mean_ari": per_mean,
            "mean_seconds": float(np.mean([r["seconds"] for r in ok])) if ok else None,
            "statuses": sorted({r.get("status", "?") for r in mrows}),
            "errors": [r.get("error") for r in mrows if r.get("error")][:5],
        }
    return by_method


def _append_sota_csv(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "5x15_spatial_aware" / "sota_benchmark_long.csv"
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
    existing = []
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    # Only replace methods present in this run (do not wipe sibling methods).
    touched = {str(r.get("method")) for r in rows if r.get("method")}
    keep = [r for r in existing if r.get("method") not in touched]
    for r in rows:
        keep.append({k: r.get(k, "") for k in fieldnames})
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in keep:
            w.writerow(r)
    logger.info("updated %s (%s rows; replaced %s)", path, len(keep), sorted(touched))


def main(argv: list[str] | None = None) -> int:
    import argparse

    _setup()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", default="graphst,stagate")
    p.add_argument("--slices", default=",".join(SLICES))
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-obs", type=int, default=None, help="Subsample spots for faster runs")
    args = p.parse_args(argv)

    if args.max_obs:
        os.environ["HISTOWEAVE_REAL_ARI_MAX_OBS"] = str(args.max_obs)

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    slices = tuple(s.strip() for s in args.slices.split(",") if s.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    for m in methods:
        probe = probe_backend(m)
        logger.info("probe %s available=%s detail=%s", m, probe.available, probe.detail)

    rows: list[dict[str, Any]] = []
    for method in methods:
        for sid in slices:
            for seed in seeds:
                rows.append(_run_one(method, sid, seed, force=args.force))

    summary = _summarize(rows)
    payload = {
        "protocol": "histoweave.sota_dlpfc.v1",
        "backend_mode": "official_real",
        "methods": list(methods),
        "slices": list(slices),
        "seeds": list(seeds),
        "max_obs": args.max_obs or os.environ.get("HISTOWEAVE_REAL_ARI_MAX_OBS"),
        "rows": rows,
        "summary": summary,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "graphst_stagate_real_ari.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", out_path)

    # Only append successful or explicit failed real backend runs (not missing)
    real_rows = [r for r in rows if r.get("backend", "").startswith("official_")]
    if real_rows:
        _append_sota_csv(real_rows)

    for method, body in summary.items():
        logger.info(
            "SUMMARY %s success=%s/%s mean_ari=%s per=%s",
            method,
            body["n_success"],
            body["n_cells"],
            body["mean_ari"],
            body["per_dataset_mean_ari"],
        )

    # Exit 0 if at least one method has real multi-slice ARI
    ok = any(
        (summary.get(m) or {}).get("n_success", 0) >= 3
        and (summary.get(m) or {}).get("mean_ari") is not None
        for m in methods
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
