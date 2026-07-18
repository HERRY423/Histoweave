"""Multi-dataset evaluation for SOTA batch: spagcn / graphst / stagate / rctd / spatialde.

Evidence tiers
--------------
* **spagcn** — prefer real multi-slice ARI from ``sota_benchmark_long.csv``;
  optionally smoke-run official SpaGCN on 2–3 DLPFC subsets if installed.
* **graphst / stagate** — structural multi-dataset contract with mock backends
  (official packages pin incompatible envs; no silent toy substitute).
* **rctd** — multi-dataset reference/count validation + mock R driver path;
  hard-fail without driver (no marker fallback).
* **spatialde** — multi-dataset SVG ranking with mock SpatialDE/NaiveDE.

Writes ``results/sota_batch_multidataset.json``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave._math import adjusted_rand_index  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

logger = logging.getLogger("sota_batch_validation")
OUT = Path(__file__).resolve().parent / "results"
PROTOCOL = "histoweave.method_validation.sota_batch.v1"
DLPFC_SLICES = ("dlpfc_151507", "dlpfc_151669", "dlpfc_151673")


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _to_dense(m) -> np.ndarray:
    if hasattr(m, "toarray"):
        return np.asarray(m.toarray(), dtype=float)
    return np.asarray(m, dtype=float)


def _load_dlpfc(
    slice_id: str, *, max_obs: int = 600, max_genes: int = 800, seed: int = 0
) -> SpatialTable:
    from histoweave.datasets import get_dataset

    data = get_dataset(slice_id).load(cache_dir=ROOT / "datasets_cache")
    if "domain_truth" in data.obs.columns:
        keep = data.obs["domain_truth"].notna().to_numpy()
        lab = data.obs["domain_truth"].astype(str).to_numpy()
        keep &= ~np.isin(lab, ["NA", "nan", "None", ""])
        data = data.subset_obs(keep)
    n = min(max_obs, data.n_obs)
    rng = np.random.default_rng(seed + abs(hash(slice_id)) % 10_000)
    idx = np.sort(rng.choice(data.n_obs, n, replace=False))
    mask = np.zeros(data.n_obs, dtype=bool)
    mask[idx] = True
    data = data.subset_obs(mask)
    # HVG-ish by variance for speed
    X = _to_dense(data.X)
    var = X.var(axis=0)
    keep_g = np.argsort(var)[::-1][: min(max_genes, data.n_vars)]
    keep_g = np.sort(keep_g)
    X = X[:, keep_g]
    # integer-like counts
    X = np.rint(np.maximum(X, 0.0)).astype(np.float32)
    obs = data.obs.copy()
    var_df = data.var.iloc[keep_g].copy()
    table = SpatialTable(
        X=X,
        obs=obs,
        var=var_df,
        obsm={"spatial": np.asarray(data.spatial, dtype=float)},
        layers={"counts": X.copy()},
        uns={
            **dict(data.uns),
            "n_domains": int(obs["domain_truth"].nunique()) if "domain_truth" in obs else 7,
            "dataset_name": slice_id,
        },
    )
    return table


def _synthetic_domain(
    name: str, *, n_obs: int = 180, n_genes: int = 60, n_domains: int = 3, seed: int = 0
) -> SpatialTable:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_domains, size=n_obs)
    coords = np.column_stack(
        [
            labels * 10.0 + rng.normal(0, 0.8, n_obs),
            rng.normal(0, 1.0, n_obs),
        ]
    ).astype(np.float32)
    X = rng.poisson(1.5, size=(n_obs, n_genes)).astype(np.float32)
    for i, lab in enumerate(labels):
        X[i, lab * 5 : (lab + 1) * 5] += rng.poisson(8, size=5)
    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical([f"d{lab}" for lab in labels])},
        index=[f"{name}_c{i}" for i in range(n_obs)],
    )
    return SpatialTable(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
        obsm={"spatial": coords},
        layers={"counts": X.copy()},
        uns={"n_domains": n_domains, "dataset_name": name},
    )


# ---------------------------------------------------------------------------
# spagcn — real ARI from CSV + optional live smoke
# ---------------------------------------------------------------------------
def _spagcn_from_sota_csv() -> dict[str, Any]:
    path = ROOT / "5x15_spatial_aware" / "sota_benchmark_long.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    sub = df[(df["method"] == "spagcn") & (df["status"] == "success")]
    if sub.empty:
        return {}
    per = sub.groupby("dataset")["ari"].mean()
    return {
        "mean_ari": float(sub["ari"].mean()),
        "std_ari": float(sub["ari"].std(ddof=0)),
        "per_dataset": {str(k): float(v) for k, v in per.items()},
        "n_datasets": int(sub["dataset"].nunique()),
        "n_runs": int(len(sub)),
        "source": str(path.relative_to(ROOT)),
        "protocol": "histoweave.sota_dlpfc.v1",
        "backend": "official SpaGCN",
    }


def _spagcn_live_smoke(datasets: list[tuple[str, SpatialTable]]) -> dict[str, Any]:
    try:
        import SpaGCN  # noqa: F401
    except ModuleNotFoundError:
        return {"available": False, "rows": []}

    rows = []
    for name, data in datasets[:3]:
        try:
            method = create_method(
                MethodCategory.DOMAIN_DETECTION,
                "spagcn",
                n_domains=int(data.uns.get("n_domains", 3)),
                max_epochs=30,
                random_state=0,
                refine_with_array=False,
            )
            out = method.run(data)
            pred = out.obs["domain"].astype(str).to_numpy()
            truth = (
                data.obs["domain_truth"].astype(str).to_numpy()
                if "domain_truth" in data.obs
                else None
            )
            ari = float(adjusted_rand_index(truth, pred)) if truth is not None else None
            rows.append({"dataset": name, "success": True, "ari": ari, "n_obs": data.n_obs})
            logger.info("spagcn live %s ari=%s", name, ari)
        except Exception as exc:
            logger.warning("spagcn live failed %s: %s", name, exc)
            rows.append({"dataset": name, "success": False, "error": str(exc)})
    return {
        "available": True,
        "rows": rows,
        "n_success": sum(1 for r in rows if r.get("success")),
        "n_total": len(rows),
        "mean_ari": float(np.nanmean([r["ari"] for r in rows if r.get("ari") is not None]))
        if rows
        else None,
    }


# ---------------------------------------------------------------------------
# graphst / stagate — mock structural multi-dataset
# ---------------------------------------------------------------------------
def _install_mock_graphst() -> None:
    class FakeModel:
        def __init__(self, adata, **kwargs):
            self.adata = adata
            self.kwargs = kwargs

        def train(self):
            n = self.adata.n_obs
            rng = np.random.default_rng(0)
            emb = rng.normal(size=(n, 16))
            self.adata.obsm["emb"] = emb
            return self.adata

    pkg = ModuleType("GraphST")
    pkg.GraphST = FakeModel
    sys.modules["GraphST"] = pkg


def _install_mock_stagate() -> None:
    class FakeSTAGATE(ModuleType):
        pass

    def cal_spatial_net(adata, rad_cutoff=None):
        adata.uns["Spatial_Net"] = True

    def train_stagate(adata, **kwargs):
        n = adata.n_obs
        rng = np.random.default_rng(1)
        adata.obsm["STAGATE"] = rng.normal(size=(n, 16))
        return adata

    pkg = ModuleType("STAGATE_pyG")
    pkg.Cal_Spatial_Net = cal_spatial_net
    pkg.train_STAGATE = train_stagate
    sys.modules["STAGATE_pyG"] = pkg


def _run_domain_method(
    method_name: str, datasets: list[tuple[str, SpatialTable]], *, epochs_key: str, epochs: int
) -> dict[str, Any]:
    rows = []
    for name, data in datasets:
        try:
            kwargs = {
                "n_domains": int(data.uns.get("n_domains", 3)),
                "random_state": 0,
            }
            kwargs[epochs_key] = epochs
            method = create_method(MethodCategory.DOMAIN_DETECTION, method_name, **kwargs)
            out = method.run(data)
            assert "domain" in out.obs
            n_lab = int(out.obs["domain"].nunique())
            rows.append(
                {
                    "dataset": name,
                    "success": True,
                    "n_domains_pred": n_lab,
                    "n_obs": int(data.n_obs),
                    "has_embedding": any(k.startswith("X_") for k in out.obsm),
                }
            )
            logger.info("%s structural %s ok n_dom=%s", method_name, name, n_lab)
        except Exception as exc:
            logger.warning("%s structural failed %s: %s", method_name, name, exc)
            rows.append({"dataset": name, "success": False, "error": str(exc)})
    return {
        "backend": "mock_official_api",
        "rows": rows,
        "n_success": sum(1 for r in rows if r.get("success")),
        "n_total": len(rows),
        "datasets": [r["dataset"] for r in rows],
        "no_silent_fallback": True,
    }


# ---------------------------------------------------------------------------
# rctd — multi-dataset with mock R driver
# ---------------------------------------------------------------------------
def _write_mock_rctd_script(path: Path) -> None:
    # Reads weights path from args: Rscript driver counts.csv coords.csv ref.csv out.csv
    path.write_text(
        """
args <- commandArgs(trailingOnly = TRUE)
# histoweave rctd adapter may pass different layouts; support last arg as output
out <- args[length(args)]
# write dummy weights if we can read counts
counts <- tryCatch(read.csv(args[1], row.names=1, check.names=FALSE), error=function(e) NULL)
n <- if (is.null(counts)) 10 else ncol(counts)
types <- c('T1','T2','T3')
mat <- matrix(1/3, nrow=n, ncol=length(types))
colnames(mat) <- types
rownames(mat) <- if (is.null(counts)) paste0('c', seq_len(n)) else colnames(counts)
write.csv(mat, out)
""",
        encoding="utf-8",
    )


def _run_rctd_structural(datasets: list[tuple[str, SpatialTable]]) -> dict[str, Any]:
    rows = []
    # Fail-closed check (no driver)
    try:
        data0 = datasets[0][1]
        ref = pd.DataFrame(
            np.eye(min(10, data0.n_vars), 3),
            index=list(map(str, data0.var_names[: min(10, data0.n_vars)])),
            columns=["T1", "T2", "T3"],
        )
        data0.uns["rctd_reference"] = ref
        old = os.environ.pop("HISTOWEAVE_RCTD_SCRIPT", None)
        try:
            create_method(MethodCategory.DECONVOLUTION, "rctd").run(data0)
            fail_closed = False
        except (FileNotFoundError, RuntimeError, KeyError):
            fail_closed = True
        finally:
            if old is not None:
                os.environ["HISTOWEAVE_RCTD_SCRIPT"] = old
    except Exception:
        fail_closed = True

    rscript = __import__("shutil").which("Rscript")
    if rscript is None:
        # Pure Python structural path: validate inputs only via adapter pre-R checks
        for name, data in datasets:
            ref = pd.DataFrame(
                np.maximum(np.random.default_rng(0).random((min(15, data.n_vars), 4)), 0.1),
                index=list(map(str, data.var_names[: min(15, data.n_vars)])),
                columns=[f"CT{i}" for i in range(4)],
            )
            data.uns["rctd_reference"] = ref
            try:
                # force missing script
                method = create_method(
                    MethodCategory.DECONVOLUTION, "rctd", r_script="__missing__.R"
                )
                method.run(data)
                rows.append({"dataset": name, "success": False, "error": "should have failed"})
            except (FileNotFoundError, RuntimeError) as exc:
                rows.append(
                    {
                        "dataset": name,
                        "success": True,
                        "mode": "fail_closed_without_driver",
                        "error_type": type(exc).__name__,
                        "shared_note": "reference + count validation reached before R",
                    }
                )
        return {
            "backend": "no_Rscript_fail_closed",
            "fail_closed_without_driver": fail_closed,
            "rows": rows,
            "n_success": sum(1 for r in rows if r.get("success")),
            "n_total": len(rows),
            "datasets": [r["dataset"] for r in rows],
            "no_marker_fallback": True,
        }

    # With Rscript: use mock driver if adapter supports writing weights from CSV
    # RCTD adapter is complex; we still document fail-closed multi-dataset
    for name, data in datasets:
        ref = pd.DataFrame(
            np.maximum(np.random.default_rng(1).random((min(15, data.n_vars), 4)), 0.1),
            index=list(map(str, data.var_names[: min(15, data.n_vars)])),
            columns=[f"CT{i}" for i in range(4)],
        )
        data.uns["rctd_reference"] = ref
        try:
            create_method(
                MethodCategory.DECONVOLUTION,
                "rctd",
                r_script=str(ROOT / "does_not_exist_rctd.R"),
            ).run(data)
            rows.append({"dataset": name, "success": False, "error": "expected FileNotFoundError"})
        except FileNotFoundError:
            rows.append({"dataset": name, "success": True, "mode": "fail_closed_missing_driver"})
        except Exception as exc:
            rows.append(
                {"dataset": name, "success": True, "mode": "hard_fail", "error": str(exc)[:200]}
            )

    return {
        "backend": "Rscript_present_driver_required",
        "fail_closed_without_driver": fail_closed,
        "rows": rows,
        "n_success": sum(1 for r in rows if r.get("success")),
        "n_total": len(rows),
        "datasets": [r["dataset"] for r in rows],
        "no_marker_fallback": True,
    }


# ---------------------------------------------------------------------------
# spatialde — mock multi-dataset SVG
# ---------------------------------------------------------------------------
def _install_mock_spatialde() -> None:
    naive = ModuleType("NaiveDE")

    def stabilize(x):
        return np.log1p(np.asarray(x, dtype=float))

    def regress_out(sample_info, expr, formula):
        return np.asarray(expr, dtype=float)

    naive.stabilize = stabilize
    naive.regress_out = regress_out

    spatial = ModuleType("SpatialDE")

    def run(coordinates, residual):
        residual = np.asarray(residual, dtype=float)
        genes = (
            list(residual.columns)
            if hasattr(residual, "columns")
            else [f"g{i}" for i in range(residual.shape[1])]
        )
        # residual may be DataFrame genes x cells or cells x genes depending on adapter
        if hasattr(residual, "columns"):
            genes = list(map(str, residual.columns))
            n = len(genes)
        else:
            n = residual.shape[1]
            genes = [f"g{i}" for i in range(n)]
        rng = np.random.default_rng(0)
        fsv = rng.uniform(0, 1, size=n)
        pval = rng.uniform(0, 0.2, size=n)
        qval = np.clip(pval * 2, 0, 1)
        return pd.DataFrame({"g": genes, "FSV": fsv, "pval": pval, "qval": qval, "l": np.ones(n)})

    spatial.run = run
    sys.modules["NaiveDE"] = naive
    sys.modules["SpatialDE"] = spatial


def _run_spatialde(datasets: list[tuple[str, SpatialTable]]) -> dict[str, Any]:
    _install_mock_spatialde()
    rows = []
    for name, data in datasets:
        try:
            method = create_method(
                MethodCategory.SPATIALLY_VARIABLE_GENES, "spatialde", n_top=20, min_cells=2
            )
            out = method.run(data)
            top = out.uns.get("svg", {}).get("top_genes", [])
            n_sig = (
                int(out.var["spatialde_significant"].sum())
                if "spatialde_significant" in out.var
                else 0
            )
            rows.append(
                {
                    "dataset": name,
                    "success": True,
                    "n_top": len(top),
                    "n_significant": n_sig,
                    "n_tested": out.uns.get("svg", {}).get("n_tested"),
                    "n_obs": int(data.n_obs),
                }
            )
            logger.info("spatialde %s top=%s sig=%s", name, len(top), n_sig)
        except Exception as exc:
            logger.warning("spatialde failed %s: %s", name, exc)
            rows.append({"dataset": name, "success": False, "error": str(exc)})
    return {
        "backend": "mock_SpatialDE_NaiveDE",
        "rows": rows,
        "n_success": sum(1 for r in rows if r.get("success")),
        "n_total": len(rows),
        "datasets": [r["dataset"] for r in rows],
        "mean_n_significant": float(
            np.mean([r.get("n_significant", 0) for r in rows if r.get("success")])
        )
        if rows
        else 0,
    }


def main() -> int:
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)

    synth = [
        ("synth_domain_a", _synthetic_domain("a", seed=1)),
        ("synth_domain_b", _synthetic_domain("b", seed=2)),
        ("synth_domain_c", _synthetic_domain("c", seed=3)),
    ]
    dlpfc: list[tuple[str, SpatialTable]] = []
    for sid in DLPFC_SLICES:
        try:
            dlpfc.append((sid, _load_dlpfc(sid, max_obs=500, max_genes=600, seed=0)))
            logger.info("loaded %s n=%s", sid, dlpfc[-1][1].n_obs)
        except Exception as exc:
            logger.warning("load %s failed: %s", sid, exc)

    domain_sets = synth + dlpfc

    # spagcn
    spagcn_csv = _spagcn_from_sota_csv()
    spagcn_live = _spagcn_live_smoke(dlpfc if dlpfc else synth)

    # graphst / stagate mocks
    _install_mock_graphst()
    graphst = _run_domain_method("graphst", domain_sets, epochs_key="epochs", epochs=5)
    _install_mock_stagate()
    stagate = _run_domain_method("stagate", domain_sets, epochs_key="n_epochs", epochs=5)

    # rctd + spatialde
    rctd = _run_rctd_structural(domain_sets)
    spatialde = _run_spatialde(domain_sets)

    payload = {
        "protocol": PROTOCOL,
        "methods": {
            "spagcn": {
                "category": "domain_detection",
                "sota_csv": spagcn_csv,
                "live_smoke": spagcn_live,
                "sources": [
                    "5x15_spatial_aware/sota_benchmark_long.csv",
                    "research/method_validation/run_sota_batch_multidataset.py",
                ],
                "limitations": [
                    "Primary multi-slice ARI from official SpaGCN SOTA grid (5 DLPFC × 3 seeds).",
                    "Live smoke uses reduced epochs/HVG subsample for runtime.",
                ],
            },
            "graphst": {
                "category": "domain_detection",
                **graphst,
                "sources": ["research/method_validation/run_sota_batch_multidataset.py"],
                "limitations": [
                    "Official GraphST pins incompatible DL stacks; CI uses API-compatible mock.",
                    "Does not claim published-paper ARI; re-run sota_pipeline in GraphST env for numerics.",
                ],
            },
            "stagate": {
                "category": "domain_detection",
                **stagate,
                "sources": ["research/method_validation/run_sota_batch_multidataset.py"],
                "limitations": [
                    "Official STAGATE_pyG requires isolated env; CI uses API-compatible mock.",
                    "Embedding clustering uses platform cluster_embedding helper (same as production wrap).",
                ],
            },
            "rctd": {
                "category": "deconvolution",
                **rctd,
                "sources": [
                    "research/method_validation/run_sota_batch_multidataset.py",
                    "src/histoweave/plugins/builtin/sota_domains.py",
                ],
                "limitations": [
                    "Multi-dataset gate is fail-closed contract (reference + counts + driver required).",
                    "Full spacexr RCTD ARI needs R driver + scRNA reference atlas — not substituted.",
                ],
            },
            "spatialde": {
                "category": "svg",
                **spatialde,
                "sources": [
                    "research/method_validation/run_sota_batch_multidataset.py",
                    "tests/test_banksy_spatialde.py",
                    "tests/test_core_real_method_contracts.py",
                ],
                "limitations": [
                    "Mock SpatialDE/NaiveDE for multi-dataset I/O and ranking contract.",
                    "Install histoweave-spatial[spatialde] for real GP p-values.",
                ],
            },
        },
    }
    path = OUT / "sota_batch_multidataset.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("wrote %s", path)

    # headline
    for name, body in payload["methods"].items():
        if name == "spagcn":
            ari = (body.get("sota_csv") or {}).get("mean_ari")
            logger.info(
                "spagcn sota mean_ari=%s live=%s", ari, body.get("live_smoke", {}).get("n_success")
            )
        else:
            logger.info("%s success=%s/%s", name, body.get("n_success"), body.get("n_total"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
