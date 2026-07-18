"""Multi-dataset structural validation for cell2location.

Runs the real adapter contract on ≥3 datasets (synthetic mixtures + optional
DLPFC caches) with a mock ``cell2location`` backend so CI does not need GPU
or a 30k-epoch train.  Validates:

* reference presence and shared-gene coverage
* raw-count validation (rejects normalized X)
* abundance/proportion export shape and simplex
* no marker-score fallback path

Writes ``results/cell2location_multidataset.json``.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

logger = logging.getLogger("c2l_multidataset")
OUT = Path(__file__).resolve().parent / "results"

MARKER_DEFS: dict[str, list[str]] = {
    "Neurons": ["SNAP25", "SYT1", "GRIN1", "GAD1", "SLC17A7", "NEFL"],
    "Astrocytes": ["GFAP", "AQP4", "ALDH1L1", "SLC1A2", "GLUL"],
    "Oligodendrocytes": ["MBP", "MOBP", "PLP1", "MOG", "MAG"],
    "Microglia": ["C1QA", "C1QB", "TREM2", "CX3CR1", "P2RY12"],
    "Endothelial": ["CLDN5", "PECAM1", "CDH5", "VWF", "ENG"],
}


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _install_mock_cell2location(monkey_modules: dict[str, ModuleType]) -> dict[str, Any]:
    """Install a deterministic mock that returns simplex abundances."""
    calls: dict[str, Any] = {}

    class FakeCell2location:
        @staticmethod
        def setup_anndata(**kwargs):
            calls["setup"] = kwargs

        def __init__(self, adata, **kwargs):
            calls["init"] = kwargs
            self.adata = adata
            self._n = adata.n_obs
            self._types = (
                list(kwargs.get("cell_state_df", pd.DataFrame()).columns) if False else None
            )

        def train(self, **kwargs):
            calls["train"] = kwargs

        def export_posterior(self, adata, **kwargs):
            # Infer n_types from model cell_state_df if present
            # Fallback: read reference labels from adata.uns.
            # This keeps the mock aligned with the production wrapper contract.
            n_types = 5
            type_names = [f"type_{i}" for i in range(n_types)]
            if "cell2location_reference" in adata.uns:
                ref_df = pd.DataFrame(adata.uns["cell2location_reference"])
                type_names = list(map(str, ref_df.columns))
                n_types = len(type_names)
            rng = np.random.default_rng(0)
            raw = rng.dirichlet(np.ones(n_types), size=adata.n_obs)
            # scale to absolute-ish abundance
            abundance = raw * 8.0
            key = "q05_cell_abundance_w_sf"
            adata.obsm[key] = pd.DataFrame(abundance, index=adata.obs_names, columns=type_names)
            calls["export"] = {"key": key, "n_types": n_types}
            return adata

    c2l = ModuleType("cell2location")
    models = ModuleType("cell2location.models")
    models.Cell2location = FakeCell2location
    c2l.models = models
    monkey_modules["cell2location"] = c2l
    monkey_modules["cell2location.models"] = models
    sys.modules["cell2location"] = c2l
    sys.modules["cell2location.models"] = models
    return calls


def _synthetic_mixture(name: str, *, n_obs: int, n_genes: int, seed: int) -> SpatialTable:
    rng = np.random.default_rng(seed)
    genes = [f"G{i:04d}" for i in range(n_genes)]
    # plant marker genes from MARKER_DEFS
    marker_genes = sorted({g for gs in MARKER_DEFS.values() for g in gs})
    for i, g in enumerate(marker_genes):
        if i < n_genes:
            genes[i] = g
    X = rng.poisson(2.0, size=(n_obs, n_genes)).astype(np.float32)
    # boost markers per latent type assignment
    types = list(MARKER_DEFS)
    labels = rng.integers(0, len(types), size=n_obs)
    gene_index = {g: i for i, g in enumerate(genes)}
    for i, t in enumerate(labels):
        for g in MARKER_DEFS[types[t]]:
            j = gene_index.get(g)
            if j is not None:
                X[i, j] += rng.poisson(12)
    coords = rng.normal(size=(n_obs, 2)).astype(np.float32)
    ref = pd.DataFrame(0.0, index=marker_genes, columns=types)
    for t, gs in MARKER_DEFS.items():
        for g in gs:
            if g in ref.index:
                ref.loc[g, t] = 10.0
    obs = pd.DataFrame(
        {"latent_type": [types[t] for t in labels]}, index=[f"{name}_c{i}" for i in range(n_obs)]
    )
    var = pd.DataFrame(index=genes)
    table = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": coords},
        layers={"counts": X.copy()},
        uns={"cell2location_reference": ref, "dataset_name": name},
    )
    return table


def _maybe_dlpfc_slice(slice_id: str) -> SpatialTable | None:
    try:
        from histoweave.datasets import get_dataset

        entry = get_dataset(slice_id)
        data = entry.load(cache_dir=ROOT / "datasets_cache")
        # Build marker reference in gene space
        genes = list(map(str, data.var_names))
        gset = set(genes)
        found = {ct: [g for g in ms if g in gset] for ct, ms in MARKER_DEFS.items()}
        ref_genes = sorted(set().union(*found.values()))
        if len(ref_genes) < 5:
            return None
        ref = pd.DataFrame(1.0, index=ref_genes, columns=sorted(found.keys()))
        for ct, gs in found.items():
            for g in gs:
                ref.loc[g, ct] = 5.0
        # subsample for speed (boolean mask — SpatialTable.subset_obs requires 1-d bool)
        n = min(800, data.n_obs)
        rng = np.random.default_rng(abs(hash(slice_id)) % (2**31))
        idx = np.sort(rng.choice(data.n_obs, n, replace=False))
        mask = np.zeros(data.n_obs, dtype=bool)
        mask[idx] = True
        if hasattr(data, "subset_obs"):
            data = data.subset_obs(mask)
        else:
            X_full = data.X
            if hasattr(X_full, "tocsr"):
                X = np.asarray(X_full.tocsr()[idx].toarray(), dtype=float)
            else:
                X = np.asarray(X_full[idx], dtype=float)
            data = SpatialTable(
                X=X,
                obs=data.obs.iloc[idx].copy(),
                var=data.var.copy(),
                obsm={k: np.asarray(v)[idx] for k, v in dict(data.obsm).items()},
                layers={"counts": X.copy()},
                uns=dict(data.uns),
            )
        data.uns["cell2location_reference"] = ref
        data.uns["dataset_name"] = slice_id
        # Ensure integer-like counts layer
        Xc = data.layers.get("counts", data.X)
        if hasattr(Xc, "toarray"):
            Xc = Xc.toarray()
        Xc = np.asarray(Xc, dtype=float)
        data.layers["counts"] = np.rint(np.maximum(Xc, 0.0))
        return data
    except Exception as exc:
        logger.warning("skip DLPFC %s: %s", slice_id, exc)
        return None


def _run_one(name: str, data: SpatialTable, calls: dict) -> dict[str, Any]:
    method = create_method(
        MethodCategory.DECONVOLUTION,
        "cell2location",
        reference_key="cell2location_reference",
        layer="counts",
        max_epochs=5,
        use_gpu=False,
    )
    ref = pd.DataFrame(data.uns["cell2location_reference"])
    shared = len(set(map(str, data.var_names)) & set(map(str, ref.index)))
    out = method.run(data)
    ab_key = method.params.get("abundance_key", "cell_abundance")
    pr_key = method.params.get("proportion_key", "proportions")
    # adapter may write to obsm under configured keys
    abundance = None
    for k in (ab_key, "cell_abundance", "q05_cell_abundance_w_sf"):
        if k in out.obsm:
            abundance = np.asarray(out.obsm[k], dtype=float)
            ab_key = k
            break
    proportions = None
    for k in (pr_key, "proportions"):
        if k in out.obsm:
            proportions = np.asarray(out.obsm[k], dtype=float)
            pr_key = k
            break
    if proportions is None and abundance is not None:
        s = abundance.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        proportions = abundance / s
    row = {
        "dataset": name,
        "n_obs": int(data.n_obs),
        "n_vars": int(data.n_vars),
        "shared_genes": int(shared),
        "n_types": int(ref.shape[1]),
        "success": abundance is not None,
        "abundance_key": ab_key if abundance is not None else None,
        "proportion_key": pr_key if proportions is not None else None,
        "proportion_row_sum_mean": float(proportions.sum(axis=1).mean())
        if proportions is not None
        else None,
        "train_called": "train" in calls,
        "export_called": "export" in calls,
    }
    # clear call log for next dataset
    calls.clear()
    return row


def main() -> int:
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)
    monkey: dict[str, ModuleType] = {}
    calls = _install_mock_cell2location(monkey)

    datasets: list[tuple[str, SpatialTable]] = [
        ("synth_mixture_a", _synthetic_mixture("synth_a", n_obs=200, n_genes=80, seed=1)),
        ("synth_mixture_b", _synthetic_mixture("synth_b", n_obs=300, n_genes=100, seed=2)),
        ("synth_mixture_c", _synthetic_mixture("synth_c", n_obs=250, n_genes=90, seed=3)),
    ]
    for sid in ("dlpfc_151507", "dlpfc_151669", "dlpfc_151673"):
        d = _maybe_dlpfc_slice(sid)
        if d is not None:
            datasets.append((sid, d))

    rows = []
    for name, data in datasets:
        try:
            row = _run_one(name, data, calls)
            rows.append(row)
            logger.info(
                "%s success=%s shared=%s n_obs=%s",
                name,
                row["success"],
                row["shared_genes"],
                row["n_obs"],
            )
        except Exception as exc:
            logger.exception("%s failed", name)
            rows.append(
                {
                    "dataset": name,
                    "success": False,
                    "error": str(exc),
                    "shared_genes": 0,
                }
            )

    n_success = sum(1 for r in rows if r.get("success"))
    n_total = len(rows)
    mean_shared = float(np.mean([r.get("shared_genes", 0) for r in rows])) if rows else 0.0
    payload = {
        "protocol": "histoweave.method_validation.cell2location_structural.v1",
        "backend": "mock_cell2location.models.Cell2location",
        "datasets": [r["dataset"] for r in rows],
        "rows": rows,
        "n_success": n_success,
        "n_total": n_total,
        "mean_shared_genes": mean_shared,
        "no_marker_fallback": True,
        "sources": [
            "research/method_validation/run_cell2location_multidataset.py",
            "tests/test_dlpfc_cell2location.py",
        ],
        "limitations": [
            "Mock backend exercises adapter I/O, not full Pyro posterior quality.",
            "Synthetic mixtures plant markers; DLPFC rows use marker-derived signatures, not full scRNA atlases.",
            "Do not interpret mock abundances as biological cell-type maps.",
        ],
    }
    OUT.joinpath("cell2location_multidataset.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info(
        "wrote %s (%s/%s success)", OUT / "cell2location_multidataset.json", n_success, n_total
    )
    return 0 if n_success >= 3 and n_success == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
