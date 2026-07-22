"""Evaluate BANKSY-Python on the strict Xenium lymph-node unit.

This uses the same deterministic 6,000-cell stratified sample as
``scripts/expand_real_independent_studies.py`` and the same adapter used by the
DLPFC and five-study external benchmarks.  The result closes the BANKSY cell
for the added strict-panel unit; it does not imply coverage for other SOTA
backends.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = ROOT / "5x15_spatial_aware"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from adapters import banksy_py_adapter  # noqa: E402

DATA = ROOT / "datasets_cache" / "xenium" / "xenium_human_lymph_node.h5ad"
OUT = ROOT / "benchmark_external_validation" / "strict_external_panel_v2"
SEEDS = (42, 1, 2)
MAX_CELLS = 6000
SUBSAMPLE_SEED = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subsample_indices(labels: np.ndarray) -> np.ndarray:
    if len(labels) <= MAX_CELLS:
        return np.arange(len(labels))
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    keep: list[np.ndarray] = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        quota = max(1, int(round(len(idx) / len(labels) * MAX_CELLS)))
        keep.append(rng.choice(idx, size=min(len(idx), quota), replace=False))
    merged = np.unique(np.concatenate(keep))
    if len(merged) > MAX_CELLS:
        merged = np.sort(rng.choice(merged, size=MAX_CELLS, replace=False))
    elif len(merged) < MAX_CELLS:
        rest = np.setdiff1d(np.arange(len(labels)), merged)
        extra = rng.choice(rest, size=min(MAX_CELLS - len(merged), len(rest)), replace=False)
        merged = np.sort(np.concatenate([merged, extra]))
    return merged


def main() -> None:
    import anndata as ad

    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    adata = ad.read_h5ad(DATA)
    truth_all = adata.obs["domain_truth"].astype(str).to_numpy()
    selected = _subsample_indices(truth_all)
    subset = adata[selected].copy()
    truth = subset.obs["domain_truth"].astype(str).to_numpy()
    n_domains = int(pd.Series(truth).nunique())
    counts = subset.layers.get("counts", subset.X)
    spatial = np.asarray(subset.obsm["spatial"], dtype=float)

    rows = []
    for seed in SEEDS:
        started = time.perf_counter()
        labels = banksy_py_adapter.run(
            counts,
            spatial,
            seed=seed,
            n_domains=n_domains,
        )
        rows.append(
            {
                "dataset": "xenium_human_lymph_node",
                "method": "banksy_py",
                "seed": seed,
                "ari": float(adjusted_rand_score(truth, labels)),
                "seconds": float(time.perf_counter() - started),
                "status": "success",
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "banksy_lymph_long.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    values = [row["ari"] for row in rows]
    summary = {
        "schema_version": "histoweave.strict_external_panel.banksy_lymph.v1",
        "dataset": "xenium_human_lymph_node",
        "method": "banksy_py",
        "n_obs": int(len(selected)),
        "n_domains": n_domains,
        "seeds": list(SEEDS),
        "subsample_seed": SUBSAMPLE_SEED,
        "sampling": "stratified by domain_truth to 6000 cells",
        "mean_ari": float(np.mean(values)),
        "std_ari": float(np.std(values, ddof=1)),
        "n_success": len(rows),
        "source_data_sha256": _sha256(DATA),
        "rows": rows,
    }
    json_path = OUT / "banksy_lymph_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "input": {"path": DATA.relative_to(ROOT).as_posix(), "sha256": _sha256(DATA)},
        "artifacts": {
            csv_path.name: {"sha256": _sha256(csv_path), "bytes": csv_path.stat().st_size},
            json_path.name: {"sha256": _sha256(json_path), "bytes": json_path.stat().st_size},
        },
    }
    (OUT / "banksy_lymph_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    logging.getLogger(__name__).info("BANKSY lymph-node mean ARI=%.6f", summary["mean_ari"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
