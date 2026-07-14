"""Tutorial 1 companion — end-to-end analysis of a real Visium DLPFC slide.

    python examples/tutorial_real_visium.py

Downloads one DLPFC Visium slice (cached under ~/.cache/histoweave/datasets), runs QC ->
normalize -> Harmony -> scANVI -> SVG -> domains, and writes an interactive report.
Optional steps degrade gracefully when their extra/container is missing.
See docs/tutorials/01_real_visium_dlpfc.md.
"""

from __future__ import annotations

import numpy as np

import histoweave as ts
from histoweave.datasets import get_dataset
from histoweave.plugins import create_method


def main() -> None:
    print(f"HistoWeave v{ts.__version__}\n")

    try:
        entry = get_dataset("dlpfc_151507")
        data = entry.load()
    except Exception as exc:  # network / cache miss
        print("Could not load the real DLPFC slide (no network?):", exc)
        print("Falling back to a synthetic Visium-like dataset for the walkthrough.")
        data = ts.datasets.make_synthetic(n_cells=800, n_genes=200, n_domains=6, seed=0)

    print("Loaded:", repr(data))

    # QC + normalization
    data = create_method("qc", "basic_qc").run(data)
    data = create_method("normalization", "log1p_cp10k").run(data)

    # Harmony (single-batch here -> pass-through; real power is multi-slide)
    batch_col = "sample_id" if "sample_id" in data.obs else data.obs.columns[0]
    try:
        data = create_method(
            "integration", "harmony", batch_key=batch_col, n_pcs=30
        ).run(data)
        print("Harmony embedding:", data.obsm["X_pca_harmony"].shape)
    except ModuleNotFoundError as exc:
        print("Harmony skipped (install extra 'harmony'):", exc)

    # scANVI annotation from partial ground-truth labels, when available
    if "spatialLIBD_layer" in data.obs:
        rng = np.random.default_rng(0)
        seed = data.obs["spatialLIBD_layer"].astype("object").to_numpy().copy()
        seed[rng.random(data.n_obs) > 0.30] = "Unknown"
        data.obs["cell_type_seed"] = seed
        try:
            data = create_method(
                "annotation", "scanvi",
                labels_key="cell_type_seed", unlabeled_category="Unknown",
                layer="counts", scvi_epochs=50, scanvi_epochs=25,
            ).run(data)
            print("scANVI labels:", data.obs["cell_type"].value_counts().to_dict())
        except (ModuleNotFoundError, ImportError) as exc:
            print("scANVI skipped (install extra 'scanvi'):", exc)

    # SVG: prefer nnSVG (R container), fall back to Moran's I
    try:
        data = create_method("svg", "nnsvg", n_top=50).run(data)
        print("Top nnSVG genes:", list(data.var.sort_values("nnsvg_rank").head(5).index))
    except (RuntimeError, FileNotFoundError) as exc:
        print("nnSVG unavailable, using Moran's I:", exc)
        data = create_method("svg", "morans_i", n_top=50).run(data)

    # Domains + report
    data = create_method("domain_detection", "banksy", n_domains=7).run(data)
    out = ts.build_report(data, "dlpfc_report.html")
    print("Report written to", out)


if __name__ == "__main__":
    main()
