"""DLPFC cross-method consistency case study (BANKSY vs GMM vs scANVI).

Runs three complementary spatial-domain / annotation methods on DLPFC slice 151673
(spatialLIBD manual layers, Maynard et al. 2021) and quantifies where each method
*misses layer boundaries* that the others capture — the boundary spots are the
biologically hard, clinically interesting transition zones.

Methods
-------
* **banksy_py** — neighbourhood-augmented spatial domains (native BANKSY scaffold).
* **gaussian_mixture** — expression-only soft clustering (no spatial term).
* **scanvi** — semi-supervised annotation trained on a 20% random seed of the
  manual layer labels (the realistic "partial annotation" regime scANVI targets).

Outputs (written next to this script)
------------------------------------
* ``results/consistency_metrics.csv`` — per-method ARI / NMI / boundary error.
* ``results/boundary_missed.csv`` — boundary spots missed uniquely by one method.
* ``results/per_spot.csv`` — aligned per-spot predictions + agreement mask.
* ``vitessce_config.json`` — self-contained interactive comparison view.
* ``figures/*.svg|png`` — layer comparison, boundary-miss bars, agreement map.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from histoweave._math import knn_indices  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "151673.h5ad"
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 0
LAYER_ORDER = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6", "WM"]


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def load() -> tuple[SpatialTable, ad.AnnData]:
    a = sc.read_h5ad(DATA)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    tab = SpatialTable(
        X=X.astype(float),
        obs=pd.DataFrame(
            {"domain_truth": a.obs["domain_truth"].astype(str).values},
            index=a.obs_names.astype(str),
        ),
        var=pd.DataFrame(index=a.var_names.astype(str)),
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=float)},
        uns={"n_domains": int(a.obs["domain_truth"].nunique()), "assay": "visium"},
    )
    return tab, a


def align_labels(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Map predicted cluster ids onto truth labels by Hungarian assignment."""
    t_lab, t_idx = np.unique(truth, return_inverse=True)
    p_lab, p_idx = np.unique(pred, return_inverse=True)
    cont = np.zeros((len(t_lab), len(p_lab)), dtype=int)
    np.add.at(cont, (t_idx, p_idx), 1)
    # maximize overlap -> minimize negative
    row, col = linear_sum_assignment(-cont)
    mapping = {p_lab[c]: t_lab[r] for r, c in zip(row, col, strict=True)}
    # any unmapped predicted label -> its argmax truth
    out = np.empty(len(pred), dtype=object)
    for i, p in enumerate(pred):
        if p in mapping:
            out[i] = mapping[p]
        else:
            out[i] = t_lab[cont[:, list(p_lab).index(p)].argmax()]
    return out.astype(str)


def boundary_mask(coords: np.ndarray, truth: np.ndarray, k: int = 6) -> np.ndarray:
    """A spot is a boundary spot if any of its k spatial neighbours has a
    different ground-truth layer."""
    idx = knn_indices(coords, k + 1)[:, 1:]
    nbr_truth = truth[idx]
    return (nbr_truth != truth[:, None]).any(axis=1)


def run_methods(tab: SpatialTable, a: ad.AnnData) -> dict[str, np.ndarray]:
    """Return aligned per-spot predictions for each method."""
    truth = tab.obs["domain_truth"].to_numpy()
    k = tab.uns["n_domains"]
    preds: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    # BANKSY (spatial)
    t0 = time.time()
    r = create_method(
        MethodCategory.DOMAIN_DETECTION, "banksy_py", n_domains=k, lambda_param=0.8
    ).run(tab.copy())
    timings["banksy_py"] = time.time() - t0
    preds["banksy_py"] = align_labels(truth, r.obs["domain"].to_numpy())

    # GMM (expression only)
    t0 = time.time()
    r = create_method(
        MethodCategory.DOMAIN_DETECTION, "gaussian_mixture", n_domains=k, random_state=SEED
    ).run(tab.copy())
    timings["gaussian_mixture"] = time.time() - t0
    preds["gaussian_mixture"] = align_labels(truth, r.obs["domain"].to_numpy())

    # scANVI (semi-supervised, 20% seed labels)
    t0 = time.time()
    preds["scanvi"] = run_scanvi(a, truth, seed_frac=0.2)
    timings["scanvi"] = time.time() - t0

    globals()["_TIMINGS"] = timings
    return preds


def run_scanvi(a: ad.AnnData, truth: np.ndarray, seed_frac: float = 0.2) -> np.ndarray:
    import scvi

    scvi.settings.seed = SEED
    adata = ad.AnnData(X=a.layers["counts"].copy(), obs=a.obs.copy(), var=a.var.copy())
    adata.obs_names = a.obs_names.astype(str)
    rng = np.random.default_rng(SEED)
    seed_labels = np.array(truth, dtype=object)
    hide = rng.random(len(truth)) > seed_frac
    seed_labels[hide] = "Unknown"
    adata.obs["cell_type_seed"] = pd.Categorical(seed_labels)

    scvi.model.SCVI.setup_anndata(adata, labels_key="cell_type_seed")
    vae = scvi.model.SCVI(adata, n_latent=20, n_layers=2)
    vae.train(max_epochs=50, accelerator="cpu")
    model = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category="Unknown")
    model.train(max_epochs=25, accelerator="cpu")
    return np.asarray(model.predict()).astype(str)


def compute_metrics(
    truth: np.ndarray, preds: dict[str, np.ndarray], is_boundary: np.ndarray
) -> pd.DataFrame:
    rows = []
    for method, pred in preds.items():
        overall_acc = float((pred == truth).mean())
        b_acc = float((pred[is_boundary] == truth[is_boundary]).mean())
        i_acc = float((pred[~is_boundary] == truth[~is_boundary]).mean())
        rows.append(
            {
                "method": method,
                "ari": round(adjusted_rand_score(truth, pred), 4),
                "nmi": round(normalized_mutual_info_score(truth, pred), 4),
                "accuracy": round(overall_acc, 4),
                "boundary_accuracy": round(b_acc, 4),
                "interior_accuracy": round(i_acc, 4),
                "boundary_error": round(1 - b_acc, 4),
                "seconds": round(globals().get("_TIMINGS", {}).get(method, float("nan")), 2),
            }
        )
    return pd.DataFrame(rows)


def unique_boundary_misses(
    truth: np.ndarray, preds: dict[str, np.ndarray], is_boundary: np.ndarray
) -> pd.DataFrame:
    """For each method, count boundary spots that IT misses but the OTHER two get
    right — the domains a single method uniquely fails to resolve."""
    methods = list(preds)
    correct = {m: (preds[m] == truth) for m in methods}
    rows = []
    for m in methods:
        others = [o for o in methods if o != m]
        others_right = np.logical_and.reduce([correct[o] for o in others])
        uniquely_missed = is_boundary & (~correct[m]) & others_right
        # break down by which layer-pair boundary
        rows.append(
            {
                "method": m,
                "boundary_spots": int(is_boundary.sum()),
                "boundary_missed_total": int((is_boundary & ~correct[m]).sum()),
                "uniquely_missed_boundary": int(uniquely_missed.sum()),
                "unique_miss_pct_of_boundary": round(
                    100 * uniquely_missed.sum() / max(is_boundary.sum(), 1), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    coords: np.ndarray,
    truth: np.ndarray,
    preds: dict[str, np.ndarray],
    is_boundary: np.ndarray,
    metrics: pd.DataFrame,
    miss: pd.DataFrame,
) -> None:
    palette = plt.get_cmap("tab10")
    lab2col = {lab: palette(i) for i, lab in enumerate(LAYER_ORDER)}

    def scatter(ax, labels, title):
        for lab in LAYER_ORDER:
            m = labels == lab
            if m.any():
                ax.scatter(coords[m, 1], -coords[m, 0], s=4, color=lab2col[lab], label=lab)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    # Fig 1: spatial layer comparison (truth + 3 methods)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    scatter(axes[0], truth, "Manual layers (truth)")
    scatter(
        axes[1],
        preds["banksy_py"],
        f"BANKSY  ARI={metrics.set_index('method').loc['banksy_py', 'ari']:.3f}",
    )
    scatter(
        axes[2],
        preds["gaussian_mixture"],
        f"GMM  ARI={metrics.set_index('method').loc['gaussian_mixture', 'ari']:.3f}",
    )
    scatter(
        axes[3],
        preds["scanvi"],
        f"scANVI  ARI={metrics.set_index('method').loc['scanvi', 'ari']:.3f}",
    )
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=7, fontsize=8, markerscale=2)
    fig.suptitle("DLPFC 151673 — cross-method layer comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(FIGURES / "fig1_layer_comparison.svg")
    fig.savefig(FIGURES / "fig1_layer_comparison.png", dpi=130)
    plt.close(fig)

    # Fig 2: boundary vs interior accuracy + unique boundary misses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    mi = metrics.set_index("method").loc[list(preds)]
    x = np.arange(len(mi))
    ax1.bar(x - 0.2, mi["interior_accuracy"], 0.4, label="interior", color="#4C78A8")
    ax1.bar(x + 0.2, mi["boundary_accuracy"], 0.4, label="boundary", color="#E45756")
    ax1.set_xticks(x)
    ax1.set_xticklabels(mi.index, rotation=15)
    ax1.set_ylabel("accuracy vs manual layer")
    ax1.set_title("Interior vs boundary accuracy")
    ax1.legend()
    ms = miss.set_index("method").loc[list(preds)]
    ax2.bar(ms.index, ms["uniquely_missed_boundary"], color="#F58518")
    for i, v in enumerate(ms["uniquely_missed_boundary"]):
        ax2.text(
            i,
            v,
            f"{v}\n({ms['unique_miss_pct_of_boundary'].iloc[i]:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_ylabel("boundary spots missed uniquely")
    ax2.set_title("Boundaries missed by ONE method\n(other two correct)")
    plt.setp(ax2.get_xticklabels(), rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_boundary_misses.svg")
    fig.savefig(FIGURES / "fig2_boundary_misses.png", dpi=130)
    plt.close(fig)

    # Fig 3: cross-method agreement map
    agree = np.sum([preds[m] == truth for m in preds], axis=0)  # 0..3 correct
    fig, ax = plt.subplots(figsize=(5.2, 5))
    cmap = plt.get_cmap("RdYlGn", 4)
    sc_ = ax.scatter(coords[:, 1], -coords[:, 0], c=agree, s=6, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax.set_title("Cross-method agreement\n(# of 3 methods correct per spot)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    cb = fig.colorbar(sc_, ax=ax, ticks=[0, 1, 2, 3])
    cb.set_label("methods correct")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_agreement_map.svg")
    fig.savefig(FIGURES / "fig3_agreement_map.png", dpi=130)
    plt.close(fig)


def build_vitessce(tab: SpatialTable, truth, preds, is_boundary):
    """Attach method predictions + agreement to obs and build a Vitessce config
    with the enhanced linked-views (task 3) infrastructure."""
    from histoweave.report.vitessce_data import build_vitessce_view_config

    t = tab.copy()
    t.obs["manual_layer"] = pd.Categorical(truth)
    t.obs["banksy"] = pd.Categorical(preds["banksy_py"])
    t.obs["gmm"] = pd.Categorical(preds["gaussian_mixture"])
    t.obs["scanvi"] = pd.Categorical(preds["scanvi"])
    agree = np.sum([preds[m] == truth for m in preds], axis=0)
    t.obs["n_methods_correct"] = pd.Categorical([f"{v}/3" for v in agree])
    t.obs["is_boundary"] = pd.Categorical(np.where(is_boundary, "boundary", "interior"))
    # give the heatmap real marker genes to show
    t.uns["svg"] = {"top_genes": [{"gene": g} for g in list(t.var.index[:30])]}
    vc = build_vitessce_view_config(t, top_genes=30)
    return vc


def main() -> None:
    tab, a = load()
    coords = np.asarray(tab.spatial, dtype=float)
    truth = tab.obs["domain_truth"].to_numpy()
    is_boundary = boundary_mask(coords, truth, k=6)
    _log(
        f"[data] {tab.n_obs} spots, {tab.n_vars} genes, "
        f"{is_boundary.sum()} boundary spots ({100 * is_boundary.mean():.1f}%)"
    )

    preds = run_methods(tab, a)
    metrics = compute_metrics(truth, preds, is_boundary)
    miss = unique_boundary_misses(truth, preds, is_boundary)
    _log(metrics.to_string(index=False))
    _log(miss.to_string(index=False))

    metrics.to_csv(RESULTS / "consistency_metrics.csv", index=False)
    miss.to_csv(RESULTS / "boundary_missed.csv", index=False)

    per_spot = pd.DataFrame(
        {
            "obs_id": tab.obs.index.to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "manual_layer": truth,
            "banksy": preds["banksy_py"],
            "gmm": preds["gaussian_mixture"],
            "scanvi": preds["scanvi"],
            "is_boundary": is_boundary,
            "n_methods_correct": np.sum([preds[m] == truth for m in preds], axis=0),
        }
    )
    per_spot.to_csv(RESULTS / "per_spot.csv", index=False)

    make_figures(coords, truth, preds, is_boundary, metrics, miss)

    vc = build_vitessce(tab, truth, preds, is_boundary)
    (BASE / "vitessce_config.json").write_text(json.dumps(vc, indent=2, default=str))
    # also emit a standalone interactive report using the enhanced template
    from histoweave.report.report import build_report

    t = tab.copy()
    t.obs["manual_layer"] = pd.Categorical(truth)
    t.obs["banksy"] = pd.Categorical(preds["banksy_py"])
    t.obs["gmm"] = pd.Categorical(preds["gaussian_mixture"])
    t.obs["scanvi"] = pd.Categorical(preds["scanvi"])
    t.obs["domain"] = pd.Categorical(preds["banksy_py"])
    t.uns["svg"] = {"top_genes": [{"gene": g} for g in list(t.var.index[:30])]}
    t.uns["assay"] = "visium (DLPFC 151673)"
    build_report(t, BASE / "case_study_report.html")

    summary = {
        "slice": "151673",
        "n_spots": int(tab.n_obs),
        "n_boundary_spots": int(is_boundary.sum()),
        "metrics": metrics.to_dict(orient="records"),
        "boundary_misses": miss.to_dict(orient="records"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _log("\n=== case study DONE ===")


if __name__ == "__main__":
    main()
