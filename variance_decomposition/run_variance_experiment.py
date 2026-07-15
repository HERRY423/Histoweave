"""Method-variance decomposition: how much does the *choice* of method matter?

A factorial experiment on DLPFC slice 151673 that isolates the four levers an analyst
turns when detecting spatial domains, and decomposes the variance in accuracy (ARI vs
manual layers) attributable to each:

    4 preprocessing  x  5 methods  x  3 parameter settings  x  3 subsamples  =  180 runs

Factors
-------
* **preprocessing** (4): log1p_cp10k, sqrt, scaled (z-score), arcsinh — applied to raw
  counts. These are the normalization choices that upstream every domain method.
* **method** (5): kmeans, gaussian_mixture, agglomerative, spectral, banksy_py.
* **param** (3): a method-appropriate "low / default / high" setting of one key
  hyper-parameter (documented in ``PARAM_GRID``).
* **subsample** (3): three random 80% spot subsamples (seeds 0/1/2).

Variance decomposition
----------------------
ARI ~ C(preprocessing) + C(method) + C(param) + C(subsample)  (Type-II ANOVA), reporting
each factor's share of the total sum of squares — i.e. "method choice explains X% of the
spread in achievable accuracy, preprocessing Y%, parameters Z%".

Outputs
-------
* ``variance_runs.csv``        — 180 rows (one per run) with ARI + metadata.
* ``variance_components.csv``  — per-factor variance share (SS, %, F, p).
* ``figures/fig_variance_decomposition.svg|png``
* ``report_variance.md``
"""

from __future__ import annotations

import logging
import time
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scanpy as sc

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402

from histoweave._math import adjusted_rand_index  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "case_study_dlpfc_consistency" / "data" / "151673.h5ad"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

PREPROCESSINGS = ["log1p_cp10k", "sqrt", "scaled", "arcsinh"]
METHODS = ["kmeans", "gaussian_mixture", "agglomerative", "spectral", "banksy_py"]
SUBSAMPLE_SEEDS = [0, 1, 2]
SUBSAMPLE_FRAC = 0.8

# One key spatial-context hyper-parameter per method, three levels (low/default/high).
# We vary the amount of spatial neighbourhood context each method uses — the single
# most consequential and *directly comparable* knob across the domain-detection family:
#   * sklearn methods (kmeans/gmm/agglomerative/spectral) -> `spatial_weight` on the
#     PCA+neighbourhood embedding (0 = expression only ... 0.6 = strong spatial mixing).
#   * banksy_py -> its native `lambda_param` (spatial weight in the BANKSY features).
PARAM_GRID: dict[str, list[dict]] = {
    "kmeans": [{"spatial_weight": 0.0}, {"spatial_weight": 0.3}, {"spatial_weight": 0.6}],
    "gaussian_mixture": [{"spatial_weight": 0.0}, {"spatial_weight": 0.3}, {"spatial_weight": 0.6}],
    "agglomerative": [{"spatial_weight": 0.0}, {"spatial_weight": 0.3}, {"spatial_weight": 0.6}],
    "spectral": [{"spatial_weight": 0.0}, {"spatial_weight": 0.3}, {"spatial_weight": 0.6}],
    "banksy_py": [{"lambda_param": 0.2}, {"lambda_param": 0.5}, {"lambda_param": 0.8}],
}
PARAM_LEVEL = ["low", "default", "high"]


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


def preprocess(counts: np.ndarray, kind: str) -> np.ndarray:
    """Apply one normalization variant to a raw-count matrix."""
    X = counts.astype(float)
    if kind == "log1p_cp10k":
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        return np.log1p(X / lib * 1e4)
    if kind == "sqrt":
        return np.sqrt(X)
    if kind == "scaled":  # log1p_cp10k then z-score per gene
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        L = np.log1p(X / lib * 1e4)
        mu = L.mean(axis=0, keepdims=True)
        sd = L.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (L - mu) / sd
    if kind == "arcsinh":
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        return np.arcsinh(X / lib * 1e4)
    raise ValueError(kind)


def load_counts():
    a = sc.read_h5ad(DATA)
    counts = a.layers["counts"]
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    truth = a.obs["domain_truth"].astype(str).to_numpy()
    coords = np.asarray(a.obsm["spatial"], dtype=float)
    obs_names = a.obs_names.astype(str).to_numpy()
    k = int(a.obs["domain_truth"].nunique())
    return counts.astype(float), truth, coords, obs_names, k


def make_method(method: str, param: dict, k: int, seed: int):
    kwargs = dict(param)
    if method == "kmeans":
        kwargs.update(n_domains=k, random_state=seed)
    elif method == "gaussian_mixture":
        kwargs.update(n_domains=k, random_state=seed)
    elif method == "agglomerative":
        kwargs.update(n_domains=k)
    elif method == "spectral":
        kwargs.update(n_domains=k, random_state=seed)
    elif method == "banksy_py":
        kwargs.update(n_domains=k, random_state=seed)
    return create_method(MethodCategory.DOMAIN_DETECTION, method, **kwargs)


def run() -> pd.DataFrame:
    counts, truth, coords, obs_names, k = load_counts()
    n = counts.shape[0]
    _log(f"[data] {n} spots, {counts.shape[1]} genes, k={k}")

    # pre-compute subsample index sets (shared across all factor combos)
    sub_idx = {}
    for s in SUBSAMPLE_SEEDS:
        rng = np.random.default_rng(1000 + s)
        sub_idx[s] = np.sort(rng.choice(n, size=int(SUBSAMPLE_FRAC * n), replace=False))

    # pre-compute each preprocessing once on the full matrix, then subset
    prep_full = {p: preprocess(counts, p) for p in PREPROCESSINGS}

    rows = []
    total = len(PREPROCESSINGS) * len(METHODS) * 3 * len(SUBSAMPLE_SEEDS)
    done = 0
    for prep, method, (pi, _param), seed in product(
        PREPROCESSINGS, METHODS, enumerate(range(3)), SUBSAMPLE_SEEDS
    ):
        param_dict = PARAM_GRID[method][pi]
        idx = sub_idx[seed]
        X = prep_full[prep][idx]
        tab = SpatialTable(
            X=X.copy(),
            obs=pd.DataFrame({"_i": np.arange(len(idx))}, index=obs_names[idx]),
            var=pd.DataFrame(index=[f"g{j}" for j in range(X.shape[1])]),
            obsm={"spatial": coords[idx]},
            uns={"n_domains": k},
        )
        t0 = time.time()
        try:
            m = make_method(method, param_dict, k, seed)
            r = m.run(tab)
            ari = adjusted_rand_index(truth[idx], r.obs["domain"].to_numpy())
            err = ""
        except Exception as exc:  # noqa: BLE001
            ari = np.nan
            err = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "preprocessing": prep,
                "method": method,
                "param": PARAM_LEVEL[pi],
                "param_value": str(param_dict),
                "subsample": f"sub{seed}",
                "ari": round(ari, 5) if np.isfinite(ari) else np.nan,
                "seconds": round(time.time() - t0, 3),
                "error": err,
            }
        )
        done += 1
        if done % 20 == 0 or done == total:
            _log(f"  {done}/{total} runs done")
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "variance_runs.csv", index=False)
    return df


def decompose(df: pd.DataFrame) -> pd.DataFrame:
    """Type-II ANOVA variance decomposition of ARI across the four factors."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    d = df.dropna(subset=["ari"]).copy()
    model = ols("ari ~ C(preprocessing) + C(method) + C(param) + C(subsample)", data=d).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    ss_total = anova["sum_sq"].sum()
    anova["pct_variance"] = 100 * anova["sum_sq"] / ss_total
    anova = anova.rename_axis("factor").reset_index()
    anova["factor"] = anova["factor"].str.replace(r"C\((.*)\)", r"\1", regex=True)
    anova.to_csv(BASE / "variance_components.csv", index=False)
    return anova


def make_figures(df: pd.DataFrame, comp: pd.DataFrame) -> None:
    d = df.dropna(subset=["ari"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A: variance share (excluding residual for the pie-like bar)
    order = ["method", "preprocessing", "param", "subsample", "Residual"]
    c = comp.set_index("factor").reindex(order)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#BAB0AC"]
    axes[0].bar(c.index, c["pct_variance"], color=colors)
    for i, v in enumerate(c["pct_variance"]):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("% of total variance in ARI")
    axes[0].set_title("Variance decomposition of ARI")
    plt.setp(axes[0].get_xticklabels(), rotation=20)

    # Panel B: ARI by method (the dominant factor, shown as box)
    m_order = d.groupby("method")["ari"].median().sort_values(ascending=False).index
    data_by_m = [d[d.method == m]["ari"].values for m in m_order]
    bp = axes[1].boxplot(data_by_m, tick_labels=m_order, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_alpha(0.6)
    axes[1].set_ylabel("ARI vs manual layers")
    axes[1].set_title("ARI spread by method")
    plt.setp(axes[1].get_xticklabels(), rotation=20)

    # Panel C: ARI by preprocessing
    p_order = d.groupby("preprocessing")["ari"].median().sort_values(ascending=False).index
    data_by_p = [d[d.preprocessing == p]["ari"].values for p in p_order]
    bp2 = axes[2].boxplot(data_by_p, tick_labels=p_order, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor("#F58518")
        patch.set_alpha(0.6)
    axes[2].set_ylabel("ARI vs manual layers")
    axes[2].set_title("ARI spread by preprocessing")
    plt.setp(axes[2].get_xticklabels(), rotation=20)

    fig.suptitle(
        "Method-variance decomposition — DLPFC 151673 "
        "(180 runs: 4 prep x 5 method x 3 param x 3 subsample)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "fig_variance_decomposition.svg")
    fig.savefig(FIG / "fig_variance_decomposition.png", dpi=130)
    plt.close(fig)


def main() -> None:
    df = run()
    n_ok = df["ari"].notna().sum()
    _log(f"[runs] {len(df)} total, {n_ok} succeeded")
    comp = decompose(df)
    _log(comp[["factor", "sum_sq", "pct_variance", "F", "PR(>F)"]].to_string(index=False))
    make_figures(df, comp)
    _log("=== variance experiment DONE ===")


if __name__ == "__main__":
    main()
