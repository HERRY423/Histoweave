"""Pre-registered panel validation + IF-ready ROI export.

Follows COMPONENT_COMPARISON.md upgrade path:

1. **L3 program** (neuronal / mid-layer): ENC1, HOPX, GAP43, GRIA2, CARTPT
2. **Myelin program**: MBP, PLP1, MOBP
3. Test on known components with **same-layer** contrast (critical) + whole-slice contrast
4. Spatial-shift nulls on composite scores
5. Export ROIs for IF (coordinates + spot barcodes when available)
6. Spatial score maps (SVG/PNG)

Components analysed
-------------------
* dlpfc_151508 largest  (L6, n=154)  — expect myelin ↑ vs rest; weak vs L6
* dlpfc_151508 rank-1   (L3, n=138)  — expect L3-program ↑, myelin ↓
* dlpfc_151669 largest  (L3, n=137)  — same direction as L3 on 151508

Gates (pre-registered before reading new outputs)
-------------------------------------------------
* L3 niche: composite L3 z-score mean(in) − mean(same-layer out) > 0 and
  shift-null p ≤ 0.05; myelin composite delta < 0 preferred.
* L6 niche: myelin composite delta vs rest > 0 and shift-null p ≤ 0.05;
  same-layer (vs L6) may be non-significant (intra-layer).
* Cross-donor L3: both 151508-rank1 and 151669-largest pass L3 direction.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.datasets import get_dataset  # noqa: E402

logger = logging.getLogger("validate_panel")

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "panel_validation"

# Pre-registered panels (do not expand after seeing results for primary claim).
L3_PANEL = ("ENC1", "HOPX", "GAP43", "GRIA2", "CARTPT")
MYELIN_PANEL = ("MBP", "PLP1", "MOBP")
# Secondary / exploratory only (reported separately).
EXPLORATORY = ("GFAP", "SST", "SAA1", "SCGB2A2")

COMPONENTS: list[dict[str, Any]] = [
    {
        "slice_id": "dlpfc_151508",
        "tag": "largest_component",
        "label": "151508_L6_n154",
        "expected_layer": "Layer 6",
        "expected_class": "L6_myelin",
    },
    {
        "slice_id": "dlpfc_151508",
        "tag": "component_rank1_n138",
        "label": "151508_L3_n138",
        "expected_layer": "Layer 3",
        "expected_class": "L3_program",
    },
    {
        "slice_id": "dlpfc_151669",
        "tag": "largest_component",
        "label": "151669_L3_n137",
        "expected_layer": "Layer 3",
        "expected_class": "L3_program",
    },
    # Third donor (Maynard DLPFC): largest cryptic = pure L3 (n=47);
    # rank-1 = pure L6 (n=24) for L6-class direction check.
    {
        "slice_id": "dlpfc_151673",
        "tag": "largest_component",
        "label": "151673_L3_n47",
        "expected_layer": "Layer 3",
        "expected_class": "L3_program",
    },
    {
        "slice_id": "dlpfc_151673",
        "tag": "component_rank1_n24",
        "label": "151673_L6_n24",
        "expected_layer": "Layer 6",
        "expected_class": "L6_myelin",
    },
]

N_SHIFT = 199
SEED = 0


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def log1p_norm(X: np.ndarray) -> np.ndarray:
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * 1e4)


def rank_sum_p(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return 1.0
    combined = np.concatenate([x, y])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[order[j + 1]] == combined[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    _, counts = np.unique(combined, return_counts=True)
    tie = (counts**3 - counts).sum() / (len(combined) * (len(combined) - 1))
    sigma2 = n1 * n2 / 12.0 * ((n1 + n2 + 1) - tie)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    z = (u1 - mu - 0.5 * np.sign(u1 - mu)) / sigma
    return float(erfc(abs(z) / sqrt(2.0)))


def shift_null_delta(
    coords: np.ndarray,
    scores: np.ndarray,
    in_mask: np.ndarray,
    *,
    n_null: int = N_SHIFT,
    seed: int = SEED,
) -> tuple[float, float]:
    """Observed mean(in)−mean(out) and one-sided shift-null p (greater)."""
    scores = np.asarray(scores, dtype=float)
    in_mask = np.asarray(in_mask, dtype=bool)
    out = ~in_mask
    if in_mask.sum() < 5 or out.sum() < 5:
        return float("nan"), float("nan")
    order = np.argsort(coords[:, 0] + 1e-3 * coords[:, 1])
    s = scores[order]
    m = in_mask[order]
    obs = float(s[m].mean() - s[~m].mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    n = len(s)
    for _ in range(n_null):
        shift = int(rng.integers(1, n))
        rolled = np.roll(s, shift)
        stat = float(rolled[m].mean() - rolled[~m].mean())
        extreme += int(stat >= obs - 1e-15)
    return obs, float((extreme + 1) / (n_null + 1))


def load_aligned_expression(
    slice_id: str, spot_map: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return X_log, coords, labels, barcodes, genes aligned to spot_map rows."""
    entry = get_dataset(slice_id)
    data = entry.load(cache_dir=ROOT / "datasets_cache")
    if "domain_truth" in data.obs.columns:
        keep = data.obs["domain_truth"].notna().to_numpy()
        lab = data.obs["domain_truth"].astype(str).to_numpy()
        keep = keep & ~np.isin(lab, ["NA", "nan", "None", ""])
        data = data.subset_obs(keep)

    map_coords = spot_map[["x", "y"]].to_numpy(dtype=float)
    if data.n_obs == len(spot_map):
        X = log1p_norm(_to_dense(data.X))
        genes = list(map(str, data.var_names))
        labels = data.obs["domain_truth"].astype(str).to_numpy()
        barcodes = list(map(str, data.obs_names))
        return X, map_coords, labels, barcodes, genes

    # Coordinate alignment fallback
    dxy = np.asarray(data.spatial, dtype=float)
    key = {(round(float(x), 1), round(float(y), 1)): i for i, (x, y) in enumerate(dxy)}
    idx = []
    for x, y in map_coords:
        j = key.get((round(float(x), 1), round(float(y), 1)))
        if j is None:
            raise RuntimeError(f"{slice_id}: coordinate alignment failed")
        idx.append(j)
    idx_arr = np.asarray(idx, dtype=int)
    X_full = log1p_norm(_to_dense(data.X))
    X = X_full[idx_arr]
    labels = data.obs["domain_truth"].astype(str).to_numpy()[idx_arr]
    barcodes = [str(data.obs_names[i]) for i in idx_arr]
    genes = list(map(str, data.var_names))
    return X, map_coords, labels, barcodes, genes


def gene_index(genes: list[str], name: str) -> int | None:
    try:
        return genes.index(name)
    except ValueError:
        return None


def zscore_1d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    mu, sd = float(np.mean(v)), float(np.std(v))
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


def composite_score(
    X: np.ndarray, genes: list[str], panel: tuple[str, ...]
) -> tuple[np.ndarray, list[str]]:
    cols = []
    used = []
    for g in panel:
        j = gene_index(genes, g)
        if j is None:
            logger.warning("gene %s missing — skipped from composite", g)
            continue
        cols.append(zscore_1d(X[:, j]))
        used.append(g)
    if not cols:
        return np.zeros(X.shape[0]), []
    return np.mean(np.column_stack(cols), axis=1), used


@dataclass
class GeneTest:
    gene: str
    panel: str
    mean_in: float
    mean_rest: float
    mean_same_layer_out: float
    delta_vs_rest: float
    delta_vs_same_layer: float
    p_vs_rest: float
    p_vs_same_layer: float
    shift_p_vs_rest: float
    shift_p_vs_same_layer: float


def analyse_component(cfg: dict[str, Any]) -> dict[str, Any]:
    slice_id = cfg["slice_id"]
    tag = cfg["tag"]
    label = cfg["label"]
    expected_layer = cfg["expected_layer"]
    expected_class = cfg["expected_class"]

    spots_path = BASE / "results" / slice_id / "spot_uncertainty_map.csv"
    comp_path = BASE / "results" / slice_id / tag / "component_spots.csv"
    if not spots_path.is_file() or not comp_path.is_file():
        raise FileNotFoundError(f"missing inputs for {label}: {spots_path} / {comp_path}")

    spots = pd.read_csv(spots_path)
    comp = pd.read_csv(comp_path)
    # Component indices refer to spot_map row order (spot_index column).
    if "spot_index" in comp.columns:
        comp_idx = comp["spot_index"].to_numpy(dtype=int)
    else:
        # Fallback: match coordinates
        raise RuntimeError("component_spots.csv must contain spot_index")

    in_mask = np.zeros(len(spots), dtype=bool)
    in_mask[comp_idx] = True

    X, coords, labels, barcodes, genes = load_aligned_expression(slice_id, spots)
    assert len(labels) == len(spots)

    same_layer_out = (labels == expected_layer) & (~in_mask)
    rest = ~in_mask

    gene_rows: list[dict[str, Any]] = []
    for panel_name, panel in (
        ("L3_program", L3_PANEL),
        ("myelin", MYELIN_PANEL),
        ("exploratory", EXPLORATORY),
    ):
        for g in panel:
            j = gene_index(genes, g)
            if j is None:
                continue
            vals = X[:, j]
            mean_in = float(vals[in_mask].mean())
            mean_rest = float(vals[rest].mean())
            mean_sl = float(vals[same_layer_out].mean()) if same_layer_out.any() else float("nan")
            p_rest = rank_sum_p(vals[in_mask], vals[rest])
            p_sl = (
                rank_sum_p(vals[in_mask], vals[same_layer_out])
                if same_layer_out.sum() >= 10
                else float("nan")
            )
            d_rest, sp_rest = shift_null_delta(
                coords, vals, in_mask, seed=SEED + abs(hash(g)) % 1000
            )
            # For same-layer: restrict score comparison mask conceptually by
            # testing enrichment of gene in component vs same-layer spots only
            # via a masked shift on the union of in + same_layer_out.
            if same_layer_out.sum() >= 10:
                union = in_mask | same_layer_out
                sub_coords = coords[union]
                sub_scores = vals[union]
                sub_in = in_mask[union]
                d_sl, sp_sl = shift_null_delta(
                    sub_coords, sub_scores, sub_in, seed=SEED + 17 + abs(hash(g)) % 1000
                )
            else:
                d_sl, sp_sl = float("nan"), float("nan")
            gene_rows.append(
                {
                    "component": label,
                    "slice_id": slice_id,
                    "gene": g,
                    "panel": panel_name,
                    "mean_in": mean_in,
                    "mean_rest": mean_rest,
                    "mean_same_layer_out": mean_sl,
                    "delta_vs_rest": mean_in - mean_rest,
                    "delta_vs_same_layer": mean_in - mean_sl
                    if np.isfinite(mean_sl)
                    else float("nan"),
                    "p_vs_rest": p_rest,
                    "p_vs_same_layer": p_sl,
                    "shift_delta_vs_rest": d_rest,
                    "shift_p_vs_rest": sp_rest,
                    "shift_delta_vs_same_layer": d_sl,
                    "shift_p_vs_same_layer": sp_sl,
                    "n_in": int(in_mask.sum()),
                    "n_same_layer_out": int(same_layer_out.sum()),
                }
            )

    l3_score, l3_used = composite_score(X, genes, L3_PANEL)
    my_score, my_used = composite_score(X, genes, MYELIN_PANEL)

    def pack_composite(name: str, score: np.ndarray, used: list[str]) -> dict[str, Any]:
        d_rest, p_rest = shift_null_delta(coords, score, in_mask, seed=SEED + len(name))
        if same_layer_out.sum() >= 10:
            union = in_mask | same_layer_out
            d_sl, p_sl = shift_null_delta(
                coords[union], score[union], in_mask[union], seed=SEED + 99 + len(name)
            )
        else:
            d_sl, p_sl = float("nan"), float("nan")
        return {
            "name": name,
            "genes_used": used,
            "mean_in": float(score[in_mask].mean()),
            "mean_rest": float(score[rest].mean()),
            "mean_same_layer_out": float(score[same_layer_out].mean())
            if same_layer_out.any()
            else float("nan"),
            "delta_vs_rest": float(score[in_mask].mean() - score[rest].mean()),
            "delta_vs_same_layer": float(score[in_mask].mean() - score[same_layer_out].mean())
            if same_layer_out.any()
            else float("nan"),
            "p_vs_rest": rank_sum_p(score[in_mask], score[rest]),
            "p_vs_same_layer": rank_sum_p(score[in_mask], score[same_layer_out])
            if same_layer_out.sum() >= 10
            else float("nan"),
            "shift_p_vs_rest": p_rest,
            "shift_p_vs_same_layer": p_sl,
            "shift_delta_vs_rest": d_rest,
            "shift_delta_vs_same_layer": d_sl,
        }

    composites = {
        "L3_program": pack_composite("L3_program", l3_score, l3_used),
        "myelin": pack_composite("myelin", my_score, my_used),
    }

    # Gate evaluation
    gates: dict[str, Any] = {"expected_class": expected_class}
    if expected_class == "L3_program":
        gates["l3_delta_same_layer_positive"] = composites["L3_program"]["delta_vs_same_layer"] > 0
        gates["l3_shift_p_same_layer_le_0.05"] = (
            composites["L3_program"]["shift_p_vs_same_layer"] <= 0.05
            if np.isfinite(composites["L3_program"]["shift_p_vs_same_layer"])
            else False
        )
        gates["l3_delta_rest_positive"] = composites["L3_program"]["delta_vs_rest"] > 0
        gates["myelin_delta_rest_negative"] = composites["myelin"]["delta_vs_rest"] < 0
        gates["pass"] = bool(
            gates["l3_delta_same_layer_positive"]
            and gates["l3_shift_p_same_layer_le_0.05"]
            and gates["myelin_delta_rest_negative"]
        )
        # Softer diagnostic: direction-only (for replication narrative)
        gates["direction_ok"] = bool(
            gates["l3_delta_rest_positive"] and gates["myelin_delta_rest_negative"]
        )
    else:  # L6_myelin
        gates["myelin_delta_rest_positive"] = composites["myelin"]["delta_vs_rest"] > 0
        gates["myelin_shift_p_rest_le_0.05"] = composites["myelin"]["shift_p_vs_rest"] <= 0.05
        gates["myelin_same_layer_may_be_weak"] = True  # documented expectation
        gates["pass"] = bool(
            gates["myelin_delta_rest_positive"] and gates["myelin_shift_p_rest_le_0.05"]
        )
        gates["direction_ok"] = gates["myelin_delta_rest_positive"]

    # ROI export for IF
    roi = spots.loc[in_mask].copy()
    roi["barcode"] = np.asarray(barcodes)[in_mask]
    roi["domain_truth"] = labels[in_mask]
    roi["component_label"] = label
    roi["expected_layer"] = expected_layer
    roi["L3_program_score"] = l3_score[in_mask]
    roi["myelin_score"] = my_score[in_mask]
    for g in list(L3_PANEL) + list(MYELIN_PANEL):
        j = gene_index(genes, g)
        if j is not None:
            roi[f"expr_{g}"] = X[in_mask, j]

    # Full-slice score map for figure
    score_map = spots.copy()
    score_map["in_component"] = in_mask.astype(int)
    score_map["L3_program_score"] = l3_score
    score_map["myelin_score"] = my_score
    score_map["domain_truth"] = labels
    score_map["barcode"] = barcodes

    return {
        "label": label,
        "slice_id": slice_id,
        "tag": tag,
        "expected_layer": expected_layer,
        "expected_class": expected_class,
        "n_component": int(in_mask.sum()),
        "n_same_layer_out": int(same_layer_out.sum()),
        "gene_tests": gene_rows,
        "composites": composites,
        "gates": gates,
        "roi": roi,
        "score_map": score_map,
    }


def make_figure(results: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.6 * n), squeeze=False)
    for row, res in enumerate(results):
        sm = res["score_map"]
        x, y = sm["x"].to_numpy(), sm["y"].to_numpy()
        # Visium often needs inverted y for histology-like view
        for col, (key, title, cmap) in enumerate(
            [
                ("in_component", f"{res['label']}\ncomponent mask", "Reds"),
                ("L3_program_score", "L3 program (z-mean)", "coolwarm"),
                ("myelin_score", "Myelin program (z-mean)", "coolwarm"),
            ]
        ):
            ax = axes[row, col]
            vals = sm[key].to_numpy(dtype=float)
            if key == "in_component":
                sc = ax.scatter(x, -y, c=vals, s=4, cmap=cmap, linewidths=0, vmin=0, vmax=1)
            else:
                lim = float(np.nanpercentile(np.abs(vals), 98)) or 1.0
                sc = ax.scatter(x, -y, c=vals, s=4, cmap=cmap, linewidths=0, vmin=-lim, vmax=lim)
            # outline component
            m = sm["in_component"].to_numpy(dtype=bool)
            ax.scatter(x[m], -y[m], s=12, facecolors="none", edgecolors="k", linewidths=0.4)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(
        "Pre-registered panel scores on cryptic components (DLPFC Visium)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote figure %s", out_path)


def write_report(results: list[dict[str, Any]], gene_df: pd.DataFrame) -> str:
    lines = [
        "# Panel validation & IF ROI export",
        "",
        "**Pre-registered panels**",
        "",
        f"- L3 program: `{', '.join(L3_PANEL)}`",
        f"- Myelin: `{', '.join(MYELIN_PANEL)}`",
        f"- Exploratory (not for primary claim): `{', '.join(EXPLORATORY)}`",
        "",
        "## Gate results",
        "",
        "| Component | Class | n | L3 Δ same-layer | L3 shift p | Myelin Δ rest | Myelin shift p | direction_ok | **pass** |",
        "|-----------|-------|--:|----------------:|-----------:|--------------:|---------------:|:------------:|:--------:|",
    ]
    for res in results:
        c = res["composites"]
        g = res["gates"]
        lines.append(
            f"| {res['label']} | {res['expected_class']} | {res['n_component']} | "
            f"{c['L3_program']['delta_vs_same_layer']:.3f} | "
            f"{c['L3_program']['shift_p_vs_same_layer']:.3f} | "
            f"{c['myelin']['delta_vs_rest']:.3f} | "
            f"{c['myelin']['shift_p_vs_rest']:.3f} | "
            f"{'Y' if g.get('direction_ok') else 'N'} | "
            f"{'**PASS**' if g.get('pass') else 'FAIL'} |"
        )

    l3_res = [r for r in results if r["expected_class"] == "L3_program"]
    l6_res = [r for r in results if r["expected_class"] == "L6_myelin"]
    cross = all(r["gates"].get("direction_ok") for r in l3_res) and len(l3_res) >= 2
    hard = all(r["gates"].get("pass") for r in l3_res) and len(l3_res) >= 2
    n_l3_dir = sum(1 for r in l3_res if r["gates"].get("direction_ok"))
    n_l6_dir = sum(1 for r in l6_res if r["gates"].get("direction_ok"))
    n_l6_pass = sum(1 for r in l6_res if r["gates"].get("pass"))
    lines += [
        "",
        f"**L3 direction OK:** `{n_l3_dir}/{len(l3_res)}` donors/components "
        f"(need L3-program ↑ vs rest and myelin ↓ vs rest).",
        "",
        f"**Cross-donor L3 direction (all listed L3 niches):** `{'YES' if cross else 'NO'}`.",
        "",
        f"**Cross-donor L3 hard gate (same-layer shift p≤0.05 on all):** "
        f"`{'YES' if hard else 'NO'}`.",
        "",
        f"**L6 myelin direction OK:** `{n_l6_dir}/{len(l6_res)}`; "
        f"**L6 hard pass:** `{n_l6_pass}/{len(l6_res)}`.",
        "",
        "## Composite scores (detail)",
        "",
    ]
    for res in results:
        lines.append(f"### {res['label']}")
        lines.append("")
        for name, comp in res["composites"].items():
            lines.append(
                f"- **{name}** genes={comp['genes_used']}: "
                f"mean_in={comp['mean_in']:.3f}, rest={comp['mean_rest']:.3f}, "
                f"same-layer out={comp['mean_same_layer_out']:.3f}; "
                f"Δrest={comp['delta_vs_rest']:.3f} (shift p={comp['shift_p_vs_rest']:.3f}); "
                f"Δsame-layer={comp['delta_vs_same_layer']:.3f} "
                f"(shift p={comp['shift_p_vs_same_layer']:.3f})"
            )
        lines.append("")

    lines += [
        "## Per-gene primary panel (same-layer contrast)",
        "",
        "| Component | Gene | Panel | Δ same-layer | p same-layer | shift p | Δ rest |",
        "|-----------|------|-------|-------------:|-------------:|--------:|-------:|",
    ]
    primary = gene_df[gene_df["panel"].isin(["L3_program", "myelin"])].copy()
    for _, r in primary.sort_values(["component", "panel", "gene"]).iterrows():
        lines.append(
            f"| {r['component']} | `{r['gene']}` | {r['panel']} | "
            f"{r['delta_vs_same_layer']:.3f} | {r['p_vs_same_layer']:.2e} | "
            f"{r['shift_p_vs_same_layer']:.3f} | {r['delta_vs_rest']:.3f} |"
        )

    lines += [
        "",
        "## IF / imaging hand-off",
        "",
        "ROI CSVs (one row per Visium spot in the component):",
        "",
        "- `results/panel_validation/ROI_151508_L6_n154.csv`",
        "- `results/panel_validation/ROI_151508_L3_n138.csv`",
        "- `results/panel_validation/ROI_151669_L3_n137.csv`",
        "- `results/panel_validation/ROI_151673_L3_n47.csv`",
        "- `results/panel_validation/ROI_151673_L6_n24.csv`",
        "",
        "**Recommended IF panel (minimal):** ENC1, HOPX, MBP (optional + PLP1, GAP43).",
        "",
        "**Sectioning notes:**",
        "",
        "1. Align Visium pixel coordinates (`x`,`y` in ROI CSV) to H&E via the "
        "original Space Ranger `spatial/` folder for that section.",
        "2. Score IF mean intensity inside a 55 µm radius of each barcode centroid "
        "(Visium center-to-center ~100 µm; use spot diameter from scalefactors).",
        "3. Primary stats: L3 ROIs should be ENC1/HOPX-high and MBP-low vs "
        "same-layer non-ROI L3; L6 ROI myelin-high vs whole section (same-layer optional).",
        "",
        "## Claim bounds",
        "",
        "1. Visium expression is a **proxy**, not IF. Hard biological claim still needs protein.",
        "2. Same-layer shift-null is stricter than whole-slice DE; failures here block "
        "naming a new cell state.",
        "3. SCGB/SAA-family genes remain exploratory / artifact-suspect and are excluded "
        "from primary panels.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    all_genes: list[dict[str, Any]] = []

    for cfg in COMPONENTS:
        logger.info("=== %s ===", cfg["label"])
        res = analyse_component(cfg)
        results.append(res)
        all_genes.extend(res["gene_tests"])
        # ROI
        roi_path = OUT / f"ROI_{res['label']}.csv"
        res["roi"].to_csv(roi_path, index=False)
        logger.info("ROI n=%s → %s", len(res["roi"]), roi_path)
        # score map
        res["score_map"].to_csv(OUT / f"score_map_{res['label']}.csv", index=False)
        # gates json
        (OUT / f"gates_{res['label']}.json").write_text(
            json.dumps(
                {
                    "label": res["label"],
                    "gates": res["gates"],
                    "composites": res["composites"],
                    "n_component": res["n_component"],
                    "n_same_layer_out": res["n_same_layer_out"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info(
            "gates pass=%s direction_ok=%s",
            res["gates"].get("pass"),
            res["gates"].get("direction_ok"),
        )

    gene_df = pd.DataFrame(all_genes)
    gene_df.to_csv(OUT / "panel_gene_tests.csv", index=False)

    summary_rows = []
    for res in results:
        row = {
            "label": res["label"],
            "slice_id": res["slice_id"],
            "expected_class": res["expected_class"],
            "n_component": res["n_component"],
            "pass": res["gates"].get("pass"),
            "direction_ok": res["gates"].get("direction_ok"),
        }
        for name, comp in res["composites"].items():
            row[f"{name}_delta_rest"] = comp["delta_vs_rest"]
            row[f"{name}_delta_same_layer"] = comp["delta_vs_same_layer"]
            row[f"{name}_shift_p_rest"] = comp["shift_p_vs_rest"]
            row[f"{name}_shift_p_same_layer"] = comp["shift_p_vs_same_layer"]
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "panel_summary.csv", index=False)

    make_figure(results, OUT / "figure_panel_scores")
    report = write_report(results, gene_df)
    (OUT / "PANEL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    (BASE / "PANEL_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s", OUT / "PANEL_VALIDATION_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
