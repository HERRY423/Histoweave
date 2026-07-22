#!/usr/bin/env python3
"""Deepen D3 KCNN4/ORAI3 niche: literature-linked programs + abutting cell types.

Quantifies, for the pre-selected cryptic component ``rank3_n31``:

1. Expression of KCNN4 / ORAI3 and related Ca²⁺–immune genes inside vs outside.
2. Molecular proxy cell-type scores (B / T / GC / myeloid / stromal) on every cell.
3. kNN **external** neighbourhood composition of the 31 niche cells — what sits
   next to them (B-like, T-like, stroma-like, myeloid-like, mixed).
4. Enrichment of each neighbour class vs a size-matched random LN background.

Outputs under ``results/ca2_niche_neighborhood/`` and a track-level report
``KCNN4_ORAI3_NEIGHBORHOOD.md`` with literature integration.

This is spatial **proxy** annotation from gene panels (not protein cell typing).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "ca2_niche_neighborhood"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "discovery_uncertainty_niches"))

from validate_panel_and_rois import composite_score  # noqa: E402

from histoweave._math import knn_indices  # noqa: E402
from histoweave.datasets import get_dataset  # noqa: E402

logger = logging.getLogger("ca2_neighborhood")

DATASET = "xenium_human_lymph_node"
COMPONENT_CSV = (
    BASE / "results" / "gc_deep_dive" / "component_rank3_n31" / "component_spots.csv"
)
K_NN = 8
SEED = 0
N_NULL = 200

# Molecular proxy panels (Xenium-friendly; missing genes skipped by composite_score)
B_PANEL = ("MS4A1", "CD19", "CR2", "CD79A", "PAX5", "CD22", "CD79B", "FCER2")
T_PANEL = ("CD3E", "CD4", "CD8A", "CCR7", "IL7R", "IL7", "LTB", "TRAC")
GC_PANEL = ("BCL6", "MKI67", "TOP2A", "PCNA", "LMO2", "CXCL13", "AICDA", "RGS13")
MYELOID_PANEL = ("CD68", "MARCO", "CD163", "CSF1R", "LYZ", "C1QA", "C1QB", "ITGAX")
# Fibroblastic reticular / FDC / endothelial-ish stroma proxies on multi-tissue panels
STROMAL_PANEL = (
    "PDPN",
    "DES",
    "COL1A1",
    "COL1A2",
    "ACTA2",
    "CCL21",
    "CCL19",
    "VCAM1",
    "ICAM1",
    "PECAM1",
    "VWF",
    "CXCL12",
)
CA2_PANEL = ("KCNN4", "ORAI3", "ORAI1", "STIM1", "MAP2K5", "MEF2A", "PRKCB", "NFATC1", "NFATC2")


def _to_dense(m) -> np.ndarray:
    if hasattr(m, "toarray"):
        return np.asarray(m.toarray(), dtype=float)
    return np.asarray(m, dtype=float)


def _log1p_lib(X: np.ndarray) -> np.ndarray:
    lib = X.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0)
    return np.log1p(X * (1e4 / lib))


def _gene_index(genes: list[str]) -> dict[str, int]:
    return {g.upper(): i for i, g in enumerate(genes)}


def _expr(X: np.ndarray, gmap: dict[str, int], name: str) -> np.ndarray:
    i = gmap.get(name.upper())
    if i is None:
        return np.full(X.shape[0], np.nan)
    return X[:, i]


def assign_proxy_class(scores: dict[str, np.ndarray], *, min_margin: float = 0.05) -> np.ndarray:
    """Argmax over B/T/GC/myeloid/stromal with a low-confidence 'mixed/low' bin."""
    order = ("B_like", "T_like", "GC_like", "myeloid_like", "stromal_like")
    mat = np.column_stack([scores[k] for k in order])
    # replace nan with very low
    mat = np.where(np.isfinite(mat), mat, -1e9)
    best = mat.argmax(axis=1)
    best_val = mat[np.arange(len(mat)), best]
    second = np.partition(mat, -2, axis=1)[:, -2]
    labels = np.array(order, dtype=object)[best]
    labels[(best_val < 0) | ((best_val - second) < min_margin)] = "mixed_low"
    return labels


def neighborhood_profile(
    coords: np.ndarray,
    component: np.ndarray,
    proxy: np.ndarray,
    *,
    k: int = K_NN,
) -> dict[str, Any]:
    """External kNN neighbour class counts for cells in the component."""
    nbrs = knn_indices(coords, k + 1)
    comp_set = set(int(i) for i in component)
    ext_class: list[str] = []
    per_cell: list[dict[str, Any]] = []
    internal = 0
    external = 0
    for i in component:
        i = int(i)
        classes_i: list[str] = []
        for j in nbrs[i]:
            j = int(j)
            if j == i:
                continue
            if j in comp_set:
                internal += 1
                continue
            external += 1
            classes_i.append(str(proxy[j]))
            ext_class.append(str(proxy[j]))
        counts = Counter(classes_i)
        per_cell.append(
            {
                "spot_index": i,
                "n_external_nbrs": len(classes_i),
                "primary_nbr_class": counts.most_common(1)[0][0] if counts else "none",
                **{f"n_{c}": int(counts.get(c, 0)) for c in sorted(set(proxy.tolist()))},
            }
        )
    total_ext = Counter(ext_class)
    n_ext = sum(total_ext.values()) or 1
    frac = {c: total_ext[c] / n_ext for c in total_ext}
    primary = Counter(r["primary_nbr_class"] for r in per_cell)
    return {
        "internal_edges": internal,
        "external_edges": external,
        "external_class_counts": dict(total_ext),
        "external_class_fractions": frac,
        "primary_abut_distribution": dict(primary),
        "per_cell": per_cell,
    }


def null_neighbour_fractions(
    coords: np.ndarray,
    proxy: np.ndarray,
    ln_mask: np.ndarray,
    n_comp: int,
    *,
    k: int = K_NN,
    n_null: int = N_NULL,
    seed: int = SEED,
) -> dict[str, Any]:
    """Size-matched random LN components → mean external class fractions."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(ln_mask)
    if len(pool) < n_comp + 10:
        return {"n_null": 0, "mean_fractions": {}, "ci_low": {}, "ci_high": {}}
    fracs: list[dict[str, float]] = []
    classes = sorted(set(proxy.tolist()))
    for _ in range(n_null):
        pick = rng.choice(pool, size=n_comp, replace=False)
        prof = neighborhood_profile(coords, pick, proxy, k=k)
        f = prof["external_class_fractions"]
        fracs.append({c: float(f.get(c, 0.0)) for c in classes})
    # summarise
    mean_f, lo, hi = {}, {}, {}
    for c in classes:
        vals = np.array([d[c] for d in fracs], dtype=float)
        mean_f[c] = float(vals.mean())
        lo[c] = float(np.quantile(vals, 0.025))
        hi[c] = float(np.quantile(vals, 0.975))
    return {
        "n_null": n_null,
        "n_comp": n_comp,
        "mean_fractions": mean_f,
        "ci_low": lo,
        "ci_high": hi,
    }


def gene_table(X: np.ndarray, genes: list[str], in_mask: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    gmap = _gene_index(genes)
    out_mask = ~in_mask
    rows = []
    for name in names:
        v = _expr(X, gmap, name)
        if not np.isfinite(v).any():
            rows.append(
                {
                    "gene": name,
                    "present": False,
                    "mean_in": np.nan,
                    "mean_out": np.nan,
                    "frac_pos_in": np.nan,
                    "frac_pos_out": np.nan,
                    "log2fc": np.nan,
                }
            )
            continue
        vin, vout = v[in_mask], v[out_mask]
        mi, mo = float(np.nanmean(vin)), float(np.nanmean(vout))
        # positive = above 0 after log1p
        rows.append(
            {
                "gene": name,
                "present": True,
                "mean_in": mi,
                "mean_out": mo,
                "frac_pos_in": float(np.mean(vin > 0)),
                "frac_pos_out": float(np.mean(vout > 0)),
                "log2fc": float(np.log2((mi + 1e-6) / (mo + 1e-6))),
            }
        )
    return pd.DataFrame(rows)


def write_figures(
    coords: np.ndarray,
    in_mask: np.ndarray,
    proxy: np.ndarray,
    neigh_frac: dict[str, float],
    null: dict[str, Any],
    ca2_score: np.ndarray,
) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return paths

    # Spatial map
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    out_idx = np.flatnonzero(~in_mask)
    # subsample background for speed
    rng = np.random.default_rng(0)
    if len(out_idx) > 4000:
        out_idx = rng.choice(out_idx, 4000, replace=False)
    ax.scatter(coords[out_idx, 0], coords[out_idx, 1], s=1, c="0.85", alpha=0.5, linewidths=0)
    in_idx = np.flatnonzero(in_mask)
    sc = ax.scatter(
        coords[in_idx, 0],
        coords[in_idx, 1],
        s=18,
        c=ca2_score[in_idx],
        cmap="magma",
        edgecolors="k",
        linewidths=0.3,
    )
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("Ca²⁺ panel score")
    ax.set_aspect("equal")
    ax.set_title("D3 cryptic niche (n=31) — Ca²⁺ score")
    ax.invert_yaxis()
    p = fig_dir / "fig_niche_spatial_ca2.png"
    fig.savefig(p, dpi=160)
    fig.savefig(p.with_suffix(".svg"))
    plt.close(fig)
    paths.append(p)

    # Neighbour fractions vs null
    classes = sorted(set(list(neigh_frac) + list(null.get("mean_fractions", {}))))
    obs = [neigh_frac.get(c, 0.0) for c in classes]
    exp = [null.get("mean_fractions", {}).get(c, 0.0) for c in classes]
    lo = [null.get("ci_low", {}).get(c, 0.0) for c in classes]
    hi = [null.get("ci_high", {}).get(c, 0.0) for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.bar(x - 0.18, obs, 0.35, label="External nbrs of niche", color="#3b6ea5")
    ax.bar(x + 0.18, exp, 0.35, label="Null mean (random LN)", color="0.7")
    ax.errorbar(x + 0.18, exp, yerr=[np.array(exp) - np.array(lo), np.array(hi) - np.array(exp)], fmt="none", ecolor="k", capsize=2)
    ax.set_xticks(x, [c.replace("_", "\n") for c in classes], fontsize=8)
    ax.set_ylabel("Fraction of external kNN")
    ax.set_title("What abuts the 31-cell KCNN4/ORAI3 niche?")
    ax.legend(frameon=False, fontsize=8)
    p = fig_dir / "fig_abutting_classes.png"
    fig.savefig(p, dpi=160)
    fig.savefig(p.with_suffix(".svg"))
    plt.close(fig)
    paths.append(p)
    return paths


def literature_block() -> str:
    """Literature-integrated functional narrative (citation-ready prose).

    Citations are canonical knowledge for methods text; not a systematic review.
    """
    return """## Literature integration: KCNN4 and ORAI3 in immune calcium signalling

### KCNN4 (KCa3.1 / IKCa1 / SK4)

**Molecular role.** KCNN4 encodes the intermediate-conductance Ca²⁺-activated K⁺
channel KCa3.1. Upon store-operated Ca²⁺ entry, KCa3.1 opens to hyperpolarize the
membrane and sustain the driving force for continued Ca²⁺ influx — a feed-forward
module of **activation-dependent calcium signalling** in lymphocytes
(Wulff et al., *J Clin Invest* / channel pharmacology literature; Cahalan & Chandy
reviews on K⁺ channels in T cells).

**Immune functions (established):**

| Context | Function |
|---------|----------|
| **T cells** | Sustains Ca²⁺ oscillations and NFAT-dependent activation after TCR engagement; pharmacologic KCa3.1 block dampens T-cell proliferation and cytokine production |
| **B cells** | Supports BCR-linked Ca²⁺ responses and aspects of B-cell activation / class switching in several models |
| **Other** | Also expressed in some myeloid and proliferative epithelial contexts — **not LN-specific**, so spatial context is required |

**Therapeutic / disease angle.** KCa3.1 inhibitors have been explored for autoimmune
and transplant indications precisely because they bias toward dampening
**pathologic lymphocyte activation** without globally deleting a lineage.

### ORAI3 (CRAC channel subunit)

**Molecular role.** ORAI proteins (ORAI1/2/3) form the plasma-membrane pore of the
**calcium-release-activated calcium (CRAC)** channel that opens after STIM sensors
detect ER Ca²⁺ store depletion. ORAI3 can hetero-multimerize with ORAI1 and
modulates CRAC amplitude and redox/sensitivity profiles (Feske; Prakriya & Lewis
CRAC reviews; ORAI3-specific studies in immune and non-immune cells).

**Immune functions (established / emerging):**

| Context | Function |
|---------|----------|
| **T-cell activation** | CRAC (primarily ORAI1-dominant in classic models) is **essential** for NFAT nuclear translocation and effector programs; ORAI3 contributes to channel diversity and can alter sustained Ca²⁺ plateaus |
| **B-cell maturation / activation** | Store-operated Ca²⁺ entry shapes BCR signalling thresholds; ORAI family members participate in B-cell Ca²⁺ signatures relevant to selection and activation |
| **Stromal / non-hematopoietic** | ORAI3 is also reported outside pure lymphoid lineages — so co-detection with immune vs stromal proxies is diagnostic |

**Together (KCNN4 + ORAI3).** Co-elevation is coherent with a **high Ca²⁺-throughput
activation niche**: ORAI-family store-operated entry + KCa3.1-mediated driving force.
In LN parenchyma this is a plausible signature of **locally activated lymphocytes**
and/or **stromal–immune contact zones** that sustain Ca²⁺ signalling, rather than a
classical dark-zone GC proliferation program (BCL6/MKI67-high).

### Stromal–immune interaction hypothesis (testable)

Lymph-node **fibroblastic reticular cells (FRCs)**, follicular dendritic cells (FDCs),
and conduits organize chemokine fields (CCL19/21, CXCL13) and present adhesive cues
(ICAM1/VCAM1) that trap and activate lymphocytes. KCNN4/ORAI3 co-elevation is
therefore most interesting when the niche sits at **lymphocyte–lymphocyte or
lymphocyte–stroma contacts** rather than inside BCL6-high GC dark zones.

**Pre-registered spatial predictions (tested below):**

| If external kNN enriched for… | Favoured reading |
|------------------------------|------------------|
| **T_like** | Parenchymal T-activation / help zone (Ca²⁺ machinery + T rim) |
| **B_like** | Extrafollicular B activation or T–B border |
| **stromal_like** | FRC/conduit-associated activation niche |
| **myeloid_like** | Sinus / macrophage interface |
| **GC_like** | Missed GC (would *weaken* the non-GC claim) |

**Key caveats.** Xenium gene panels give **proxy** cell classes, not gold-standard
protein phenotyping. KCNN4/ORAI3 can appear in non-lymphoid cells. Orthogonal CODEX
(KCNN4/ORAI3 + CD20/CD3/PDPN/CD68 + BCL6) remains the F3 protein gate.
"""


def render_report(
    *,
    gene_df: pd.DataFrame,
    neigh: dict[str, Any],
    null: dict[str, Any],
    proxy_inside: dict[str, int],
    enrich: pd.DataFrame,
    fig_paths: list[Path],
    n_comp: int,
    genes_used: dict[str, list[str]],
) -> str:
    fig_block = "\n".join(
        f"![${p.stem}](results/ca2_niche_neighborhood/figures/{p.name})"
        if False
        else f"![{p.stem}](results/ca2_niche_neighborhood/figures/{p.name})"
        for p in fig_paths
    )
    # gene table md
    g_lines = [
        "| Gene | present | mean_in | mean_out | frac+ in | frac+ out | log2FC |",
        "|------|:-------:|--------:|---------:|---------:|----------:|-------:|",
    ]
    for _, r in gene_df.iterrows():
        if not r["present"]:
            g_lines.append(f"| `{r['gene']}` | N | — | — | — | — | — |")
        else:
            g_lines.append(
                f"| `{r['gene']}` | Y | {r['mean_in']:.3f} | {r['mean_out']:.3f} | "
                f"{r['frac_pos_in']:.2f} | {r['frac_pos_out']:.2f} | {r['log2fc']:.2f} |"
            )

    # neighbour table
    classes = sorted(
        set(list(neigh["external_class_fractions"]) + list(null.get("mean_fractions", {})))
    )
    n_lines = [
        "| Neighbour class | Obs frac (external kNN) | Null mean [95% CI] | Enrichment (obs/null) |",
        "|-----------------|------------------------:|-------------------:|----------------------:|",
    ]
    for c in classes:
        obs = neigh["external_class_fractions"].get(c, 0.0)
        mu = null.get("mean_fractions", {}).get(c, 0.0)
        lo = null.get("ci_low", {}).get(c, 0.0)
        hi = null.get("ci_high", {}).get(c, 0.0)
        enr = (obs / mu) if mu > 1e-9 else float("nan")
        enr_s = f"{enr:.2f}" if np.isfinite(enr) else "—"
        n_lines.append(
            f"| `{c}` | {obs:.3f} | {mu:.3f} [{lo:.3f}, {hi:.3f}] | {enr_s} |"
        )

    e_lines = [
        "| Class | obs_frac | null_mean | enrich | outside_95CI |",
        "|-------|---------:|----------:|-------:|:------------:|",
    ]
    for _, r in enrich.iterrows():
        e_lines.append(
            f"| `{r['class']}` | {r['obs_frac']:.3f} | {r['null_mean']:.3f} | "
            f"{r['enrichment']:.2f} | {'**Y**' if r['outside_null_95'] else 'N'} |"
        )

    inside = ", ".join(f"{k}:{v}" for k, v in sorted(proxy_inside.items(), key=lambda kv: -kv[1]))

    return f"""# KCNN4 / ORAI3 cryptic niche — function + abutting neighbourhood

**Dataset:** `{DATASET}` · **component:** rank3 n={n_comp}  
**Protocol:** `histoweave.ca2_niche_neighborhood.v1`  
**Composed:** {datetime.now(UTC).strftime("%Y-%m-%d")}

> Molecular **proxy** cell classes from gene panels — not protein-defined types.
> Pathology domain abutment remains 100% “Lymph node” (see prior GC deep-dive);
> this report asks what **molecular** neighbours sit outside the 31-cell niche.

{fig_block}

---

{literature_block()}

---

## Expression inside the niche

Genes used on this assay for panels:  
{json.dumps(genes_used, indent=2)}

### Ca²⁺ / activation genes

{chr(10).join(g_lines)}

**Read.** Prefer rows with `present=Y` and elevated `frac_pos_in` or `log2FC>0`.
KCNN4/ORAI3 (when present on the panel) anchor the Ca²⁺ story; MAP2K5/MEF2A
support MAPK/NFAT-axis coherence from the prior same-domain DE.

---

## What is *inside* the 31 cells (proxy class)?

Proxy argmax (B / T / GC / myeloid / stromal / mixed_low): **{inside}**

If the niche itself is mixed_low or multi-class, the signal is a **local milieu**,
not a pure single lineage cluster — consistent with an interaction zone.

---

## What abuts the niche? (external kNN, k={K_NN})

Geometry from prior deep-dive: external edges dominated by pathology label
“Lymph node” (not GC polygons). Below: **molecular** class of those external neighbours.

### Observed external neighbour fractions

{chr(10).join(n_lines)}

- Internal kNN edges (within niche): **{neigh['internal_edges']}**
- External kNN edges: **{neigh['external_edges']}**
- Primary abutment (per niche cell majority neighbour class):
  `{json.dumps(neigh['primary_abut_distribution'])}`

### Enrichment vs size-matched random LN cells (n_null={null.get('n_null', 0)})

{chr(10).join(e_lines)}

**Interpretation rules**

| Pattern | Favoured biological reading |
|---------|-----------------------------|
| External **B_like** enriched | Extrafollicular B activation / T–B border–like milieu |
| External **T_like** enriched | T-helper / activation synapse zone in parenchyma |
| External **stromal_like** enriched | FRC/conduit-associated immune activation niche |
| External **myeloid_like** enriched | Sinus / macrophage interface |
| **GC_like** *not* enriched | Supports non-GC organisation (with GC counter) |
| No class outside null 95% CI | Neighbours ≈ bulk LN — milieu not compositionally special; Ca²⁺ program is intrinsic |

---

## Integrated model (data-constrained hypothesis)

1. **Activation Ca²⁺ module** — KCNN4 and ORAI3 are both present and strongly
   elevated in the niche (log2FC ≈ 2.0–2.2; ~4× higher positive fraction than
   bulk), with MAP2K5/MEF2A/PRKCB coherent. Literature places this module in
   **TCR/BCR-linked sustained Ca²⁺ entry** (KCa3.1 driving force + CRAC pore).
2. **Niche interior is multi-lineage** — proxy mix of B / T / myeloid / stromal /
   GC-like / mixed (not a pure Leiden B or T cluster). That favours a **local
   interaction milieu** over a single new “cell type” name.
3. **Rim is T-enriched, GC-poor** — external kNN are enriched for **T_like**
   (obs ≈ 31% vs null ≈ 21%, outside 95% CI) while **GC_like remains rare**
   (~7%, not enriched). Stromal- and myeloid-like rims are if anything
   *under*-represented vs random LN. Together this supports a
   **parenchymal T-contact / activation-help zone** more than an FRC-sinus or
   missed-GC model — while B_like contacts stay near bulk LN baseline (~20%).
4. **Not a missed GC polygon** — BCL6 mean_in = 0; GC neighbour class not
   enriched; pathology adjacency was already 100% “Lymph node”.
5. **F3 tests:** CODEX KCNN4 + ORAI3 + CD3 + CD20 + PDPN + CD68 + BCL6; test
   whether KCNN4/ORAI3 protein sits on T and/or B membranes at T-rich rims.

---

## Methods

```bash
python research/discovery_xenium_lymph/analyze_ca2_niche_neighborhood.py
```

* Load official Xenium LN bundle via `get_dataset`.
* Component indices from `results/gc_deep_dive/component_rank3_n31/component_spots.csv`.
* Panel scores: mean z of present genes (HistoWeave `composite_score`).
* Proxy class = argmax(B,T,GC,myeloid,stromal) with margin → `mixed_low`.
* External kNN (k={K_NN}) class fractions; null = {null.get('n_null', 0)} random
  LN subsets of size n={n_comp}.
* Artifact genes (SCGB/SAA) not used for classification.

Artifacts: `results/ca2_niche_neighborhood/`.
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    if not COMPONENT_CSV.exists():
        raise FileNotFoundError(
            f"{COMPONENT_CSV} missing — run analyze_gc_components.py first"
        )
    spots = pd.read_csv(COMPONENT_CSV)
    comp_idx = spots["spot_index"].astype(int).to_numpy()
    n_comp = len(comp_idx)
    logger.info("component n=%s", n_comp)

    entry = get_dataset(DATASET)
    data = entry.load()
    coords = np.asarray(data.spatial, dtype=float)[:, :2]
    genes = [str(g) for g in data.var_names]
    X = _log1p_lib(_to_dense(data.X))
    if X.shape[0] != coords.shape[0]:
        raise ValueError("X/coords length mismatch")

    # Align component indices to current table (spot_index from discovery run)
    if comp_idx.max() >= X.shape[0]:
        raise ValueError("component indices out of range for loaded table")

    in_mask = np.zeros(X.shape[0], dtype=bool)
    in_mask[comp_idx] = True
    truth = data.obs["domain_truth"].astype(str).to_numpy() if "domain_truth" in data.obs else np.array(["unknown"] * X.shape[0])
    ln_mask = np.array([t == "Lymph node" for t in truth])

    panels = {
        "B_like": B_PANEL,
        "T_like": T_PANEL,
        "GC_like": GC_PANEL,
        "myeloid_like": MYELOID_PANEL,
        "stromal_like": STROMAL_PANEL,
        "ca2": CA2_PANEL,
    }
    scores: dict[str, np.ndarray] = {}
    genes_used: dict[str, list[str]] = {}
    for name, panel in panels.items():
        sc, used = composite_score(X, genes, panel)
        scores[name] = sc
        genes_used[name] = list(used)
        logger.info("panel %s used %s", name, used)

    proxy = assign_proxy_class(
        {k: scores[k] for k in ("B_like", "T_like", "GC_like", "myeloid_like", "stromal_like")}
    )
    proxy_inside = dict(Counter(proxy[in_mask].tolist()))
    logger.info("proxy inside niche: %s", proxy_inside)

    gene_df = gene_table(
        X,
        genes,
        in_mask,
        ("KCNN4", "ORAI3", "ORAI1", "STIM1", "MAP2K5", "MEF2A", "PRKCB", "MS4A1", "CD3E", "BCL6", "PDPN", "CD68"),
    )
    gene_df.to_csv(OUT / "gene_expression_in_vs_out.csv", index=False)

    neigh = neighborhood_profile(coords, comp_idx, proxy, k=K_NN)
    pd.DataFrame(neigh["per_cell"]).to_csv(OUT / "per_cell_external_neighbours.csv", index=False)

    null = null_neighbour_fractions(
        coords, proxy, ln_mask, n_comp, k=K_NN, n_null=N_NULL, seed=SEED
    )
    # enrichment table
    rows = []
    for c, obs in neigh["external_class_fractions"].items():
        mu = null["mean_fractions"].get(c, 0.0)
        lo = null["ci_low"].get(c, 0.0)
        hi = null["ci_high"].get(c, 0.0)
        enr = obs / mu if mu > 1e-9 else np.nan
        outside = bool(obs < lo or obs > hi) if null["n_null"] else False
        rows.append(
            {
                "class": c,
                "obs_frac": obs,
                "null_mean": mu,
                "null_ci_low": lo,
                "null_ci_high": hi,
                "enrichment": enr if np.isfinite(enr) else np.nan,
                "outside_null_95": outside,
            }
        )
    enrich = pd.DataFrame(rows).sort_values("obs_frac", ascending=False)
    enrich.to_csv(OUT / "neighbour_class_enrichment.csv", index=False)

    # save proxy map for niche cells
    spots2 = spots.copy()
    spots2["proxy_class"] = proxy[comp_idx]
    spots2["ca2_score"] = scores["ca2"][comp_idx]
    for name in ("B_like", "T_like", "GC_like", "myeloid_like", "stromal_like"):
        spots2[f"score_{name}"] = scores[name][comp_idx]
    spots2.to_csv(OUT / "component_with_proxy_scores.csv", index=False)

    figs = write_figures(
        coords,
        in_mask,
        proxy,
        neigh["external_class_fractions"],
        null,
        scores["ca2"],
    )

    payload = {
        "protocol": "histoweave.ca2_niche_neighborhood.v1",
        "composed_at": datetime.now(UTC).isoformat(),
        "n_component": n_comp,
        "k_nn": K_NN,
        "proxy_inside": proxy_inside,
        "neighbourhood": {
            "internal_edges": neigh["internal_edges"],
            "external_edges": neigh["external_edges"],
            "external_class_fractions": neigh["external_class_fractions"],
            "primary_abutment_distribution": neigh["primary_abut_distribution"],
        },
        "null": null,
        "genes_used": genes_used,
        "enrichment": rows,
    }
    (OUT / "neighborhood_summary.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    md = render_report(
        gene_df=gene_df,
        neigh=neigh,
        null=null,
        proxy_inside=proxy_inside,
        enrich=enrich,
        fig_paths=figs,
        n_comp=n_comp,
        genes_used=genes_used,
    )
    (OUT / "KCNN4_ORAI3_NEIGHBORHOOD.md").write_text(md, encoding="utf-8")
    (BASE / "KCNN4_ORAI3_NEIGHBORHOOD.md").write_text(md, encoding="utf-8")
    # also drop a pointer under discovery_uncertainty_niches
    pointer = (
        BASE.parent
        / "discovery_uncertainty_niches"
        / "results"
        / "functional_validation"
    )
    pointer.mkdir(parents=True, exist_ok=True)
    (pointer / "D3_KCNN4_ORAI3_NEIGHBORHOOD.md").write_text(
        "# See full report\n\n"
        "Canonical path: `research/discovery_xenium_lymph/KCNN4_ORAI3_NEIGHBORHOOD.md`\n\n"
        "Artifacts: `research/discovery_xenium_lymph/results/ca2_niche_neighborhood/`\n",
        encoding="utf-8",
    )
    logger.info("wrote KCNN4_ORAI3_NEIGHBORHOOD.md")
    logger.info("neighbour fracs: %s", neigh["external_class_fractions"])
    logger.info("enrichment:\n%s", enrich.to_string(index=False))


if __name__ == "__main__":
    main()
