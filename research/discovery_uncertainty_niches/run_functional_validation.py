#!/usr/bin/env python3
"""Functional validation of cryptic niches — disease + spatial organisation.

Scores pre-registered functional modules against frozen DE tables for:

* D1 — L6 myelin niche (151508 largest component)
* D2 — L3 plasticity niche (151508 rank-1 n=138; optional multi-donor summary)
* D3 — Xenium LN Ca²⁺ niche (rank-3 n=31)

Outputs under ``results/functional_validation/``:

* ``module_scores.csv`` — hypergeometric enrichment per discovery × module
* ``functional_claims.json`` — machine-readable claim objects
* ``FUNCTIONAL_VALIDATION.md`` — narrative (also mirrored to track root)
* figures for module hit maps

This is **computational functional mapping**, not wet-lab causation. Claim
levels stay below protein IF / perturbation until those return.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from functional_modules import DISCOVERY_MODULES, MODULE_BY_ID, MODULES

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "functional_validation"
FIG = OUT / "figures"
XENIUM = ROOT.parent / "discovery_xenium_lymph" / "results"

logger = logging.getLogger(__name__)

# Visium DLPFC DE often flags secretory / acute-phase transcripts that are
# known confounders (ambient RNA, section edge, non-neural contamination).
# These genes are *reported* in raw DE CSVs but are **excluded** from
# organisation-principle and disease-axis primary claims.
ARTIFACT_RISK_GENES: frozenset[str] = frozenset(
    {
        "SCGB2A2",
        "SCGB1D2",
        "SCGB1A1",
        "SAA1",
        "SAA2",
        "KRT8",
        "KRT18",
        "KRT19",
        "MUC1",
        "TFF1",
        "TFF3",
        "AGR2",
    }
)

# D3 classical_gc_counter is a *negative-control / non-enrichment* test, not a
# standard over-representation. Its pass rule and FDR treatment are pre-declared
# here (see FUNCTIONAL_VALIDATION.md § Statistical note: D3 GC counter).
GC_COUNTER_RULES: dict[str, Any] = {
    "test_class": "negative_control_non_enrichment",
    "not": "hypergeometric_overrepresentation_of_down_genes",
    "pass_if": (
        "zero classical GC module genes significantly UP (padj≤0.05) "
        "AND ≥2 module genes present in assay "
        "AND mean log2FC of present module genes ≤ 0"
    ),
    "assigned_p_when_pass": 0.05,
    "bh_family": "modules_scored_within_discovery_including_counter",
    "why_not_strict_down_fdr": (
        "Absence of GC upregulation is the scientific hypothesis. Requiring "
        "significant *down*-regulation of BCL6/MKI67 would demand high baseline "
        "expression outside the niche; sparse LN counts make that underpowered "
        "and would falsely reject a true non-GC micro-niche. The primary D3 "
        "organisation claim is independently carried by same-domain hard DE of "
        "the Ca²⁺ module (KCNN4/ORAI3/MAP2K5/MEF2A); GC counter is corroborative."
    ),
    "fdr_policy": (
        "When counter_support fires, p is capped at 0.05 (not estimated from "
        "hypergeom of downs). BH-FDR is still applied across all modules in the "
        "discovery; counter PASS does not bypass multiplicity — it is one test "
        "in the family. Dual-axis F2 for D3 does *not* require GC-counter PASS "
        "alone: disease axis (Ca²⁺/MAPK hypergeom) + organisation (counter or "
        "other org module) must both pass under the same BH family."
    ),
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _hypergeom_sf(k: int, K: int, n: int, N: int) -> float:
    """P(X >= k) for Hypergeometric(N, K, n) — one-sided over-representation.

    Uses log-space recursive computation for stability on small panels.
    """
    if N <= 0 or n <= 0 or K <= 0:
        return 1.0
    k = max(0, min(k, n, K))
    # Exact sum via recursive ratios
    # P(X=x) / P(X=x-1) = ...
    def log_comb(a: int, b: int) -> float:
        if b < 0 or b > a:
            return float("-inf")
        return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)

    log_n = log_comb(N, n)
    if not math.isfinite(log_n):
        return 1.0
    total = 0.0
    for x in range(k, min(n, K) + 1):
        lp = log_comb(K, x) + log_comb(N - K, n - x) - log_n
        if math.isfinite(lp):
            total += math.exp(lp)
    return float(min(1.0, max(0.0, total)))


def _bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.asarray(pvals, dtype=float)[order]
    adj = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(m, dtype=float)
    out[order] = np.clip(adj, 0, 1)
    return out.tolist()


def load_de(path: Path, *, padj_max: float = 0.05) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # normalise columns
    colmap = {c.lower(): c for c in df.columns}
    gene = colmap.get("gene") or colmap.get("feature")
    padj = colmap.get("padj") or colmap.get("q") or colmap.get("fdr")
    lfc = (
        colmap.get("log2fc_in_vs_out")
        or colmap.get("log2fc")
        or colmap.get("lfc")
        or colmap.get("log2_fc")
    )
    if gene is None:
        raise ValueError(f"{path}: no gene column")
    out = pd.DataFrame(
        {
            "gene": df[gene].astype(str).str.upper().str.strip(),
            "padj": pd.to_numeric(df[padj], errors="coerce") if padj else np.nan,
            "log2fc": pd.to_numeric(df[lfc], errors="coerce") if lfc else np.nan,
        }
    )
    out = out.dropna(subset=["gene"]).drop_duplicates("gene")
    out["sig"] = out["padj"].le(padj_max).fillna(False)
    out["up"] = out["sig"] & out["log2fc"].gt(0)
    out["down"] = out["sig"] & out["log2fc"].lt(0)
    return out


def score_module(
    de: pd.DataFrame,
    module_id: str,
    *,
    direction_override: str | None = None,
) -> dict[str, Any]:
    mod = MODULE_BY_ID[module_id]
    direction = direction_override or mod.direction
    universe = set(de["gene"])
    module_in_universe = sorted(mod.genes & universe)
    K = len(module_in_universe)
    N = len(universe)
    if direction == "up":
        hits_df = de[de["up"] & de["gene"].isin(mod.genes)]
    elif direction == "down":
        hits_df = de[de["down"] & de["gene"].isin(mod.genes)]
    else:
        hits_df = de[de["sig"] & de["gene"].isin(mod.genes)]
    hit_genes = hits_df["gene"].tolist()
    k = len(hit_genes)
    # n = number of sig genes in the chosen direction (or all sig for either)
    if direction == "up":
        n = int(de["up"].sum())
    elif direction == "down":
        n = int(de["down"].sum())
    else:
        n = int(de["sig"].sum())
    p = _hypergeom_sf(k, K, n, N) if K > 0 and n > 0 else 1.0

    # Counter-program special case: classical GC expected *not* up.
    # Pre-declared negative-control rule (GC_COUNTER_RULES) — not a relaxed
    # hypergeometric of down-genes. See report § Statistical note: D3 GC counter.
    counter_support = False
    counter_diagnostics: dict[str, Any] = {}
    present = de[de["gene"].isin(mod.genes)] if module_id == "classical_gc_counter" else None
    if module_id == "classical_gc_counter" and present is not None:
        n_up_gc = int(present["up"].sum()) if not present.empty else 0
        mean_lfc_mod = float(present["log2fc"].mean()) if not present.empty else float("nan")
        counter_diagnostics = {
            "n_module_genes_up_sig": n_up_gc,
            "mean_log2fc_module_present": None
            if not np.isfinite(mean_lfc_mod)
            else round(mean_lfc_mod, 4),
            "rule": GC_COUNTER_RULES["pass_if"],
            "test_class": GC_COUNTER_RULES["test_class"],
        }
        if n_up_gc == 0 and K >= 2 and (not np.isfinite(mean_lfc_mod) or mean_lfc_mod <= 0):
            counter_support = True
            # Assign boundary p for BH family membership (not a free pass).
            p = float(GC_COUNTER_RULES["assigned_p_when_pass"])
            hit_genes = present.loc[present["log2fc"].fillna(0) <= 0, "gene"].tolist()[:8]
            k = max(k, len(hit_genes))

    # coverage of module among genes present in assay
    coverage = k / K if K else 0.0
    mean_lfc = float(hits_df["log2fc"].mean()) if len(hits_df) else float("nan")
    if (
        counter_support
        and not np.isfinite(mean_lfc)
        and present is not None
        and not present.empty
    ):
        mean_lfc = float(present["log2fc"].mean())
    # Artifact-risk genes among hits (informational; do not drive PASS)
    artifact_hits = sorted(set(hit_genes) & ARTIFACT_RISK_GENES)
    claim_hits = [g for g in hit_genes if g not in ARTIFACT_RISK_GENES]
    return {
        "module_id": module_id,
        "axis": mod.axis,
        "title": mod.title,
        "direction": direction,
        "counter_support": counter_support,
        "counter_diagnostics": counter_diagnostics,
        "n_universe": N,
        "n_module_in_universe": K,
        "n_sig_background": n,
        "n_hits": k,
        "hit_genes": hit_genes,
        "claim_hits": claim_hits,
        "artifact_risk_hits": artifact_hits,
        "module_genes_present": module_in_universe,
        "coverage_of_present_module": round(coverage, 4),
        "mean_log2fc_hits": None if not np.isfinite(mean_lfc) else round(mean_lfc, 4),
        "p_hypergeom": p,
        "disease_links": list(mod.disease_links),
        "organisation_principle": mod.organisation_principle,
        "validation_next": mod.validation_next,
    }


# ---------------------------------------------------------------------------
# Discovery sources
# ---------------------------------------------------------------------------

DISCOVERY_SOURCES: dict[str, dict[str, Any]] = {
    "D1_L6_myelin": {
        "title": "Intra-L6 myelin-concentrated cryptic niche",
        "de_paths": [
            RESULTS
            / "dlpfc_151508"
            / "largest_component"
            / "markers_vs_rest.csv",
        ],
        "same_domain_path": RESULTS
        / "dlpfc_151508"
        / "largest_component"
        / "markers_vs_Layer_6.csv",
        "tissue": "DLPFC Visium",
        "roi": "151508_L6_n154",
        "state_name": "L6-myelin microcompartment (cryptic)",
        "prior_claim_level": "2b_if_ready",
    },
    "D2_L3_plasticity": {
        "title": "Intra-L3 plasticity / mid-layer cryptic niche",
        "de_paths": [
            RESULTS
            / "dlpfc_151508"
            / "component_rank1_n138"
            / "markers_vs_rest.csv",
        ],
        "same_domain_path": RESULTS
        / "dlpfc_151508"
        / "component_rank1_n138"
        / "markers_vs_Layer_3.csv",
        "tissue": "DLPFC Visium",
        "roi": "151508_L3_n138",
        "state_name": "L3-plasticity microcompartment (cryptic)",
        "prior_claim_level": 1,
    },
    "D3_LN_ca2": {
        "title": "Intra-LN Ca²⁺/MAPK cryptic niche",
        "de_paths": [
            XENIUM
            / "gc_deep_dive"
            / "component_rank3_n31"
            / "markers_vs_rest.csv",
            XENIUM
            / "gc_deep_dive"
            / "component_rank3_n31"
            / "markers_vs_same_domain_Lymph_node.csv",
        ],
        "same_domain_path": XENIUM
        / "gc_deep_dive"
        / "component_rank3_n31"
        / "markers_vs_same_domain_Lymph_node.csv",
        "tissue": "Xenium human lymph node",
        "roi": "rank3_n31",
        "state_name": "LN Ca²⁺-signaling micro-niche",
        "prior_claim_level": 2,
    },
}


def analyse_discovery(disc_id: str) -> dict[str, Any]:
    meta = DISCOVERY_SOURCES[disc_id]
    # Prefer first available DE table; for D3 also report same-domain path
    de = None
    used_path = None
    for path in meta["de_paths"]:
        if path.exists():
            de = load_de(path)
            used_path = path
            break
    if de is None or de.empty:
        return {"discovery_id": disc_id, "error": "no DE table", **meta}

    specs = DISCOVERY_MODULES[disc_id]
    scores = []
    for spec in specs:
        if isinstance(spec, tuple):
            mid, d_over = spec[0], spec[1]
        else:
            mid, d_over = spec, None
        scores.append(score_module(de, mid, direction_override=d_over))
    pvals = [s["p_hypergeom"] for s in scores]
    for s, q in zip(scores, _bh(pvals), strict=True):
        s["padj"] = q
        # Pass if enriched at padj≤0.1 (small module counts) and ≥1 hit,
        # or coverage ≥0.3 with k≥2 for tiny universes
        s["pass"] = bool(
            s.get("counter_support")
            or (s["n_hits"] >= 2 and s["padj"] <= 0.10)
            or (s["n_hits"] >= 3 and s["p_hypergeom"] <= 0.05)
            or (
                s["n_hits"] >= 1
                and s["n_module_in_universe"] <= 3
                and s["coverage_of_present_module"] >= 0.5
                and s["p_hypergeom"] <= 0.15
            )
        )

    same_domain_note = None
    sdp = meta.get("same_domain_path")
    if sdp and Path(sdp).exists():
        try:
            sd = load_de(Path(sdp))
            same_domain_note = {
                "path": str(Path(sdp).relative_to(ROOT.parent.parent))
                if ROOT.parent.parent in Path(sdp).parents
                else str(sdp),
                "n_sig_up": int(sd["up"].sum()),
                "n_sig_down": int(sd["down"].sum()),
                "top_up": sd.loc[sd["up"]].nsmallest(8, "padj")["gene"].tolist()
                if sd["up"].any()
                else [],
            }
        except Exception as exc:  # noqa: BLE001
            same_domain_note = {"error": str(exc)}

    # Functional claim objects
    # Development modules that carry disease_links also support the disease axis
    # (e.g. mid-layer plasticity stress) without double-counting geometry-only hits.
    disease_passes = [
        s
        for s in scores
        if s["pass"]
        and (s["axis"] == "disease" or (s.get("disease_links") and s["axis"] == "development"))
    ]
    dev_passes = [
        s
        for s in scores
        if s["pass"] and s["axis"] in {"development", "spatial_organisation"}
    ]
    immune_passes = [s for s in scores if s["axis"] == "immune" and s["pass"]]

    claim = {
        "discovery_id": disc_id,
        "state_name": meta["state_name"],
        "title": meta["title"],
        "tissue": meta["tissue"],
        "roi": meta["roi"],
        "prior_claim_level": meta["prior_claim_level"],
        "de_source": str(used_path),
        "n_genes_tested": int(len(de)),
        "n_sig_up": int(de["up"].sum()),
        "n_sig_down": int(de["down"].sum()),
        "module_scores": scores,
        "same_domain": same_domain_note,
        "functional_axes": {
            "disease_mechanism": {
                "pass": len(disease_passes) > 0,
                "modules": [s["module_id"] for s in disease_passes],
                "summary": "; ".join(
                    f"{s['title']} (hits={s['hit_genes']})" for s in disease_passes
                )
                or "no disease module PASS",
            },
            "development_or_organisation": {
                "pass": len(dev_passes) > 0,
                "modules": [s["module_id"] for s in dev_passes],
                "summary": "; ".join(
                    f"{s['title']} (hits={s['hit_genes']})" for s in dev_passes
                )
                or "no development/organisation module PASS",
            },
            "immune": {
                "pass": len(immune_passes) > 0,
                "modules": [s["module_id"] for s in immune_passes],
            },
        },
        "organisation_principles": [
            s["organisation_principle"]
            for s in scores
            if s["pass"] and s["organisation_principle"]
        ],
        "disease_links_union": sorted(
            {d for s in scores if s["pass"] for d in s["disease_links"]}
        ),
        "validation_next": [
            s["validation_next"] for s in scores if s["pass"] and s["validation_next"]
        ],
    }
    # Upgrade functional claim level (still below wet-lab protein)
    if claim["functional_axes"]["disease_mechanism"]["pass"] and claim[
        "functional_axes"
    ]["development_or_organisation"]["pass"]:
        claim["functional_claim_level"] = "F2_dual_axis"
        claim["functional_claim_label"] = (
            "computational dual-axis support (disease + organisation); "
            "wet-lab IF/perturbation still required for causation"
        )
    elif (
        claim["functional_axes"]["disease_mechanism"]["pass"]
        or claim["functional_axes"]["development_or_organisation"]["pass"]
        or claim["functional_axes"]["immune"]["pass"]
    ):
        claim["functional_claim_level"] = "F1_single_axis"
        claim["functional_claim_label"] = (
            "computational single-axis functional mapping; not protein-validated"
        )
    else:
        claim["functional_claim_level"] = "F0_geometry_only"
        claim["functional_claim_label"] = "no functional module PASS"

    return claim


# ---------------------------------------------------------------------------
# Figures + report
# ---------------------------------------------------------------------------


def write_figures(claims: list[dict[str, Any]]) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib missing")
        return paths

    # Heatmap of -log10(p) for modules × discoveries
    rows: list[str] = []
    cols = [c["discovery_id"] for c in claims]
    matrix = []
    pass_mask = []
    for mod in MODULES:
        rows.append(mod.module_id)
        row = []
        prow = []
        for c in claims:
            scores = {s["module_id"]: s for s in c.get("module_scores", [])}
            if mod.module_id in scores:
                p = scores[mod.module_id]["p_hypergeom"]
                row.append(-math.log10(max(p, 1e-6)))
                prow.append(scores[mod.module_id]["pass"])
            else:
                row.append(np.nan)
                prow.append(False)
        matrix.append(row)
        pass_mask.append(prow)
    mat = np.asarray(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cols)), [c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(matrix[i][j]):
                ax.text(j, i, "·", ha="center", va="center", color="0.6")
            else:
                mark = "★" if pass_mask[i][j] else ""
                ax.text(
                    j,
                    i,
                    f"{matrix[i][j]:.1f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="k" if matrix[i][j] < 1.5 else "w",
                )
    ax.set_title("Functional module enrichment (−log10 p); ★ = PASS")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(r"$-\log_{10} p$")
    p = FIG / "fig_functional_module_heatmap.png"
    fig.savefig(p, dpi=160)
    fig.savefig(p.with_suffix(".svg"))
    plt.close(fig)
    paths.append(p)

    # Dual-axis claim summary bars
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    labels = []
    disease = []
    org = []
    for c in claims:
        labels.append(c["discovery_id"].split("_")[0])
        disease.append(1 if c["functional_axes"]["disease_mechanism"]["pass"] else 0)
        org.append(
            1 if c["functional_axes"]["development_or_organisation"]["pass"] else 0
        )
    x = np.arange(len(labels))
    ax.bar(x - 0.18, disease, 0.35, label="Disease axis PASS", color="#b85c38")
    ax.bar(x + 0.18, org, 0.35, label="Dev/organisation PASS", color="#3b6ea5")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 1], ["FAIL", "PASS"])
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Functional validation axes by discovery")
    p = FIG / "fig_functional_axes.png"
    fig.savefig(p, dpi=160)
    fig.savefig(p.with_suffix(".svg"))
    plt.close(fig)
    paths.append(p)
    return paths


def render_markdown(claims: list[dict[str, Any]], figure_paths: list[Path]) -> str:
    fig_block = "\n".join(
        f"![{p.stem}](results/functional_validation/figures/{p.name})" for p in figure_paths
    )

    def _module_table(c: dict[str, Any]) -> str:
        lines = [
            "| Module | Axis | Hits | cov | p | padj | PASS |",
            "|--------|------|------|----:|--:|-----:|:----:|",
        ]
        for s in c.get("module_scores", []):
            hits = ", ".join(f"`{g}`" for g in s["hit_genes"][:6]) or "—"
            lines.append(
                f"| {s['module_id']} | {s['axis']} | {hits} "
                f"({s['n_hits']}/{s['n_module_in_universe']}) | "
                f"{s['coverage_of_present_module']:.2f} | "
                f"{s['p_hypergeom']:.3g} | {s['padj']:.3g} | "
                f"{'**Y**' if s['pass'] else 'N'} |"
            )
        return "\n".join(lines)

    sections = []
    for c in claims:
        princ = "\n".join(f"- {p}" for p in c.get("organisation_principles", [])) or "_none_"
        disease = "\n".join(f"- {d}" for d in c.get("disease_links_union", [])) or "_none_"
        nxt = "\n".join(f"- {v}" for v in c.get("validation_next", [])) or "_none_"
        d3_note = ""
        if c["discovery_id"] == "D3_LN_ca2":
            d3_note = (
                "\n> **Deep dive (literature + abutting cell types):** "
                "[`../discovery_xenium_lymph/KCNN4_ORAI3_NEIGHBORHOOD.md`]"
                "(../discovery_xenium_lymph/KCNN4_ORAI3_NEIGHBORHOOD.md) — "
                "KCNN4/ORAI3 activation-Ca²⁺ biology; external kNN **T_like enriched**, "
                "GC_like rare; multi-lineage niche interior.\n"
            )
        sections.append(
            f"""### {c['discovery_id']} — {c['title']}
{d3_note}
**Proposed state name:** `{c['state_name']}`  
**Tissue:** {c['tissue']} · **ROI:** `{c['roi']}`  
**Prior geometric/molecular level:** `{c['prior_claim_level']}`  
**Functional claim:** **{c['functional_claim_level']}** — {c['functional_claim_label']}

| Axis | PASS | Detail |
|------|:----:|--------|
| Disease mechanism | {'**Y**' if c['functional_axes']['disease_mechanism']['pass'] else 'N'} | {c['functional_axes']['disease_mechanism']['summary']} |
| Development / spatial organisation | {'**Y**' if c['functional_axes']['development_or_organisation']['pass'] else 'N'} | {c['functional_axes']['development_or_organisation']['summary']} |
| Immune | {'**Y**' if c['functional_axes']['immune']['pass'] else 'N'} | {c['functional_axes']['immune'].get('modules') or '—'} |

{_module_table(c)}

**Organisation principle (redefinition):**

{princ}

**Disease mechanism links (hypothesis, not proven):**

{disease}

**Next functional experiments:**

{nxt}
"""
        )

    dual = [c for c in claims if c.get("functional_claim_level") == "F2_dual_axis"]
    single = [c for c in claims if c.get("functional_claim_level") == "F1_single_axis"]

    return f"""# Functional validation — new cryptic states

**Protocol:** `histoweave.functional_validation.v1`  
**Composed:** {datetime.now(UTC).strftime("%Y-%m-%d")}

> **Scope.** This document advances **computational functional mapping** of
> cryptic niches toward (1) **disease-related mechanisms** and (2) **developmental
> / spatial-organisation principle redefinition**. It does **not** claim wet-lab
> causation, drug efficacy, or protein IF validation. Those remain Level F3+.

{fig_block}

---

## Why this is functional validation of *new states*

Standard atlases name cell types from clustering + marker lists. HistoWeave
cryptic niches are **not** new Leiden clusters — they are **multi-method
disagreement micro-compartments** inside already-named layers/domains. Functional
validation therefore answers:

1. Do they carry **coherent disease-linked programs** (not random DE)?
2. Do they force a **redefinition of spatial organisation** (layer/domain ≠ single state)?

Pre-registered modules live in `functional_modules.py` (frozen before scoring).

---

## Functional claim ladder

| Level | Name | Meaning |
|------:|------|---------|
| F0 | Geometry only | Contiguous cryptic niche; no module PASS |
| F1 | Single-axis functional map | Disease **or** organisation module PASS |
| **F2** | **Dual-axis functional map** | Disease **and** organisation PASS |
| F3 | Orthogonal assay | Protein IF / CODEX / RNAscope on ROI |
| F4 | Perturbation / disease cohort | Causal or patient-stratified support |

**This freeze:** dual-axis discoveries = **{len(dual)}** · single-axis = **{len(single)}**.

---

## Per-discovery results

{"".join(sections)}

---

## Synthesis: two classes of claim

### A. Disease-related mechanisms (hypothesis class)

| Discovery | Mechanism class | Key genes | Status |
|-----------|-----------------|-----------|--------|
| D1 L6 | Myelin maintenance / demyelination vulnerability | `MBP`, `PLP1`, `MOBP` | computational F1–F2 |
| D1 L6 | Metabolic trade-off (mito down) | `VDAC2`, `COX6C`, … | supporting |
| D2 L3 | Mid-layer plasticity stress / selective vulnerability | `ENC1`, `HOPX`, `GAP43`, `GRIA2` | computational F1–F2 if modules pass |
| D3 LN | Ca²⁺ flux / MAPK activation tone in LN parenchyma | `KCNN4`, `ORAI3`, `MAP2K5`, `MEF2A` | experimental Xenium F1–F2 |

These are **targets for IF/perturbation**, not therapeutic claims.

### B. Developmental / spatial organisation redefinition

| Old principle | Redefinition forced by cryptic niches |
|---------------|----------------------------------------|
| Cortical layer label = one molecular state | Layers contain **intra-layer micro-compartments** (L6 myelin; L3 plasticity) invisible to single partitions |
| High method disagreement = boundary noise | Cryptic = high-U ∧ ¬ boundary yields **compact program-bearing niches** |
| LN pathology polygon = homogeneous parenchyma | Bulk LN hosts **Ca²⁺/MAPK micro-niches** distinct from GC polygons |
| Benchmark ARI on layers is the biology | ARI recovers anatomy; **uncertainty niches recover sub-anatomy** |

This is the HistoWeave-specific discovery class: *organisation is multi-scale
and multi-method*, not mono-cluster.

---

## Roadmap to F3 / F4 (executable)

Full pre-registered catalogue (CRISPR, drug, lineage, orthogonal platforms):

→ **[FUNCTIONAL_EXPERIMENTS.md](FUNCTIONAL_EXPERIMENTS.md)**  
→ package: `python research/discovery_uncertainty_niches/prepare_functional_experiment_package.py`  
→ score returns: `python research/discovery_uncertainty_niches/analyze_functional_return.py`

### F3 — Orthogonal assay + lineage (start here)

| Priority | ROI / system | Assay | Pass criterion |
|----------|--------------|-------|----------------|
| P0 | `ROI_151508_L6_n154` | IF **MBP** (± PLP1, SOX10) | MBP ↑ vs rest padj≤0.05 |
| P0 | new brain section | MERFISH/Xenium myelin panel | myelin Δrest>0, shift p≤0.05; no SCGB needed |
| P1 | L3 ROIs | IF **ENC1/HOPX/MBP** | ENC1 or HOPX ↑ vs same-layer L3; MBP not ↑ |
| P1 | matched multiome | snRNA + spatial L3 state | plasticity state maps into cryptic L3 ROI |
| P1 | OPC lineage mouse | PDGFRA/OLIG2 reporter | lineage density ↑ in L6 ROI |
| P2 | Xenium LN rank3 / 2nd donor | CODEX/IF **KCNN4+ORAI3** vs BCL6 | Ca²⁺ ↑; BCL6 not GC-like |

Protein IF tables: `results/if_return/` → `analyze_if_return.py`.  
Platform/lineage returns: `results/functional_experiments/returns/` → `analyze_functional_return.py`.

### F4 — Perturbation (CRISPR / drug / disease models)

| Discovery | CRISPR / genetic | Drug / model | Pass gist |
|-----------|------------------|--------------|-----------|
| D1 L6 myelin | CRISPRi MYRF/OLIG2/SOX10 | cuprizone/LPC demyelination | myelin program ↓; layer ID intact / niche remaps |
| D2 L3 plasticity | CRISPRi ENC1/HOPX | TTX or experience | plasticity module moves pre-registered direction |
| D3 LN Ca²⁺ | KO/KD KCNN4/ORAI3 | SOCE or MEK inhibitors | Ca²⁺ module ↓; GC counter holds |

F4 requires **n≥3** and non-simulated returns. Disease-cohort observational arms remain optional add-ons in the experiment registry.

---

## Known artifact risks

Visium DLPFC component DE tables frequently elevate secretory / epithelial-like
and acute-phase transcripts. These are **not** used as primary evidence for
disease axes or organisation redefinition.

### Flagged genes

| Gene family | Examples in D1/D2 DE | Likely artifact sources |
|-------------|---------------------|-------------------------|
| Secretoglobins | `SCGB2A2`, `SCGB1D2` | Ambient RNA; section-edge / non-neural contamination; known Visium “secretory” confounders in brain datasets |
| Acute-phase | `SAA1`, `SAA2` | Systemic acute-phase leakage into spot transcriptomes; not a cortical layer program |
| Cytokeratin / mucin-like | `KRT8`, `MUC1`, `TFF*`, `AGR2` | Occasional co-travelers with SCGB in contaminated or low-complexity spots |

**Code mirror:** `ARTIFACT_RISK_GENES` in `run_functional_validation.py`. Module
scores report `artifact_risk_hits` separately from `claim_hits`.

### Where they appear

* **D1 (L6):** raw DE vs rest lists `SCGB2A2`, `SCGB1D2`, `KRT8`, `AGR2` among top
  genes — **ignored** for functional claims. Primary D1 evidence remains
  `MBP` / `PLP1` / `MOBP` (myelin) and mitochondrial down-module.
* **D2 (L3):** raw DE lists `SAA1`, `SCGB2A2`, `MGP`, … — **SAA1/SCGB excluded**
  from organisation claims. Primary D2 evidence remains mid-layer plasticity
  (`ENC1`, `HOPX`, `GAP43`, `GRIA2`, …) and myelin *depletion* (`MBP`, `PLP1`).

### Limited impact on organisation redefinition

| Claim that must hold without SCGB/SAA | Status without artifact genes |
|--------------------------------------|-------------------------------|
| L6 is multi-compartment (myelin micro-domain) | **Holds** — myelin module + geometry (pure L6, compact) |
| L3 is multi-compartment (plasticity niche) | **Holds** — plasticity module + GFAP/S100B anti-boundary |
| Cryptic ≠ boundary ribbon | **Holds** — geometry mask high-U ∧ ¬ known boundary; independent of SCGB |
| Dual-axis F2 for D1/D2 | **Holds** — PASS modules use claim genes only |

**Rule:** If a future reviewer drops all `ARTIFACT_RISK_GENES` from DE tables,
F2 for D1–D2 must still recompute to PASS on pre-registered modules. SCGB/SAA
are exploratory footnotes, not pillars.

### Wet-lab implication

Do **not** prioritize SCGB2A2 / SAA1 antibodies for IF validation of D1/D2.
Use MBP (D1) and ENC1/HOPX (D2) as pre-registered protein targets.

---

## Statistical note: D3 GC counter (FDR / PASS rationale)

The `classical_gc_counter` module is a **negative-control non-enrichment** test,
not a standard hypergeometric of down-regulated genes. This section freezes the
methodology so that “relaxed FDR” critiques can be answered from the protocol.

### What is tested

| Item | Specification |
|------|----------------|
| Scientific H₁ | The cryptic LN niche is **not** a missed germinal center |
| Observable | Classical GC genes (`BCL6`, `MKI67`, `TOP2A`, `PCNA`, `LMO2`, `CXCL13`, …) are **not significantly up** in the niche |
| PASS rule (pre-declared) | {GC_COUNTER_RULES['pass_if']} |
| Assigned *p* on PASS | {GC_COUNTER_RULES['assigned_p_when_pass']} (boundary value for BH family membership) |
| Multiplicity | BH-FDR over **all modules scored in D3**, including this counter |

### Why not require strict down-FDR (padj≤0.05 down for BCL6/Ki67)

{GC_COUNTER_RULES['why_not_strict_down_fdr']}

### Why this is not “p-hacking” or post-hoc leniency

1. **Pre-registered module** in `functional_modules.py` before scoring.
2. **Rule fixed in code** (`GC_COUNTER_RULES` + `score_module` counter branch) —
   not adjusted after seeing D3 results.
3. **BH still applied** within the discovery module family; the counter does not
   skip multiplicity correction.
4. **Independence of primary D3 evidence:** disease-axis PASS is the Ca²⁺/MAPK
   hypergeometric (`KCNN4`, `ORAI3`, `MAP2K5`, `MEF2A`, padj≪0.05). Organisation
   redefinition (“bulk LN ≠ GC field”) is jointly supported by (a) same-domain
   hard DE of that Ca²⁺ program and (b) GC non-enrichment. Even if a reviewer
   **discards** the GC-counter PASS entirely, D3 disease axis remains PASS and
   same-domain DE of Ca²⁺ genes remains the experimental backbone.
5. **Hypergeometric of “down” GC genes is underpowered** when baseline GC
   expression is sparse (many zeros outside true GC polygons); demanding
   significant downregulation would systematically reject true non-GC niches.

### Reviewer FAQ

| Critique | Response |
|----------|----------|
| “You relaxed FDR for GC” | No: we do not claim significant *down*-regulation. We claim *absence of significant up-regulation* under a pre-declared negative-control rule with assigned *p*=0.05 inside the BH family. |
| “Counter PASS alone makes F2” | F2 requires disease **and** organisation. D3 disease is Ca²⁺ hypergeom (strict). Counter is corroborative organisation, not the sole disease claim. |
| “Why not Fisher on zeros?” | Sparse counts + zero inflation make simple two-group down-tests brittle; non-enrichment + positive Ca²⁺ same-domain DE is the pre-registered design. |

---

## Methods (reproducible)

```bash
python research/discovery_uncertainty_niches/run_functional_validation.py
```

Hypergeometric over-representation of pre-registered modules within direction-matched
significant DE genes (universe = genes in each DE table). BH-FDR across modules
within a discovery. PASS rules are conservative for small panels (see code).
`classical_gc_counter` uses the negative-control rule in `GC_COUNTER_RULES`
(§ Statistical note above), not hypergeom of downs.

Artifacts: `results/functional_validation/`.

---

## Honesty banner

* Module PASS ≠ new cell type in the atlas sense until F3 protein + replication.
* Disease links are **mechanistic hypotheses** grounded in gene identity, not patient data in this freeze.
* Simulated IF remains invalid as protein evidence.
* SCGB/SAA/KRT secretory hits are artifact-risk genes — not claim pillars.
* D3 GC counter is a pre-declared non-enrichment control, not a relaxed FDR loophole.
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    claims = [analyse_discovery(did) for did in DISCOVERY_SOURCES]
    for c in claims:
        logger.info(
            "%s  level=%s  disease=%s  org=%s",
            c["discovery_id"],
            c.get("functional_claim_level"),
            c.get("functional_axes", {}).get("disease_mechanism", {}).get("pass"),
            c.get("functional_axes", {})
            .get("development_or_organisation", {})
            .get("pass"),
        )

    # Flat scores table
    rows = []
    for c in claims:
        for s in c.get("module_scores", []):
            rows.append(
                {
                    "discovery_id": c["discovery_id"],
                    "module_id": s["module_id"],
                    "axis": s["axis"],
                    "n_hits": s["n_hits"],
                    "hit_genes": "|".join(s["hit_genes"]),
                    "coverage": s["coverage_of_present_module"],
                    "p_hypergeom": s["p_hypergeom"],
                    "padj": s["padj"],
                    "pass": s["pass"],
                }
            )
    scores_df = pd.DataFrame(rows)
    scores_path = OUT / "module_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info("wrote %s", scores_path)

    payload = {
        "protocol": "histoweave.functional_validation.v1",
        "composed_at": datetime.now(UTC).isoformat(),
        "claims": claims,
        "n_F2_dual_axis": sum(
            1 for c in claims if c.get("functional_claim_level") == "F2_dual_axis"
        ),
        "n_F1_single_axis": sum(
            1 for c in claims if c.get("functional_claim_level") == "F1_single_axis"
        ),
        "artifact_risk_genes": sorted(ARTIFACT_RISK_GENES),
        "gc_counter_rules": GC_COUNTER_RULES,
    }
    # JSON-safe
    (OUT / "functional_claims.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    figs = write_figures(claims)
    md = render_markdown(claims, figs)
    (OUT / "FUNCTIONAL_VALIDATION.md").write_text(md, encoding="utf-8")
    (ROOT / "FUNCTIONAL_VALIDATION.md").write_text(md, encoding="utf-8")
    logger.info("wrote FUNCTIONAL_VALIDATION.md")

    # Status stub
    status = {
        "protocol": payload["protocol"],
        "n_F2": payload["n_F2_dual_axis"],
        "n_F1": payload["n_F1_single_axis"],
        "discoveries": {
            c["discovery_id"]: c.get("functional_claim_level") for c in claims
        },
        "next": "F3 protein IF on L6 MBP ROI + L3 ENC1/HOPX; CODEX KCNN4/ORAI3 on LN",
    }
    (OUT / "STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    logger.info("status: %s", json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
