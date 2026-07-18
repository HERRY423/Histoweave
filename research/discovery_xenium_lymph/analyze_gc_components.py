"""GC-enriched cryptic-component deep-dive for Xenium lymph node.

Mirrors ``research/discovery_uncertainty_niches/analyze_largest_component.py``
(DLPFC) with lymph-node pathology domains and pre-registered GC/B/T panels.

Selection policy (in order):

1. Components whose dominant pathology label is GC / lymphoid-aggregate.
2. Else components ranked by ``Germinal_center_delta_rest`` (panel score).
3. Always also analyse the largest cryptic component for cross-tissue parity
   with DLPFC ``largest_component``.

Outputs under ``results/gc_deep_dive/``:

* ``selection.json`` — which components were chosen and why
* ``component_rank{r}_n{n}/`` — DE + adjacency + report (DLPFC-style)
* ``GC_DEEP_DIVE_REPORT.md`` — cross-component synthesis + DLPFC对照
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT_ROOT = BASE / "results"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "discovery_uncertainty_niches"))

from analyze_largest_component import (  # noqa: E402
    BAN_PREFIXES,
    cryptic_components,
    differential_markers,
    log1p_norm,
)
from validate_panel_and_rois import composite_score  # noqa: E402

from histoweave._math import knn_indices  # noqa: E402
from histoweave.datasets import get_dataset  # noqa: E402

logger = logging.getLogger("gc_deep_dive")

DATASET = "xenium_human_lymph_node"
K_NN = 8
MIN_COMPONENT = 30
SEED = 0

B_PANEL = ("MS4A1", "CD19", "CR2", "CD79A", "PAX5", "CD22", "CD79B", "FCER2")
T_PANEL = ("CD3E", "CD4", "CD8A", "CCR7", "IL7R", "IL7", "LTB", "TRAC")
GC_PANEL = ("BCL6", "MKI67", "TOP2A", "PCNA", "LMO2", "CXCL13", "AICDA", "RGS13")
# Extended GC-adjacent genes often on Xenium multi-tissue panels
GC_EXTENDED = GC_PANEL + (
    "MEF2B",
    "GCSAM",
    "FCER2",
    "CD22",
    "CD79B",
    "BANK1",
    "BLK",
)


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _to_dense(m) -> np.ndarray:
    if hasattr(m, "toarray"):
        return np.asarray(m.toarray(), dtype=float)
    return np.asarray(m, dtype=float)


def adjacency_profile_generic(
    coords: np.ndarray,
    labels: np.ndarray,
    component: np.ndarray,
    *,
    k: int = K_NN,
    domain_order: list[str] | None = None,
) -> dict:
    """Domain-agnostic adjacency (LN pathology labels, not cortical layers)."""
    nbrs = knn_indices(coords, k + 1)
    component_set = set(int(i) for i in component)
    contact_counts: Counter[str] = Counter()
    per_spot_primary: list[str] = []
    external_edges = 0
    internal_edges = 0
    for i in component:
        i = int(i)
        neigh_labs: list[str] = []
        for v in nbrs[i]:
            v = int(v)
            if v == i:
                continue
            if v in component_set:
                internal_edges += 1
            else:
                external_edges += 1
                lab = str(labels[v])
                neigh_labs.append(lab)
                contact_counts[lab] += 1
        if neigh_labs:
            per_spot_primary.append(Counter(neigh_labs).most_common(1)[0][0])
        else:
            per_spot_primary.append("internal_only")

    total_ext = sum(contact_counts.values()) or 1
    non = np.array([i for i in range(len(labels)) if i not in component_set])
    bg = Counter(str(labels[i]) for i in non)
    bg_total = sum(bg.values()) or 1

    order = domain_order or sorted(set(list(contact_counts) + list(bg)))
    rows = []
    for layer in order:
        c = contact_counts.get(layer, 0)
        frac = c / total_ext
        bg_frac = bg.get(layer, 0) / bg_total
        enrich = frac / bg_frac if bg_frac > 0 else float("nan")
        rows.append(
            {
                "layer": layer,
                "external_neighbour_counts": c,
                "fraction_of_external_contacts": round(frac, 4),
                "background_layer_fraction": round(bg_frac, 4),
                "enrichment_vs_background": round(enrich, 3) if np.isfinite(enrich) else None,
                "n_spots_primary_abut": int(sum(1 for p in per_spot_primary if p == layer)),
            }
        )
    # include any residual labels not in order
    for layer, c in contact_counts.items():
        if layer not in order:
            frac = c / total_ext
            bg_frac = bg.get(layer, 0) / bg_total
            enrich = frac / bg_frac if bg_frac > 0 else float("nan")
            rows.append(
                {
                    "layer": layer,
                    "external_neighbour_counts": c,
                    "fraction_of_external_contacts": round(frac, 4),
                    "background_layer_fraction": round(bg_frac, 4),
                    "enrichment_vs_background": round(enrich, 3) if np.isfinite(enrich) else None,
                    "n_spots_primary_abut": int(sum(1 for p in per_spot_primary if p == layer)),
                }
            )
    primary_dist = Counter(per_spot_primary)
    return {
        "n_component": len(component),
        "internal_edges": internal_edges,
        "external_edges": external_edges,
        "internal_edge_fraction": internal_edges / max(internal_edges + external_edges, 1),
        "primary_abutment_distribution": dict(primary_dist),
        "top_abutting_layers": [
            r["layer"]
            for r in sorted(rows, key=lambda r: -r["external_neighbour_counts"])[:3]
            if r["external_neighbour_counts"] > 0
        ],
        "layer_table": rows,
    }


def _panel_deltas(X: np.ndarray, genes: list[str], in_mask: np.ndarray) -> dict[str, float]:
    rest = ~in_mask
    out: dict[str, float] = {}
    for name, panel in (
        ("B_follicle", B_PANEL),
        ("T_zone", T_PANEL),
        ("Germinal_center", GC_PANEL),
    ):
        score, used = composite_score(X, genes, panel)
        if not used:
            out[f"{name}_delta_rest"] = float("nan")
            continue
        out[f"{name}_delta_rest"] = float(score[in_mask].mean() - score[rest].mean())
        out[f"{name}_mean_in"] = float(score[in_mask].mean())
        out[f"{name}_mean_rest"] = float(score[rest].mean())
        out[f"{name}_genes_used"] = used
    return out


def select_components(
    comps: list[np.ndarray],
    truth: np.ndarray,
    X: np.ndarray,
    genes: list[str],
    *,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Pick GC-enriched + largest components for deep-dive."""
    ranked: list[dict[str, Any]] = []
    for rank, comp in enumerate(comps):
        counts = Counter(str(t) for t in truth[comp])
        top_lab, n_top = counts.most_common(1)[0]
        purity = n_top / len(comp)
        in_mask = np.zeros(len(truth), dtype=bool)
        in_mask[comp] = True
        panels = _panel_deltas(X, genes, in_mask)
        is_gc_label = (
            "germinal" in top_lab.lower()
            or "lymphoid aggregate" in top_lab.lower()
            or top_lab.lower().startswith("gc")
        )
        ranked.append(
            {
                "rank": rank,
                "n": len(comp),
                "dominant_truth": top_lab,
                "purity": purity,
                "is_gc_pathology_label": is_gc_label,
                "gc_delta_rest": panels.get("Germinal_center_delta_rest", float("nan")),
                "b_delta_rest": panels.get("B_follicle_delta_rest", float("nan")),
                "t_delta_rest": panels.get("T_zone_delta_rest", float("nan")),
            }
        )

    selected: list[dict[str, Any]] = []
    used_ranks: set[int] = set()

    # 1) pathology-GC labelled
    for rec in ranked:
        if rec["is_gc_pathology_label"] and rec["rank"] not in used_ranks:
            rec = dict(rec)
            rec["selection_reason"] = "pathology_GC_label"
            selected.append(rec)
            used_ranks.add(rec["rank"])

    # 2) highest GC panel Δrest
    by_gc = sorted(
        ranked,
        key=lambda r: (
            -1 if np.isfinite(r["gc_delta_rest"]) else 0,
            float(r["gc_delta_rest"]) if np.isfinite(r["gc_delta_rest"]) else -1e9,
        ),
        reverse=True,
    )
    for rec in by_gc:
        if len(selected) >= top_n:
            break
        if rec["rank"] in used_ranks:
            continue
        if not np.isfinite(rec["gc_delta_rest"]):
            continue
        rec = dict(rec)
        rec["selection_reason"] = "top_GC_panel_delta_rest"
        selected.append(rec)
        used_ranks.add(rec["rank"])

    # 3) always include largest (rank 0) for DLPFC parity
    if 0 not in used_ranks and ranked:
        rec = dict(ranked[0])
        rec["selection_reason"] = "largest_component_dlpfc_parity"
        selected.append(rec)
        used_ranks.add(0)

    return selected


def deep_dive_one(
    *,
    rank: int,
    component: np.ndarray,
    spots: pd.DataFrame,
    coords: np.ndarray,
    truth: np.ndarray,
    X: np.ndarray,
    genes: np.ndarray,
    domain_order: list[str],
    out_dir: Path,
    selection_meta: dict[str, Any],
    expression_source: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_df = spots.iloc[component].copy()
    comp_df["component_rank"] = rank
    comp_df["in_selected_component"] = 1
    comp_df.to_csv(out_dir / "component_spots.csv", index=False)

    truth_counts = Counter(str(t) for t in truth[component])
    truth_payload = {
        "n": len(component),
        "domain_truth_counts": dict(truth_counts),
        "domain_truth_fractions": {k: v / len(component) for k, v in truth_counts.items()},
    }
    (out_dir / "component_truth_composition.json").write_text(
        json.dumps(truth_payload, indent=2), encoding="utf-8"
    )

    adj = adjacency_profile_generic(coords, truth, component, k=K_NN, domain_order=domain_order)
    pd.DataFrame(adj["layer_table"]).to_csv(out_dir / "adjacency_to_domains.csv", index=False)
    (out_dir / "adjacency_summary.json").write_text(
        json.dumps({k: v for k, v in adj.items()}, indent=2, default=str), encoding="utf-8"
    )

    in_mask = np.zeros(len(truth), dtype=bool)
    in_mask[component] = True
    out_mask = ~in_mask

    # Panel scores
    panel_rows = []
    gene_list = list(map(str, genes))
    for name, panel in (
        ("B_follicle", B_PANEL),
        ("T_zone", T_PANEL),
        ("Germinal_center", GC_PANEL),
        ("Germinal_center_extended", GC_EXTENDED),
    ):
        score, used = composite_score(X, gene_list, panel)
        if not used:
            continue
        mean_in = float(score[in_mask].mean())
        mean_out = float(score[out_mask].mean())
        # same-domain hard contrast
        dom = max(truth_counts, key=truth_counts.get)
        same_out = (truth == dom) & out_mask
        mean_same = float(score[same_out].mean()) if same_out.sum() >= 15 else float("nan")
        panel_rows.append(
            {
                "panel": name,
                "genes_used": ",".join(used),
                "mean_in": mean_in,
                "mean_rest": mean_out,
                "delta_rest": mean_in - mean_out,
                "mean_same_domain": mean_same,
                "delta_same_domain": mean_in - mean_same
                if np.isfinite(mean_same)
                else float("nan"),
            }
        )
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(out_dir / "panel_contrasts.csv", index=False)

    # Restrict DE to top variable genes for speed (keep GC extended always)
    var = X.var(axis=0)
    ban = np.array([any(str(g).startswith(p) for p in BAN_PREFIXES) for g in genes])
    var_scored = np.where(ban, -1.0, var)
    force = [i for i, g in enumerate(genes) if str(g) in set(GC_EXTENDED + B_PANEL + T_PANEL)]
    keep = list(dict.fromkeys(force + np.argsort(var_scored)[::-1].tolist()))[
        : min(2000, X.shape[1])
    ]
    keep = np.sort(np.asarray(keep, dtype=int))
    X_sub = X[:, keep]
    genes_sub = genes[keep]

    markers_rest = differential_markers(
        X_sub, genes_sub, in_mask, out_mask, label_in=f"comp_rank{rank}", label_out="rest"
    )
    markers_rest.to_csv(out_dir / "markers_vs_rest.csv", index=False)

    markers_domains: dict[str, pd.DataFrame] = {}
    for domain in adj["top_abutting_layers"][:3]:
        dmask = (truth == domain) & out_mask
        if dmask.sum() < 20:
            logger.warning("skip DE vs %s (n=%s)", domain, int(dmask.sum()))
            continue
        frame = differential_markers(
            X_sub, genes_sub, in_mask, dmask, label_in=f"comp_rank{rank}", label_out=domain
        )
        safe = domain.replace(" ", "_").replace("+", "and")
        frame.to_csv(out_dir / f"markers_vs_{safe}.csv", index=False)
        markers_domains[domain] = frame

    if markers_domains:
        pd.concat(markers_domains.values(), ignore_index=True).to_csv(
            out_dir / "markers_vs_abutting_domains.csv", index=False
        )

    # same-domain DE (hard contrast — DLPFC L3-style)
    dom = max(truth_counts, key=truth_counts.get)
    same_out = (truth == dom) & out_mask
    markers_same = None
    if same_out.sum() >= 20:
        markers_same = differential_markers(
            X_sub,
            genes_sub,
            in_mask,
            same_out,
            label_in=f"comp_rank{rank}",
            label_out=f"same_{dom}",
        )
        safe = str(dom).replace(" ", "_").replace("+", "and")
        markers_same.to_csv(out_dir / f"markers_vs_same_domain_{safe}.csv", index=False)

    report = _write_component_report(
        rank=rank,
        n=len(component),
        truth_payload=truth_payload,
        adj=adj,
        panel_df=panel_df,
        markers_rest=markers_rest,
        markers_domains=markers_domains,
        markers_same=markers_same,
        selection_meta=selection_meta,
        expression_source=expression_source,
        out_dir=out_dir,
        same_domain=str(dom),
    )
    (out_dir / "COMPONENT_REPORT.md").write_text(report, encoding="utf-8")

    sig_up = markers_rest[(markers_rest["padj"] <= 0.05) & (markers_rest["log2fc_in_vs_out"] > 0)]
    gc_hits = [g for g in GC_EXTENDED if g in set(sig_up["gene"].astype(str))]
    return {
        "rank": rank,
        "n": len(component),
        "dominant_truth": selection_meta.get("dominant_truth"),
        "selection_reason": selection_meta.get("selection_reason"),
        "gc_delta_rest": selection_meta.get("gc_delta_rest"),
        "n_sig_up_padj05": int(len(sig_up)),
        "gc_genes_sig_up": gc_hits,
        "top_abut": adj["top_abutting_layers"],
        "internal_edge_fraction": adj["internal_edge_fraction"],
        "out_dir": str(out_dir),
    }


def _write_component_report(
    *,
    rank: int,
    n: int,
    truth_payload: dict,
    adj: dict,
    panel_df: pd.DataFrame,
    markers_rest: pd.DataFrame,
    markers_domains: dict[str, pd.DataFrame],
    markers_same: pd.DataFrame | None,
    selection_meta: dict,
    expression_source: str,
    out_dir: Path,
    same_domain: str,
) -> str:
    lines = [
        f"# GC deep-dive — cryptic component rank {rank} (n={n})",
        "",
        f"**Dataset:** `{DATASET}` · **expression_source:** `{expression_source}`",
        f"**Selection:** `{selection_meta.get('selection_reason')}`",
        f"**GC Δrest (panel):** {selection_meta.get('gc_delta_rest')}",
        f"**B Δrest / T Δrest:** {selection_meta.get('b_delta_rest')} / {selection_meta.get('t_delta_rest')}",
        "",
        "## Pathology composition inside component",
        "",
        "| Domain | n | fraction |",
        "|--------|--:|---------:|",
    ]
    counts = truth_payload.get("domain_truth_counts", {})
    for lab, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {lab} | {c} | {c / n:.3f} |")

    lines += [
        "",
        "## Spatial adjacency (pathology domains)",
        "",
        f"- Internal kNN edges: **{adj['internal_edges']}** ({adj['internal_edge_fraction']:.1%})",
        f"- External edges: **{adj['external_edges']}**",
        f"- Top abutting domains: **{', '.join(adj['top_abutting_layers']) or '—'}**",
        "",
        "| Domain | Ext contacts | Frac | BG frac | Enrichment | Primary abut spots |",
        "|--------|-------------:|-----:|--------:|-----------:|-------------------:|",
    ]
    for row in adj["layer_table"]:
        if row["external_neighbour_counts"] == 0 and row["background_layer_fraction"] == 0:
            continue
        lines.append(
            f"| {row['layer']} | {row['external_neighbour_counts']} | "
            f"{row['fraction_of_external_contacts']:.3f} | "
            f"{row['background_layer_fraction']:.3f} | "
            f"{row['enrichment_vs_background'] if row['enrichment_vs_background'] is not None else '—'} | "
            f"{row['n_spots_primary_abut']} |"
        )

    lines += ["", "## Pre-registered panel contrasts", ""]
    if panel_df.empty:
        lines.append("_No panel genes present in matrix._")
    else:
        lines.append("| Panel | Δrest | Δsame-domain | mean_in | mean_rest | genes |")
        lines.append("|-------|------:|-------------:|--------:|----------:|-------|")
        for _, r in panel_df.iterrows():
            lines.append(
                f"| {r['panel']} | {r['delta_rest']:.3f} | "
                f"{r['delta_same_domain'] if np.isfinite(r['delta_same_domain']) else float('nan'):.3f} | "
                f"{r['mean_in']:.3f} | {r['mean_rest']:.3f} | `{r['genes_used']}` |"
            )

    lines += [
        "",
        "## Marker DE (component vs rest)",
        "",
        "Wilcoxon rank-sum on library-size log1p; BH-FDR. MT/Ribo/IG/HB banned.",
        "",
    ]
    sig = markers_rest[markers_rest["padj"] <= 0.05]
    up = sig[sig["log2fc_in_vs_out"] > 0].head(15)
    down = sig[sig["log2fc_in_vs_out"] < 0].head(10)
    if up.empty and down.empty:
        lines.append("_No genes at padj ≤ 0.05._")
    else:
        lines.append("### Up in component")
        lines.append("")
        lines.append("| Gene | mean_in | mean_out | log2FC | padj |")
        lines.append("|------|--------:|---------:|-------:|-----:|")
        for _, r in up.iterrows():
            flag = " **GC**" if r["gene"] in GC_EXTENDED else ""
            lines.append(
                f"| `{r['gene']}`{flag} | {r['mean_in']:.3f} | {r['mean_out']:.3f} | "
                f"{r['log2fc_in_vs_out']:.3f} | {r['padj']:.2e} |"
            )
        if not down.empty:
            lines.append("")
            lines.append("### Down in component")
            lines.append("")
            lines.append("| Gene | mean_in | mean_out | log2FC | padj |")
            lines.append("|------|--------:|---------:|-------:|-----:|")
            for _, r in down.iterrows():
                lines.append(
                    f"| `{r['gene']}` | {r['mean_in']:.3f} | {r['mean_out']:.3f} | "
                    f"{r['log2fc_in_vs_out']:.3f} | {r['padj']:.2e} |"
                )

    lines += ["", f"## Markers vs same domain (`{same_domain}`) — hard contrast", ""]
    if markers_same is None or markers_same.empty:
        lines.append("_Insufficient same-domain background (n<20)._")
    else:
        top = markers_same[
            (markers_same["padj"] <= 0.05) & (markers_same["log2fc_in_vs_out"] > 0)
        ].head(10)
        if top.empty:
            lines.append("_No up-genes at padj ≤ 0.05 vs same-domain background._")
        else:
            lines.append("| Gene | log2FC | padj | mean_in | mean_same |")
            lines.append("|------|-------:|-----:|--------:|----------:|")
            for _, r in top.iterrows():
                flag = " **GC**" if r["gene"] in GC_EXTENDED else ""
                lines.append(
                    f"| `{r['gene']}`{flag} | {r['log2fc_in_vs_out']:.3f} | {r['padj']:.2e} | "
                    f"{r['mean_in']:.3f} | {r['mean_out']:.3f} |"
                )

    lines += ["", "## Markers vs abutting domains", ""]
    for domain, frame in markers_domains.items():
        lines.append(f"### vs `{domain}`")
        lines.append("")
        top = frame[(frame["padj"] <= 0.05) & (frame["log2fc_in_vs_out"] > 0)].head(10)
        if top.empty:
            lines.append("_No up-genes at padj ≤ 0.05._")
        else:
            lines.append("| Gene | log2FC | padj | mean_in | mean_domain |")
            lines.append("|------|-------:|-----:|--------:|------------:|")
            for _, r in top.iterrows():
                lines.append(
                    f"| `{r['gene']}` | {r['log2fc_in_vs_out']:.3f} | {r['padj']:.2e} | "
                    f"{r['mean_in']:.3f} | {r['mean_out']:.3f} |"
                )
        lines.append("")

    lines += [
        "## Claim bounds",
        "",
        "1. Single-section geometric cryptic component (methods disagree inside tissue).",
        "2. Marker lists are differential expression, not causal cell-state proof.",
        "3. If `expression_source` is synthetic, panel/DE claims stay at architecture Level 0–1.",
        "4. With official matrix, direction claims can rise to multi-method Level 2 pending replication.",
        "",
        f"Artifacts: `{out_dir.as_posix()}`",
        "",
    ]
    return "\n".join(lines)


def write_synthesis(
    summaries: list[dict], expression_source: str, out_path: Path, auroc: float | None
) -> str:
    lines = [
        "# GC-enriched cryptic components — Xenium lymph node deep-dive",
        "",
        f"**Dataset:** `{DATASET}` · **expression_source:** `{expression_source}`",
        f"**AUROC(U → pathology boundary)** from discovery run: "
        f"{auroc if auroc is not None else 'n/a'}",
        "",
        "## Selected components",
        "",
        "| Rank | n | Dominant pathology | Reason | GC Δrest | B Δrest | T Δrest | "
        "Sig↑ genes | GC genes↑ | Abut | Internal edge frac |",
        "|-----:|--:|--------------------|--------|---------:|--------:|--------:|"
        "-----------:|-----------|------|-------------------:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['rank']} | {s['n']} | {s.get('dominant_truth')} | "
            f"{s.get('selection_reason')} | "
            f"{s.get('gc_delta_rest', float('nan')):.3f} | "
            f"{s.get('b_delta_rest', float('nan')):.3f} | "
            f"{s.get('t_delta_rest', float('nan')):.3f} | "
            f"{s.get('n_sig_up_padj05', 0)} | "
            f"{','.join(s.get('gc_genes_sig_up') or []) or '—'} | "
            f"{','.join(s.get('top_abut') or []) or '—'} | "
            f"{s.get('internal_edge_fraction', float('nan')):.2f} |"
        )

    lines += [
        "",
        "## Cross-tissue对照 (DLPFC largest-component protocol)",
        "",
        "| Aspect | DLPFC Visium | Xenium LN (this run) |",
        "|--------|--------------|----------------------|",
        "| Contiguous cryptic components | yes | yes |",
        "| Adjacency table | WM / L1–L6 | pathology domains (LN / GC / adipose) |",
        "| DE vs rest + abutting | yes | yes |",
        "| Hard same-domain contrast | Layer 3 / Layer 6 | dominant pathology label |",
        "| Pre-registered panels | ENC1/HOPX vs MBP | B / T / GC lymphoid |",
        "| Largest-component parity | rank-0 always reported | rank-0 always included |",
        "",
        "## Interpretation guide",
        "",
        "- **Pathology GC label + GC panel↑**: strongest architectural hit for GC-like niche.",
        "- **LN parenchyma + GC panel↑**: candidate cryptic GC-like subregion inside bulk LN "
        "(methods disagree; polygon GT may be too coarse).",
        "- **High internal edge fraction**: compact niche (blob), not ribbon boundary.",
        "- **same-domain hard DE empty**: mirrors DLPFC L3 pattern — direction vs rest works, "
        "intra-domain hard gate often fails.",
        "",
        f"Per-component reports live under `{(OUT_ROOT / 'gc_deep_dive').as_posix()}/`.",
        "",
    ]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    (BASE / "GC_DEEP_DIVE_REPORT.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    _setup()
    map_path = OUT_ROOT / "spot_uncertainty_map.csv"
    if not map_path.is_file():
        logger.error("%s missing — run run_discovery_ln.py first", map_path)
        return 2

    spots = pd.read_csv(map_path)
    coords = spots[["x", "y"]].to_numpy(dtype=float)
    cryptic = spots["cryptic_niche"].to_numpy(dtype=bool)
    truth = spots["domain_truth"].astype(str).to_numpy()
    domain_order = sorted(pd.unique(truth).tolist())

    comps = cryptic_components(coords, cryptic, k=K_NN, min_size=MIN_COMPONENT)
    if not comps:
        logger.error("no cryptic components ≥ %s", MIN_COMPONENT)
        return 1
    logger.info("found %s components ≥%s: %s", len(comps), MIN_COMPONENT, [len(c) for c in comps])

    entry = get_dataset(DATASET)
    data = entry.load(cache_dir=ROOT / "datasets_cache")
    if "domain_truth" in data.obs.columns:
        keep = data.obs["domain_truth"].notna().to_numpy()
        lab = data.obs["domain_truth"].astype(str).to_numpy()
        keep = keep & ~np.isin(lab, ["NA", "nan", "None", ""])
        data = data.subset_obs(keep) if hasattr(data, "subset_obs") else data[keep]
    expression_source = str(data.uns.get("expression_source", "unknown"))

    if data.n_obs != len(spots):
        logger.warning("n_obs mismatch data=%s map=%s — coord align", data.n_obs, len(spots))
        dxy = np.asarray(data.spatial, dtype=float)
        key_data = {(round(float(x), 1), round(float(y), 1)): i for i, (x, y) in enumerate(dxy)}
        remap = np.asarray(
            [key_data.get((round(float(x), 1), round(float(y), 1)), -1) for x, y in coords],
            dtype=int,
        )
        if (remap < 0).any():
            raise RuntimeError("could not align expression to uncertainty map")
        # reorder data via index of unique map order
        order = remap
        X_full = log1p_norm(_to_dense(data.X))[order]
        # truth from map already
    else:
        X_full = log1p_norm(_to_dense(data.X))
    genes = np.asarray(list(map(str, data.var_names)))

    selected = select_components(comps, truth, X_full, list(genes), top_n=3)
    dive_dir = OUT_ROOT / "gc_deep_dive"
    dive_dir.mkdir(parents=True, exist_ok=True)
    (dive_dir / "selection.json").write_text(
        json.dumps(
            {
                "expression_source": expression_source,
                "n_components_total": len(comps),
                "component_sizes": [len(c) for c in comps],
                "selected": selected,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    for meta in selected:
        rank = int(meta["rank"])
        component = comps[rank]
        cdir = dive_dir / f"component_rank{rank}_n{len(component)}"
        logger.info(
            "deep-dive rank=%s n=%s reason=%s GCΔ=%.3f",
            rank,
            len(component),
            meta.get("selection_reason"),
            float(meta.get("gc_delta_rest") or float("nan")),
        )
        summary = deep_dive_one(
            rank=rank,
            component=component,
            spots=spots,
            coords=coords,
            truth=truth,
            X=X_full,
            genes=genes,
            domain_order=domain_order,
            out_dir=cdir,
            selection_meta=meta,
            expression_source=expression_source,
        )
        # attach panel deltas from selection
        summary["b_delta_rest"] = meta.get("b_delta_rest")
        summary["t_delta_rest"] = meta.get("t_delta_rest")
        summaries.append(summary)

    auroc = None
    summary_path = OUT_ROOT / "slice_summary.json"
    if summary_path.is_file():
        auroc = json.loads(summary_path.read_text(encoding="utf-8")).get("auroc_known_boundary")

    write_synthesis(summaries, expression_source, dive_dir / "GC_DEEP_DIVE_REPORT.md", auroc)
    logger.info("Wrote %s", dive_dir / "GC_DEEP_DIVE_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
