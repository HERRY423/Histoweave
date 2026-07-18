"""Deep-dive the largest cryptic component on a DLPFC slice.

Outputs (under ``results/<slice>/largest_component/``):

* ``component_spots.csv`` — spots in the largest cryptic connected component
* ``adjacency_to_layers.csv`` — neighbour-layer contact rates (WM / L1–L6)
* ``adjacency_summary.json`` — primary abutting layers + enrichment
* ``markers_vs_rest.csv`` — Wilcoxon-style rank-sum DE (component vs all other)
* ``markers_vs_abutting_layers.csv`` — DE vs each major abutting layer
* ``COMPONENT_REPORT.md`` — human-readable summary

Default target: ``dlpfc_151508`` (largest component ~154 spots in prior run).
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave._math import knn_indices  # noqa: E402
from histoweave.benchmark.multiple_testing import fdr_adjust  # noqa: E402
from histoweave.datasets import get_dataset  # noqa: E402

logger = logging.getLogger("largest_component")

SLICE_ID = "dlpfc_151508"
K_NN = 6
MIN_COMPONENT = 15
N_TOP_MARKERS = 40
LAYER_ORDER = ["Layer 1", "Layer 2", "Layer 3", "Layer 4", "Layer 5", "Layer 6", "WM"]
BAN_PREFIXES = ("MT-", "mt-", "RPL", "RPS", "IGK", "IGL", "IGH", "HB")


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _load_spot_map(slice_id: str) -> pd.DataFrame:
    path = Path(__file__).resolve().parent / "results" / slice_id / "spot_uncertainty_map.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing — run run_discovery.py first for {slice_id}")
    return pd.read_csv(path)


def cryptic_components(
    coords: np.ndarray,
    cryptic: np.ndarray,
    *,
    k: int = K_NN,
    min_size: int = MIN_COMPONENT,
) -> list[np.ndarray]:
    """Return list of index arrays, largest component first."""
    cryptic = np.asarray(cryptic, dtype=bool)
    n = len(cryptic)
    nbrs = knn_indices(coords, k + 1)
    seen = np.zeros(n, dtype=bool)
    comps: list[np.ndarray] = []
    for i in range(n):
        if not cryptic[i] or seen[i]:
            continue
        stack = [i]
        seen[i] = True
        members: list[int] = []
        while stack:
            u = stack.pop()
            members.append(u)
            for v in nbrs[u]:
                v = int(v)
                if v == u or seen[v] or not cryptic[v]:
                    continue
                seen[v] = True
                stack.append(v)
        if len(members) >= min_size:
            comps.append(np.asarray(members, dtype=int))
    comps.sort(key=lambda a: -len(a))
    return comps


def rank_sum_z(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann–Whitney U via normal approximation → z and p.

    Uses mid-ranks; suitable for large n without scipy dependency.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return 0.0, 1.0
    combined = np.concatenate([x, y])
    # Average ranks for ties
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
    # Tie correction
    _, counts = np.unique(combined, return_counts=True)
    tie = (counts**3 - counts).sum() / (len(combined) * (len(combined) - 1))
    sigma2 = n1 * n2 / 12.0 * ((n1 + n2 + 1) - tie)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    # Continuity correction
    z = (u1 - mu - 0.5 * np.sign(u1 - mu)) / sigma
    # two-sided normal p via erfc
    from math import erfc, sqrt

    p = float(erfc(abs(z) / sqrt(2.0)))
    return float(z), p


def differential_markers(
    X: np.ndarray,
    gene_names: np.ndarray,
    in_mask: np.ndarray,
    out_mask: np.ndarray,
    *,
    label_in: str,
    label_out: str,
) -> pd.DataFrame:
    rows = []
    for j, gene in enumerate(gene_names):
        g = str(gene)
        if any(g.startswith(p) for p in BAN_PREFIXES):
            continue
        a = X[in_mask, j]
        b = X[out_mask, j]
        mean_in = float(a.mean())
        mean_out = float(b.mean())
        # log2 fold on log1p-scale means (pseudo)
        lfc = float(np.log2((mean_in + 1e-6) / (mean_out + 1e-6)))
        z, p = rank_sum_z(a, b)
        rows.append(
            {
                "gene": g,
                "mean_in": mean_in,
                "mean_out": mean_out,
                "log2fc_in_vs_out": lfc,
                "z": z,
                "p_raw": p,
                "group_in": label_in,
                "group_out": label_out,
                "n_in": int(in_mask.sum()),
                "n_out": int(out_mask.sum()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["padj"] = fdr_adjust(frame["p_raw"].to_numpy(), method="bh")
    # Prefer higher expression inside component among significant hits
    frame = frame.sort_values(["padj", "log2fc_in_vs_out"], ascending=[True, False])
    return frame


def adjacency_profile(
    coords: np.ndarray,
    labels: np.ndarray,
    component: np.ndarray,
    *,
    k: int = K_NN,
) -> dict:
    """For each component spot, count kNN neighbour layer labels (outside self)."""
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
    # Background: layer prevalence among non-component spots
    non = np.array([i for i in range(len(labels)) if i not in component_set])
    bg = Counter(str(labels[i]) for i in non)
    bg_total = sum(bg.values()) or 1

    rows = []
    for layer in LAYER_ORDER:
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


def log1p_norm(X: np.ndarray) -> np.ndarray:
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * 1e4)


def write_report(
    slice_id: str,
    comp_size: int,
    adj: dict,
    markers_rest: pd.DataFrame,
    markers_layers: dict[str, pd.DataFrame],
    out_dir: Path,
    truth_comp: dict | None = None,
    *,
    component_rank: int = 0,
) -> str:
    top_abut = adj["top_abutting_layers"]
    rank_label = "largest" if component_rank == 0 else f"rank-{component_rank + 1}"
    lines = [
        f"# Cryptic component ({rank_label}) — `{slice_id}`",
        "",
        f"**Size:** {comp_size} spots · **rank:** {component_rank} "
        f"(0 = largest; spatially contiguous cryptic niche).",
        "",
        "## Manual-layer composition (domain_truth inside component)",
        "",
    ]
    if truth_comp:
        counts = truth_comp.get("domain_truth_counts", {})
        lines.append("| Layer | n spots | fraction |")
        lines.append("|-------|--------:|---------:|")
        for layer, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {layer} | {n} | {n / comp_size:.3f} |")
        dominant = max(counts, key=counts.get) if counts else None
        if dominant and counts[dominant] / comp_size >= 0.9:
            lines.append("")
            lines.append(
                f"**Dominant truth label:** `{dominant}` "
                f"({counts[dominant] / comp_size:.0%} of component). "
                "If external contacts are also almost exclusively this layer, the niche is "
                "**intra-layer substructure** (methods disagree *inside* a manual domain), "
                "not an inter-layer boundary ribbon."
            )
    else:
        lines.append("_Truth composition not available._")

    lines += [
        "",
        "## Adjacency to WM / L1–L6",
        "",
        f"- Internal kNN edges (within component): **{adj['internal_edges']}** "
        f"({adj['internal_edge_fraction']:.1%} of all component-incident edges)",
        f"- External edges (to outside): **{adj['external_edges']}**",
        f"- Top abutting layers by contact count: **{', '.join(top_abut) or '—'}**",
        f"- Primary abutment distribution: `{adj.get('primary_abutment_distribution', {})}`",
        "",
        "| Layer | External contacts | Fraction | Background frac | Enrichment | Spots with this as primary abut |",
        "|-------|------------------:|---------:|----------------:|-----------:|--------------------------------:|",
    ]
    for row in adj["layer_table"]:
        lines.append(
            f"| {row['layer']} | {row['external_neighbour_counts']} | "
            f"{row['fraction_of_external_contacts']:.3f} | "
            f"{row['background_layer_fraction']:.3f} | "
            f"{row['enrichment_vs_background'] if row['enrichment_vs_background'] is not None else '—'} | "
            f"{row['n_spots_primary_abut']} |"
        )

    lines += [
        "",
        "### Geometric interpretation",
        "",
    ]
    pure_layer = None
    if truth_comp:
        counts = truth_comp.get("domain_truth_counts", {})
        if counts:
            pure_layer = max(counts, key=counts.get)
            if counts[pure_layer] < 0.9 * comp_size:
                pure_layer = None
    if pure_layer and top_abut == [pure_layer]:
        lines.append(
            f"- **Intra-`{pure_layer}` compact niche:** every component spot and every "
            f"external neighbour contact is `{pure_layer}`. Multi-method uncertainty is "
            f"flagging a **subregion inside {pure_layer}**, not a WM↔cortex or L5↔L6 border."
        )
    elif top_abut:
        if "WM" in top_abut[:2]:
            lines.append(
                "- Component heavily abuts **WM** → candidate white-matter / deep-layer "
                "transition niche (or WM-adjacent deep grey)."
            )
        if any(L in top_abut[:2] for L in ("Layer 1", "Layer 2")):
            lines.append(
                "- Strong contact with **superficial layers** → candidate pia / L1–L2 "
                "boundary or superficial domain disagreement zone."
            )
        if any(L in top_abut[:2] for L in ("Layer 5", "Layer 6")):
            lines.append(
                "- Strong contact with **L5/L6** → candidate deep laminar transition "
                "or deep-layer interior disagreement."
            )
        mid = [L for L in top_abut if L in ("Layer 3", "Layer 4")]
        if mid:
            lines.append(
                f"- Contacts with **{', '.join(mid)}** → mid-cortical disagreement; "
                "may mark laminar borders L3/L4 that multi-method ensembles split inconsistently."
            )
    lines.append(
        f"- Internal edge fraction {adj['internal_edge_fraction']:.1%} indicates "
        + (
            "a compact blob (more self-contained niche)."
            if adj["internal_edge_fraction"] >= 0.35
            else "a more ribbon-like / boundary-associated structure."
        )
    )

    lines += [
        "",
        "## Marker genes (component vs all other spots)",
        "",
        "Wilcoxon rank-sum on library-size log1p expression; BH-FDR. "
        "MT/Ribo/IG/HB genes excluded from ranking.",
        "",
    ]
    sig = markers_rest[markers_rest["padj"] <= 0.05].head(N_TOP_MARKERS)
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
            lines.append(
                f"| `{r['gene']}` | {r['mean_in']:.3f} | {r['mean_out']:.3f} | "
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

    lines += ["", "## Markers vs major abutting layers", ""]
    for layer, frame in markers_layers.items():
        lines.append(f"### vs `{layer}`")
        lines.append("")
        top = frame[(frame["padj"] <= 0.05) & (frame["log2fc_in_vs_out"] > 0)].head(10)
        if top.empty:
            lines.append("_No up-genes at padj ≤ 0.05._")
        else:
            lines.append("| Gene | log2FC | padj | mean_in | mean_layer |")
            lines.append("|------|-------:|-----:|--------:|-----------:|")
            for _, r in top.iterrows():
                lines.append(
                    f"| `{r['gene']}` | {r['log2fc_in_vs_out']:.3f} | {r['padj']:.2e} | "
                    f"{r['mean_in']:.3f} | {r['mean_out']:.3f} |"
                )
        lines.append("")

    lines += [
        "## Claim bounds",
        "",
        "1. This is a **single-slice** deep-dive of a geometric cryptic component.",
        "2. Marker lists are differential expression, not causal cell-state proof.",
        "3. Upgrade requires protein/IF validation and multi-slice replication of the "
        "same abutting-layer pattern + marker panel.",
        "",
        f"Artifacts: `{out_dir.as_posix()}`",
        "",
    ]
    return "\n".join(lines)


def main(slice_id: str = SLICE_ID, component_rank: int = 0) -> int:
    _setup()
    spots = _load_spot_map(slice_id)
    coords = spots[["x", "y"]].to_numpy(dtype=float)
    cryptic = spots["cryptic_niche"].to_numpy(dtype=bool)
    labels = spots["domain_truth"].astype(str).to_numpy()

    comps = cryptic_components(coords, cryptic)
    if not comps:
        logger.error("No cryptic components ≥ %s spots", MIN_COMPONENT)
        return 1
    if component_rank < 0 or component_rank >= len(comps):
        logger.error(
            "component_rank=%s out of range (0..%s); sizes=%s",
            component_rank,
            len(comps) - 1,
            [len(c) for c in comps],
        )
        return 1
    component = comps[component_rank]
    tag = (
        "largest_component"
        if component_rank == 0
        else f"component_rank{component_rank}_n{len(component)}"
    )
    out_dir = Path(__file__).resolve().parent / "results" / slice_id / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Analysing %s rank=%s size=%s (of %s components ≥%s)",
        slice_id,
        component_rank,
        len(component),
        len(comps),
        MIN_COMPONENT,
    )
    for i, c in enumerate(comps[:8]):
        logger.info("  component[%s] size=%s%s", i, len(c), " ←" if i == component_rank else "")

    # Spot table
    comp_df = spots.iloc[component].copy()
    comp_df["component_rank"] = component_rank
    comp_df["in_selected_component"] = 1
    comp_df.to_csv(out_dir / "component_spots.csv", index=False)

    # Layer composition inside component (where methods disagree but truth has a layer)
    truth_counts = Counter(labels[component])
    (out_dir / "component_truth_composition.json").write_text(
        json.dumps(
            {
                "n": len(component),
                "domain_truth_counts": dict(truth_counts),
                "domain_truth_fractions": {k: v / len(component) for k, v in truth_counts.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    adj = adjacency_profile(coords, labels, component, k=K_NN)
    pd.DataFrame(adj["layer_table"]).to_csv(out_dir / "adjacency_to_layers.csv", index=False)
    adj_json = {k: v for k, v in adj.items() if k != "layer_table"}
    adj_json["layer_table"] = adj["layer_table"]
    (out_dir / "adjacency_summary.json").write_text(
        json.dumps(adj_json, indent=2), encoding="utf-8"
    )
    logger.info("Top abutting layers: %s", adj["top_abutting_layers"])

    # Expression for DE
    entry = get_dataset(slice_id)
    data = entry.load(cache_dir=ROOT / "datasets_cache")
    # Align: discovery map was built on filtered annotated spots in same order as load+filter.
    # Re-apply the same truth filter as run_discovery.
    if "domain_truth" in data.obs.columns:
        keep = data.obs["domain_truth"].notna().to_numpy()
        lab = data.obs["domain_truth"].astype(str).to_numpy()
        keep = keep & ~np.isin(lab, ["NA", "nan", "None", ""])
        data = data.subset_obs(keep)
    if data.n_obs != len(spots):
        logger.warning(
            "n_obs mismatch data=%s map=%s — aligning by spatial coordinates",
            data.n_obs,
            len(spots),
        )
        # Fallback: match by rounded coords
        dxy = np.asarray(data.spatial, dtype=float)
        key_data = {(round(x, 1), round(y, 1)): i for i, (x, y) in enumerate(dxy)}
        remap = []
        for x, y in coords:
            remap.append(key_data.get((round(x, 1), round(y, 1)), -1))
        remap_arr = np.asarray(remap, dtype=int)
        if (remap_arr < 0).any():
            raise RuntimeError("Could not align expression table to uncertainty map")
        # Build component mask on data order
        in_mask = np.zeros(data.n_obs, dtype=bool)
        in_mask[remap_arr[component]] = True
        labels_data = data.obs["domain_truth"].astype(str).to_numpy()
    else:
        in_mask = np.zeros(data.n_obs, dtype=bool)
        in_mask[component] = True
        labels_data = labels
        remap_arr = np.arange(data.n_obs)

    X = log1p_norm(_to_dense(data.X))
    genes = np.asarray(data.var_names.astype(str))
    # Restrict to top variable genes for speed/signal
    var = X.var(axis=0)
    ban = np.array([any(g.startswith(p) for p in BAN_PREFIXES) for g in genes])
    var = np.where(ban, -1.0, var)
    keep_g = np.argsort(var)[::-1][:2000]
    keep_g = np.sort(keep_g[var[keep_g] > 0])
    X = X[:, keep_g]
    genes = genes[keep_g]

    out_mask = ~in_mask
    markers_rest = differential_markers(
        X, genes, in_mask, out_mask, label_in="largest_cryptic", label_out="rest"
    )
    markers_rest.to_csv(out_dir / "markers_vs_rest.csv", index=False)
    logger.info(
        "Markers vs rest: %s at padj<=0.05",
        int((markers_rest["padj"] <= 0.05).sum()),
    )

    # DE vs each top abutting layer (layer spots outside component)
    markers_layers: dict[str, pd.DataFrame] = {}
    for layer in adj["top_abutting_layers"][:3]:
        layer_mask = (labels_data == layer) & (~in_mask)
        if layer_mask.sum() < 20:
            logger.warning("Skip DE vs %s (n=%s)", layer, layer_mask.sum())
            continue
        frame = differential_markers(
            X,
            genes,
            in_mask,
            layer_mask,
            label_in="largest_cryptic",
            label_out=layer,
        )
        frame.to_csv(out_dir / f"markers_vs_{layer.replace(' ', '_')}.csv", index=False)
        markers_layers[layer] = frame
        logger.info(
            "Markers vs %s: %s up padj<=0.05",
            layer,
            int(((frame["padj"] <= 0.05) & (frame["log2fc_in_vs_out"] > 0)).sum()),
        )

    # Combined abutting-layer DE table
    if markers_layers:
        pd.concat(markers_layers.values(), ignore_index=True).to_csv(
            out_dir / "markers_vs_abutting_layers.csv", index=False
        )

    truth_payload = json.loads(
        (out_dir / "component_truth_composition.json").read_text(encoding="utf-8")
    )
    report = write_report(
        slice_id,
        len(component),
        adj,
        markers_rest,
        markers_layers,
        out_dir,
        truth_comp=truth_payload,
        component_rank=component_rank,
    )
    (out_dir / "COMPONENT_REPORT.md").write_text(report, encoding="utf-8")
    # Promote a copy for quick browsing
    parent_report = (
        Path(__file__).resolve().parent
        / f"COMPONENT_REPORT_{slice_id}_rank{component_rank}_n{len(component)}.md"
    )
    parent_report.write_text(report, encoding="utf-8")
    if component_rank == 0:
        # Keep legacy alias for the largest component.
        legacy = Path(__file__).resolve().parent / f"COMPONENT_REPORT_{slice_id}.md"
        legacy.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", out_dir / "COMPONENT_REPORT.md")
    return 0


if __name__ == "__main__":
    # Usage:
    #   python analyze_largest_component.py [slice_id] [component_rank]
    # Examples:
    #   python analyze_largest_component.py dlpfc_151508 0   # largest (154)
    #   python analyze_largest_component.py dlpfc_151508 1   # second (138)
    #   python analyze_largest_component.py dlpfc_151669 0   # 151669 largest
    sid = sys.argv[1] if len(sys.argv) > 1 else SLICE_ID
    rank = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    raise SystemExit(main(sid, rank))
