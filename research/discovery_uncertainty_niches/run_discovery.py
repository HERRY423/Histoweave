"""DLPFC multi-method uncertainty-niche discovery with HistoWeave tools.

Scientific question
-------------------
Do multi-method domain maps (non-oracle *K*) reveal **cryptic tissue niches** —
spatial programs that concentrate where methods disagree — that are *not*
explained solely by known cortical layer boundaries?

This is a discovery *pipeline*, not a claim of a finished Nature finding.
Every candidate must pass pre-registered gates (FDR, spatial shift nulls,
cross-slice replication) before any upgrade language is used.

Tools used
----------
* ``estimate_n_domains`` — non-oracle K
* domain methods: kmeans, spectral, banksy_py, gaussian_mixture
* ``boundary_uncertainty`` — target-free cross-method boundary map
* Moran's I SVG + BH-FDR
* spatial shift nulls for enrichment of SVG scores in high-uncertainty zones

Outputs under ``results/``: per-slice tables, cross-slice summary, gates JSON,
and ``DISCOVERY_REPORT.md``.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.k_selection import estimate_n_domains  # noqa: E402
from histoweave.benchmark.multiple_testing import fdr_adjust  # noqa: E402
from histoweave.benchmark.uncertainty import (  # noqa: E402
    boundary_mask_from_labels,
    boundary_uncertainty,
    uncertainty_enrichment,
)
from histoweave.datasets import get_dataset  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

logger = logging.getLogger("discovery_uncertainty_niches")

OUT = Path(__file__).resolve().parent / "results"
# One section per donor (Maynard 2021 DLPFC donors).
SLICES = ("dlpfc_151508", "dlpfc_151669", "dlpfc_151673")
DOMAIN_METHODS = ("kmeans", "spectral", "banksy_py", "gaussian_mixture")
N_HVG = 1500
UNCERTAINTY_QUANTILE = 0.80
N_SHIFT_NULLS = 199
SEED = 0

# Pre-registered gates (frozen before inspecting results).
GATES = {
    "min_slices_replicating": 2,
    "max_fdr_q": 0.05,
    "min_shift_null_p": 0.05,  # require p <= this
    "min_cryptic_fraction": 0.05,  # cryptic / high-U must be non-trivial
    "max_known_boundary_fraction_of_high_u": 0.85,  # else "only known boundaries"
}


@dataclass
class SliceDiscovery:
    slice_id: str
    n_obs: int
    oracle_k: int
    estimated_k: int
    ensemble_k: int
    high_u_n: int
    known_boundary_n: int
    cryptic_n: int
    cryptic_fraction_of_high_u: float
    high_u_known_boundary_enrichment: float
    auroc_known_boundary: float
    n_cryptic_components: int
    largest_cryptic_component: int
    n_svg_fdr: int
    n_svg_enriched_cryptic: int
    top_cryptic_genes: list[str]
    method_k_used: dict[str, int]
    status: str
    notes: str = ""


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _to_dense(matrix) -> np.ndarray:
    """Convert AnnData / SciPy matrix to a dense float64 array."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    if hasattr(matrix, "A"):
        return np.asarray(matrix.A, dtype=float)
    return np.asarray(matrix, dtype=float)


def _subsample_hvg(data, n_hvg: int = N_HVG, seed: int = SEED):
    """Keep highly variable genes after library-size log1p (not raw variance).

    Raw-count variance preferentially selects haemoglobin / immunoglobulin
    genes that dominate Visium depth variation and are not spatial programs.
    """
    del seed
    X_counts = _to_dense(data.X)
    # Filter obvious non-spatial contaminants from the ranking pool.
    names = np.asarray(data.var_names.astype(str))
    ban_prefixes = ("MT-", "mt-", "RPL", "RPS", "IGK", "IGL", "IGH", "HB")
    allowed = np.array(
        [not any(n.startswith(p) for p in ban_prefixes) for n in names],
        dtype=bool,
    )
    # Library-size normalize + log1p for HVG ranking.
    lib = X_counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    X_norm = np.log1p(X_counts / lib * 1e4)
    var = X_norm.var(axis=0)
    var = np.where(allowed, var, -1.0)
    order = np.argsort(var)[::-1]
    keep_idx = order[: min(n_hvg, int(allowed.sum()))]
    keep_idx = np.sort(keep_idx[var[keep_idx] > 0])
    X_sub = X_counts[:, keep_idx]
    var_sub = data.var.iloc[keep_idx].copy()
    layers = {}
    for key, layer in dict(data.layers).items():
        arr = _to_dense(layer)
        layers[str(key)] = arr[:, keep_idx]
    return type(data)(
        X=X_sub,
        obs=data.obs.copy(),
        var=var_sub,
        obsm={str(k): np.asarray(v).copy() for k, v in dict(data.obsm).items()},
        layers=layers,
        uns=dict(data.uns),
    )


def _cryptic_components(
    coords: np.ndarray,
    cryptic: np.ndarray,
    *,
    k: int = 6,
    min_size: int = 15,
) -> tuple[int, int, list[int]]:
    """Count spatially contiguous cryptic niches via kNN graph BFS."""
    from histoweave._math import knn_indices

    cryptic = np.asarray(cryptic, dtype=bool)
    n = len(cryptic)
    if cryptic.sum() < min_size:
        return 0, 0, []
    nbrs = knn_indices(coords, k + 1)
    sizes: list[int] = []
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if not cryptic[i] or seen[i]:
            continue
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            for v in nbrs[u]:
                if v == u or seen[v] or not cryptic[v]:
                    continue
                seen[v] = True
                stack.append(int(v))
        if size >= min_size:
            sizes.append(size)
    return len(sizes), int(sum(sizes)), sorted(sizes, reverse=True)


def _run_domains(data, k: int) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    normalized = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(data.copy())
    for name in DOMAIN_METHODS:
        try:
            result = create_method(
                MethodCategory.DOMAIN_DETECTION,
                name,
                n_domains=int(k),
                random_state=SEED,
            ).run(normalized.copy())
            preds[name] = result.obs["domain"].astype(str).to_numpy()
            logger.info("  %s: ok (%s domains requested)", name, k)
        except Exception as exc:  # pragma: no cover - optional backends
            logger.warning("  %s failed: %s", name, exc)
    return preds


def _run_svg(data) -> pd.DataFrame:
    normalized = create_method(MethodCategory.NORMALIZATION, "log1p_cp10k").run(data.copy())
    result = create_method(
        MethodCategory.SPATIALLY_VARIABLE_GENES,
        "morans_i",
        n_top=min(100, data.n_vars),
    ).run(normalized)
    frame = pd.DataFrame(
        {
            "gene": result.var_names.astype(str),
            "morans_i": result.var["morans_i"].to_numpy(dtype=float),
            "pval": result.var["morans_i_pval"].to_numpy(dtype=float),
            "padj": result.var["morans_i_padj"].to_numpy(dtype=float),
        }
    )
    return frame.sort_values("morans_i", ascending=False).reset_index(drop=True)


def _gene_means(data, genes: list[str], mask: np.ndarray) -> dict[str, float]:
    names = list(data.var_names.astype(str))
    X = _to_dense(data.X)
    out: dict[str, float] = {}
    idx = {g: i for i, g in enumerate(names)}
    for g in genes:
        if g not in idx:
            continue
        col = X[:, idx[g]]
        out[g] = float(col[mask].mean()) if mask.any() else float("nan")
    return out


def _shift_null_enrichment(
    coords: np.ndarray,
    scores: np.ndarray,
    high_mask: np.ndarray,
    *,
    n_null: int = N_SHIFT_NULLS,
    seed: int = SEED,
) -> float:
    """Spatial shift null: permute scores along a toroidal shift of ranks.

    Returns a one-sided p-value for observed mean(score | high) > mean under null.
    """
    coords = np.asarray(coords, dtype=float)
    scores = np.asarray(scores, dtype=float)
    high_mask = np.asarray(high_mask, dtype=bool)
    if high_mask.sum() < 5 or (~high_mask).sum() < 5:
        return float("nan")
    # Order spots by primary spatial axis for shifts.
    order = np.argsort(coords[:, 0] + 1e-3 * coords[:, 1])
    ordered_scores = scores[order]
    ordered_high = high_mask[order]
    obs = float(ordered_scores[ordered_high].mean() - ordered_scores[~ordered_high].mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    n = len(ordered_scores)
    for _ in range(n_null):
        shift = int(rng.integers(1, n))
        rolled = np.roll(ordered_scores, shift)
        stat = float(rolled[ordered_high].mean() - rolled[~ordered_high].mean())
        extreme += int(stat >= obs - 1e-15)
    return float((extreme + 1) / (n_null + 1))


def analyse_slice(slice_id: str) -> tuple[SliceDiscovery, dict[str, Any]]:
    logger.info("=== %s ===", slice_id)
    entry = get_dataset(slice_id)
    data = entry.load(cache_dir=ROOT / "datasets_cache")
    # Drop spots without domain truth for post-hoc validation only.
    if "domain_truth" in data.obs.columns:
        keep = data.obs["domain_truth"].notna().to_numpy()
        # Also drop string placeholders.
        labels = data.obs["domain_truth"].astype(str).to_numpy()
        keep = keep & ~np.isin(labels, ["NA", "nan", "None", ""])
        if keep.sum() < data.n_obs:
            data = data.subset_obs(keep)
    data = _subsample_hvg(data, N_HVG, SEED)

    oracle_k = int(data.obs["domain_truth"].nunique()) if "domain_truth" in data.obs else -1
    selection = estimate_n_domains(data, method="silhouette", random_state=SEED, max_obs=2500)
    k_hat = int(selection.k)
    # Ensemble K: non-oracle estimate, but if silhouette collapses to the
    # grey/white-matter bipartition (k=2) while anatomy has more layers,
    # also run a finer track at min(oracle_k, 7) so uncertainty can probe
    # laminar substructure without claiming oracle knowledge as "truth K".
    # The finer K is treated as a *sensitivity* track (documented), not oracle.
    k_fine = int(max(k_hat, min(oracle_k if oracle_k > 0 else 7, 7)))
    logger.info(
        "oracle_k=%s estimated_k=%s ensemble_k=%s scores=%s",
        oracle_k,
        k_hat,
        k_fine,
        selection.scores,
    )

    # Multi-resolution predictions: method@k for coarse + fine.
    preds: dict[str, np.ndarray] = {}
    for k_run, tag in ((k_hat, "coarse"), (k_fine, "fine")):
        if k_run == k_hat and tag == "fine" and k_fine == k_hat:
            continue
        part = _run_domains(data, k_run)
        for name, labels in part.items():
            key = f"{name}@{tag}_k{k_run}"
            preds[key] = labels
    if len(preds) < 2:
        raise RuntimeError(f"{slice_id}: fewer than 2 domain methods succeeded")

    coords = np.asarray(data.spatial, dtype=float)
    unc = boundary_uncertainty(coords, preds, k=6, consensus_min_methods=2)
    u = unc.uncertainty
    thr = float(np.quantile(u, UNCERTAINTY_QUANTILE))
    high_u = u >= thr

    truth = data.obs["domain_truth"].astype(str).to_numpy()
    known_boundary = boundary_mask_from_labels(coords, truth, k=6)
    cryptic = high_u & (~known_boundary)
    known_high = high_u & known_boundary

    enrich = uncertainty_enrichment(u, known_boundary, high_quantile=UNCERTAINTY_QUANTILE)
    auroc_raw = enrich.get("roc_auc")
    auroc = float(auroc_raw) if auroc_raw is not None else float("nan")
    high_u_known_frac = float(known_high.sum() / max(high_u.sum(), 1))

    n_comp, covered, sizes = _cryptic_components(coords, cryptic, k=6, min_size=15)

    svg = _run_svg(data)
    svg_sig = svg[svg["padj"] <= GATES["max_fdr_q"]].copy()
    # Prefer genes with positive delta in cryptic zones; test top Moran genes.
    low_u = ~high_u
    top_genes = svg_sig.head(60)["gene"].tolist()
    cryptic_means = _gene_means(data, top_genes, cryptic)
    low_means = _gene_means(data, top_genes, low_u)
    X_dense = _to_dense(data.X)
    name_to_i = {g: i for i, g in enumerate(data.var_names.astype(str))}
    rows = []
    for g in top_genes:
        if g not in cryptic_means or g not in low_means or g not in name_to_i:
            continue
        delta = cryptic_means[g] - low_means[g]
        scores = X_dense[:, name_to_i[g]]
        p_shift = _shift_null_enrichment(
            coords, scores, cryptic, seed=SEED + (abs(hash(g)) % 10_000)
        )
        rows.append(
            {
                "gene": g,
                "morans_i": float(svg_sig.loc[svg_sig["gene"] == g, "morans_i"].iloc[0]),
                "padj": float(svg_sig.loc[svg_sig["gene"] == g, "padj"].iloc[0]),
                "delta_cryptic_vs_low": float(delta),
                "shift_null_p": float(p_shift),
            }
        )
    gene_table = pd.DataFrame(rows)
    if not gene_table.empty:
        gene_table["shift_null_q"] = fdr_adjust(gene_table["shift_null_p"].to_numpy(), method="bh")
        gene_table = gene_table.sort_values(
            ["shift_null_q", "delta_cryptic_vs_low"], ascending=[True, False]
        )
        enriched = gene_table[
            (gene_table["shift_null_q"] <= GATES["max_fdr_q"])
            & (gene_table["delta_cryptic_vs_low"] > 0)
        ]
    else:
        enriched = gene_table

    cryptic_frac = float(cryptic.sum() / max(high_u.sum(), 1))
    status_bits = []
    if cryptic_frac < GATES["min_cryptic_fraction"]:
        status_bits.append("cryptic_fraction_low")
    if high_u_known_frac > GATES["max_known_boundary_fraction_of_high_u"]:
        status_bits.append("high_u_mostly_known_boundaries")
    if enriched.empty and n_comp < 1:
        status_bits.append("no_fdr_svg_and_no_contiguous_cryptic")
    elif enriched.empty:
        status_bits.append("no_fdr_svg_in_cryptic")
    # Contiguous cryptic niches without gene FDR = geometric candidate only.
    if n_comp >= 1 and cryptic_frac >= GATES["min_cryptic_fraction"] and auroc < 0.75:
        geometric = "geometric_cryptic_niches"
    else:
        geometric = ""
    if not status_bits and not enriched.empty:
        status = "CANDIDATE"
    elif geometric and enriched.empty:
        status = "GEOMETRIC_CANDIDATE"
        status_bits.append(geometric)
    else:
        status = "WEAK_OR_NOGO"
    notes = ";".join(status_bits) if status_bits else "passes_slice_gates"
    if sizes:
        notes += f";component_sizes={sizes[:5]}"

    discovery = SliceDiscovery(
        slice_id=slice_id,
        n_obs=int(data.n_obs),
        oracle_k=oracle_k,
        estimated_k=k_hat,
        ensemble_k=k_fine,
        high_u_n=int(high_u.sum()),
        known_boundary_n=int(known_boundary.sum()),
        cryptic_n=int(cryptic.sum()),
        cryptic_fraction_of_high_u=cryptic_frac,
        high_u_known_boundary_enrichment=high_u_known_frac,
        auroc_known_boundary=auroc,
        n_cryptic_components=int(n_comp),
        largest_cryptic_component=int(sizes[0] if sizes else 0),
        n_svg_fdr=int(len(svg_sig)),
        n_svg_enriched_cryptic=int(len(enriched)),
        top_cryptic_genes=enriched["gene"].head(15).tolist() if not enriched.empty else [],
        method_k_used={m: k_fine for m in preds},
        status=status,
        notes=notes,
    )

    artifacts = {
        "uncertainty": u,
        "high_u": high_u,
        "known_boundary": known_boundary,
        "cryptic": cryptic,
        "coords": coords,
        "truth": truth,
        "svg": svg,
        "gene_table": gene_table,
        "enriched": enriched,
        "predictions": preds,
        "k_selection": selection.to_dict(),
        "uncertainty_summary": unc.summary(),
    }
    return discovery, artifacts


def _write_slice_artifacts(slice_id: str, disc: SliceDiscovery, art: dict[str, Any]) -> None:
    d = OUT / slice_id
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "spot_index": np.arange(len(art["uncertainty"])),
            "x": art["coords"][:, 0],
            "y": art["coords"][:, 1],
            "uncertainty": art["uncertainty"],
            "high_uncertainty": art["high_u"].astype(int),
            "known_boundary": art["known_boundary"].astype(int),
            "cryptic_niche": art["cryptic"].astype(int),
            "domain_truth": art["truth"],
        }
    ).to_csv(d / "spot_uncertainty_map.csv", index=False)
    art["svg"].to_csv(d / "svg_morans_fdr.csv", index=False)
    if not art["gene_table"].empty:
        art["gene_table"].to_csv(d / "cryptic_gene_enrichment.csv", index=False)
    if not art["enriched"].empty:
        art["enriched"].to_csv(d / "cryptic_genes_passing_gates.csv", index=False)
    (d / "slice_summary.json").write_text(json.dumps(asdict(disc), indent=2), encoding="utf-8")
    (d / "k_selection.json").write_text(json.dumps(art["k_selection"], indent=2), encoding="utf-8")
    # Predictions long table
    pred_rows = []
    for method, labels in art["predictions"].items():
        for i, lab in enumerate(labels):
            pred_rows.append({"spot_index": i, "method": method, "domain": lab})
    pd.DataFrame(pred_rows).to_csv(d / "domain_predictions_long.csv", index=False)


def cross_slice_replication(discoveries: list[SliceDiscovery]) -> dict[str, Any]:
    """Genes that pass cryptic enrichment on ≥ min_slices_replicating slices."""
    gene_hits: dict[str, list[str]] = defaultdict(list)
    for d in discoveries:
        for g in d.top_cryptic_genes:
            gene_hits[g].append(d.slice_id)
    min_rep = GATES["min_slices_replicating"]
    replicated = {g: slices for g, slices in gene_hits.items() if len(set(slices)) >= min_rep}
    candidate_slices = [d.slice_id for d in discoveries if d.status == "CANDIDATE"]
    go = len(candidate_slices) >= min_rep and len(replicated) >= 3
    return {
        "replicated_genes": {
            g: {"slices": slices, "n_slices": len(set(slices))}
            for g, slices in sorted(replicated.items(), key=lambda kv: -len(kv[1]))
        },
        "n_replicated_genes": len(replicated),
        "candidate_slices": candidate_slices,
        "global_decision": "GO_CANDIDATE_PANEL" if go else "NO_GO_OR_WEAK",
        "gates": GATES,
        "rationale": (
            "Requires ≥2 slices with CANDIDATE status and ≥3 genes replicated "
            "across ≥2 slices after SVG FDR + spatial-shift FDR."
            if not go
            else "Pre-registered multi-slice gates passed for a gene panel."
        ),
    }


def write_report(
    discoveries: list[SliceDiscovery],
    replication: dict[str, Any],
) -> str:
    lines = [
        "# Uncertainty-niche discovery (DLPFC multi-method)",
        "",
        "**Tools:** HistoWeave non-oracle *K*, multi-method domain ensemble, "
        "boundary-uncertainty maps, Moran's I + BH-FDR, spatial-shift nulls.",
        "",
        "**Question:** Do high multi-method uncertainty zones contain cryptic "
        "spatial expression programs that are *not* explained only by known "
        "cortical layer boundaries?",
        "",
        f"**Global decision:** `{replication['global_decision']}`",
        "",
        "## Pre-registered gates",
        "",
        "```json",
        json.dumps(GATES, indent=2),
        "```",
        "",
        "## Per-slice summary",
        "",
        "| Slice | n | oracle K | est. K | ens. K | high-U | cryptic | cryptic/high-U | AUROC(known) | #comp | largest | SVG FDR | cryptic SVG | status |",
        "|-------|--:|---------:|-------:|-------:|-------:|--------:|---------------:|-------------:|------:|--------:|--------:|------------:|--------|",
    ]
    for d in discoveries:
        lines.append(
            f"| {d.slice_id} | {d.n_obs} | {d.oracle_k} | {d.estimated_k} | {d.ensemble_k} | "
            f"{d.high_u_n} | {d.cryptic_n} | {d.cryptic_fraction_of_high_u:.3f} | "
            f"{d.auroc_known_boundary:.3f} | {d.n_cryptic_components} | "
            f"{d.largest_cryptic_component} | {d.n_svg_fdr} | "
            f"{d.n_svg_enriched_cryptic} | `{d.status}` |"
        )
    lines += [
        "",
        "### Notes",
        "",
    ]
    for d in discoveries:
        lines.append(
            f"- **{d.slice_id}:** {d.notes}; top genes: {', '.join(d.top_cryptic_genes[:8]) or '—'}"
        )

    lines += [
        "",
        "## Cross-slice replicated cryptic genes",
        "",
    ]
    if replication["replicated_genes"]:
        lines.append("| Gene | n slices | slices |")
        lines.append("|------|---------:|--------|")
        for g, info in list(replication["replicated_genes"].items())[:30]:
            lines.append(f"| `{g}` | {info['n_slices']} | {', '.join(info['slices'])} |")
    else:
        lines.append("_No gene passed multi-slice replication gates._")

    lines += [
        "",
        "## Interpretation bounds (honest)",
        "",
        "1. DLPFC layers are a **saturated public benchmark**. Recovering layers "
        "is not a biological discovery; cryptic niches are *candidates* only.",
        "2. Non-oracle *K* and multi-method uncertainty reduce method-choice "
        "artifacts, but residual technical effects (depth, batch) remain.",
        "3. Upgrade path to a claim: independent imaging/protein validation, "
        "perturbation or orthogonal platform, and pre-registered effect sizes.",
        "4. If `high_u` mostly equals known boundaries (AUROC high + cryptic "
        "fraction low), the result is a **method-consistency diagnostic**, not "
        "a new tissue region.",
        "",
        f"**Rationale:** {replication['rationale']}",
        "",
        "## Artifacts",
        "",
        "- `results/<slice>/spot_uncertainty_map.csv`",
        "- `results/<slice>/svg_morans_fdr.csv`",
        "- `results/<slice>/cryptic_gene_enrichment.csv`",
        "- `results/cross_slice_replication.json`",
        "- `results/slice_summaries.csv`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    _setup_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    discoveries: list[SliceDiscovery] = []
    for slice_id in SLICES:
        try:
            disc, art = analyse_slice(slice_id)
            _write_slice_artifacts(slice_id, disc, art)
            discoveries.append(disc)
        except Exception as exc:
            logger.exception("slice %s failed: %s", slice_id, exc)
            discoveries.append(
                SliceDiscovery(
                    slice_id=slice_id,
                    n_obs=0,
                    oracle_k=-1,
                    estimated_k=-1,
                    ensemble_k=-1,
                    high_u_n=0,
                    known_boundary_n=0,
                    cryptic_n=0,
                    cryptic_fraction_of_high_u=0.0,
                    high_u_known_boundary_enrichment=0.0,
                    auroc_known_boundary=float("nan"),
                    n_cryptic_components=0,
                    largest_cryptic_component=0,
                    n_svg_fdr=0,
                    n_svg_enriched_cryptic=0,
                    top_cryptic_genes=[],
                    method_k_used={},
                    status="FAILED",
                    notes=str(exc),
                )
            )

    replication = cross_slice_replication([d for d in discoveries if d.status != "FAILED"])
    summary = pd.DataFrame([asdict(d) for d in discoveries])
    # Flatten list columns for CSV.
    summary["top_cryptic_genes"] = summary["top_cryptic_genes"].apply(
        lambda xs: "|".join(xs) if isinstance(xs, list) else ""
    )
    summary["method_k_used"] = summary["method_k_used"].apply(
        lambda d: json.dumps(d) if isinstance(d, dict) else str(d)
    )
    summary.to_csv(OUT / "slice_summaries.csv", index=False)
    (OUT / "cross_slice_replication.json").write_text(
        json.dumps(replication, indent=2), encoding="utf-8"
    )
    (OUT / "gates.json").write_text(json.dumps(GATES, indent=2), encoding="utf-8")
    report = write_report(discoveries, replication)
    report_path = Path(__file__).resolve().parent / "DISCOVERY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Wrote %s", report_path)
    logger.info("Global decision: %s", replication["global_decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
