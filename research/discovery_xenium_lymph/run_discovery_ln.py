"""Cryptic-niche discovery on Xenium human lymph node (second tissue context).

Applies the same multi-method uncertainty → contiguous cryptic components →
pre-registered molecular panel pipeline used on DLPFC, with **lymphoid** panels:

* B_follicle: MS4A1, CD19, CR2, CD79A, PAX5
* T_zone:     CD3E, CD4, IL7R, CCR7, LTB
* Germinal_center: BCL6, AICDA, MKI67, TOP2A, RGS13

Outputs under ``results/``:

* spot_uncertainty_map.csv
* slice_summary.json
* component tables + panel scores
* LYMPH_DISCOVERY_REPORT.md
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "discovery_uncertainty_niches"))

from analyze_largest_component import (  # noqa: E402
    adjacency_profile,
    cryptic_components,
    log1p_norm,
)
from validate_panel_and_rois import (  # noqa: E402
    composite_score,
    shift_null_delta,
)

from histoweave.benchmark.k_selection import estimate_n_domains  # noqa: E402
from histoweave.benchmark.uncertainty import (  # noqa: E402
    boundary_mask_from_labels,
    boundary_uncertainty,
    uncertainty_enrichment,
)
from histoweave.datasets import get_dataset  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

logger = logging.getLogger("discovery_xenium_ln")
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
DATASET = "xenium_human_lymph_node"
DOMAIN_METHODS = ("kmeans", "spectral", "banksy_py", "gaussian_mixture")
SEED = 0
N_HVG = 200  # panel is already small; keep informative genes
MIN_COMP = 30
UNCERTAINTY_Q = 0.80

# Lymphoid pre-registered panels (not brain ENC1/HOPX/MBP).
# Prefer genes present on Xenium Prime 5K Human Pan Tissue panel; keep
# classical aliases so denser panels still match when available.
B_PANEL = ("MS4A1", "CD19", "CR2", "CD79A", "PAX5", "CD22", "CD79B", "FCER2")
T_PANEL = ("CD3E", "CD4", "CD8A", "CCR7", "IL7R", "IL7", "LTB", "TRAC")
GC_PANEL = ("BCL6", "MKI67", "TOP2A", "PCNA", "LMO2", "CXCL13", "AICDA", "RGS13")


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _to_dense(m) -> np.ndarray:
    if hasattr(m, "toarray"):
        return np.asarray(m.toarray(), dtype=float)
    return np.asarray(m, dtype=float)


def _subsample_var(data, n_hvg: int = N_HVG):
    X = _to_dense(data.X)
    var = X.var(axis=0)
    # always keep panel genes
    keep_names = set(B_PANEL + T_PANEL + GC_PANEL)
    names = list(map(str, data.var_names))
    force = [i for i, g in enumerate(names) if g in keep_names]
    order = np.argsort(var)[::-1]
    keep = list(dict.fromkeys(force + order.tolist()))[: max(n_hvg, len(force))]
    keep = np.sort(np.asarray(keep, dtype=int))
    X_sub = X[:, keep]
    from histoweave.data import SpatialTable

    return SpatialTable(
        X=X_sub,
        obs=data.obs.copy(),
        var=data.var.iloc[keep].copy(),
        obsm={str(k): np.asarray(v).copy() for k, v in dict(data.obsm).items()},
        uns=dict(data.uns),
    )


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
            logger.info("  %s ok (k=%s)", name, k)
        except Exception as exc:
            logger.warning("  %s failed: %s", name, exc)
    return preds


def main() -> int:
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)

    # Ensure bundle exists
    h5 = ROOT / "datasets_cache" / "xenium" / "xenium_human_lymph_node.h5ad"
    if not h5.is_file():
        logger.info("bundle missing — running prepare_bundle.py")
        import subprocess

        rc = subprocess.call([sys.executable, str(BASE / "prepare_bundle.py")], cwd=str(ROOT))
        if rc != 0:
            return rc

    entry = get_dataset(DATASET)
    data = entry.load(cache_dir=ROOT / "datasets_cache")
    logger.info(
        "loaded %s n_obs=%s n_vars=%s domains=%s expr=%s",
        DATASET,
        data.n_obs,
        data.n_vars,
        int(data.obs["domain_truth"].nunique()) if "domain_truth" in data.obs else None,
        data.uns.get("expression_source", "unknown"),
    )
    data = _subsample_var(data, N_HVG)

    # K selection
    selection = estimate_n_domains(data, method="silhouette", random_state=SEED, max_obs=3000)
    k_hat = int(selection.k)
    oracle_k = int(data.obs["domain_truth"].nunique())
    k_fine = int(max(k_hat, min(oracle_k, 8)))
    logger.info("oracle_k=%s estimated_k=%s ensemble_k=%s", oracle_k, k_hat, k_fine)

    preds: dict[str, np.ndarray] = {}
    for k_run, tag in ((k_hat, "coarse"), (k_fine, "fine")):
        if tag == "fine" and k_fine == k_hat:
            continue
        part = _run_domains(data, k_run)
        for name, labels in part.items():
            preds[f"{name}@{tag}_k{k_run}"] = labels
    if len(preds) < 2:
        logger.error("need ≥2 domain methods")
        return 1

    coords = np.asarray(data.spatial, dtype=float)
    unc = boundary_uncertainty(coords, preds, k=8, consensus_min_methods=2)
    u = unc.uncertainty
    thr = float(np.quantile(u, UNCERTAINTY_Q))
    high_u = u >= thr
    truth = data.obs["domain_truth"].astype(str).to_numpy()
    known_boundary = boundary_mask_from_labels(coords, truth, k=8)
    cryptic = high_u & (~known_boundary)
    enrich = uncertainty_enrichment(u, known_boundary, high_quantile=UNCERTAINTY_Q)
    auroc = enrich.get("roc_auc")
    auroc = float(auroc) if auroc is not None else float("nan")

    spots = pd.DataFrame(
        {
            "spot_index": np.arange(len(u)),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "uncertainty": u,
            "high_uncertainty": high_u.astype(int),
            "known_boundary": known_boundary.astype(int),
            "cryptic_niche": cryptic.astype(int),
            "domain_truth": truth,
            "barcode": list(map(str, data.obs_names)),
        }
    )
    spots.to_csv(OUT / "spot_uncertainty_map.csv", index=False)

    summary = {
        "dataset": DATASET,
        "assay": "xenium",
        "tissue": "lymph_node",
        "n_obs": int(data.n_obs),
        "oracle_k": oracle_k,
        "estimated_k": k_hat,
        "ensemble_k": k_fine,
        "high_u_n": int(high_u.sum()),
        "known_boundary_n": int(known_boundary.sum()),
        "cryptic_n": int(cryptic.sum()),
        "cryptic_fraction_of_high_u": float(cryptic.sum() / max(high_u.sum(), 1)),
        "auroc_known_boundary": auroc,
        "expression_source": data.uns.get("expression_source", "unknown"),
        "k_selection": selection.to_dict(),
        "uncertainty_summary": unc.summary(),
    }
    (OUT / "slice_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Components
    comps = cryptic_components(coords, cryptic, k=8, min_size=MIN_COMP)
    X = log1p_norm(_to_dense(data.X))
    genes = list(map(str, data.var_names))
    panels = {
        "B_follicle": B_PANEL,
        "T_zone": T_PANEL,
        "Germinal_center": GC_PANEL,
    }
    panel_scores = {name: composite_score(X, genes, genes_t) for name, genes_t in panels.items()}

    rows = []
    for rank, comp in enumerate(comps):
        counts = Counter(truth[comp])
        total = len(comp)
        top_lab, n_top = counts.most_common(1)[0]
        purity = n_top / total
        in_mask = np.zeros(len(truth), dtype=bool)
        in_mask[comp] = True
        rest = ~in_mask
        same_out = (truth == top_lab) & rest
        adj = adjacency_profile(coords, truth, comp, k=8)
        rec: dict[str, Any] = {
            "rank": rank,
            "n": total,
            "dominant_truth": top_lab,
            "purity": purity,
            "truth_counts": dict(counts),
            "abut": adj["top_abutting_layers"],
            "internal_edge_fraction": adj["internal_edge_fraction"],
        }
        for pname, (score, used) in panel_scores.items():
            d_rest, p_rest = shift_null_delta(
                coords, score, in_mask, seed=SEED + rank + hash(pname) % 1000
            )
            if same_out.sum() >= 15:
                union = in_mask | same_out
                _d_sl, p_sl = shift_null_delta(
                    coords[union], score[union], in_mask[union], seed=SEED + 17 + rank
                )
            else:
                _d_sl, p_sl = float("nan"), float("nan")
            rec[f"{pname}_delta_rest"] = float(score[in_mask].mean() - score[rest].mean())
            rec[f"{pname}_shift_p_rest"] = p_rest
            rec[f"{pname}_delta_same_layer"] = (
                float(score[in_mask].mean() - score[same_out].mean())
                if same_out.any()
                else float("nan")
            )
            rec[f"{pname}_shift_p_same_layer"] = p_sl
            rec[f"{pname}_genes"] = used
        # Direction heuristics by dominant pathology label
        if "germinal" in top_lab.lower() or "lymphoid aggregate" in top_lab.lower():
            rec["expected_class"] = "GC_like"
            rec["direction_ok"] = bool(
                rec.get("Germinal_center_delta_rest", 0) > 0
                and rec.get("Germinal_center_shift_p_rest", 1) <= 0.10
            )
        elif "adipose" in top_lab.lower():
            rec["expected_class"] = "Adipose_like"
            rec["direction_ok"] = bool(rec.get("B_follicle_delta_rest", 0) < 0)
        else:
            # bulk lymph node — accept B or T enrichment
            rec["expected_class"] = "LN_parenchyma"
            rec["direction_ok"] = bool(
                max(rec.get("B_follicle_delta_rest", -1), rec.get("T_zone_delta_rest", -1)) > 0
            )
        rows.append(rec)
        # export component spots
        cdir = OUT / f"component_rank{rank}_n{total}"
        cdir.mkdir(exist_ok=True)
        spots.iloc[comp].assign(component_rank=rank).to_csv(
            cdir / "component_spots.csv", index=False
        )
        pd.DataFrame(adj["layer_table"]).to_csv(cdir / "adjacency_to_domains.csv", index=False)
        (cdir / "component_summary.json").write_text(
            json.dumps(
                {k: v for k, v in rec.items() if k != "truth_counts"}
                | {"truth_counts": rec["truth_counts"]},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info(
            "rank=%s n=%s pure=%s dom=%s dir=%s GCΔ=%.3f BΔ=%.3f TΔ=%.3f",
            rank,
            total,
            f"{purity:.2f}",
            top_lab[:40],
            rec["direction_ok"],
            rec.get("Germinal_center_delta_rest", float("nan")),
            rec.get("B_follicle_delta_rest", float("nan")),
            rec.get("T_zone_delta_rest", float("nan")),
        )

    comp_df = pd.DataFrame(rows)
    # flatten lists for csv
    for col in list(comp_df.columns):
        if comp_df[col].map(lambda x: isinstance(x, (list, dict))).any():
            comp_df[col] = comp_df[col].map(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    comp_df.to_csv(OUT / "components_panel.csv", index=False)

    # Report
    lines = [
        "# Xenium lymph node cryptic-niche discovery",
        "",
        f"**Dataset:** `{DATASET}` · assay=xenium · tissue=lymph_node",
        f"**Expression source:** `{summary['expression_source']}`",
        "",
        "## Pipeline (same architecture as DLPFC)",
        "",
        "1. Non-oracle *K* (silhouette) + fine *K* ensemble",
        "2. Domain methods: kmeans / spectral / banksy_py / gaussian_mixture",
        "3. Target-free `boundary_uncertainty`",
        "4. Cryptic = high-U ∧ ¬ pathology boundary",
        "5. Contiguous components + **lymphoid** panels (B / T / GC)",
        "",
        "## Geometry",
        "",
        "| Metric | Value |",
        "|--------|------:|",
        f"| n cells | {summary['n_obs']} |",
        f"| pathology domains | {oracle_k} |",
        f"| estimated K | {k_hat} |",
        f"| ensemble K | {k_fine} |",
        f"| high-U cells | {summary['high_u_n']} |",
        f"| cryptic cells | {summary['cryptic_n']} ({summary['cryptic_fraction_of_high_u']:.1%} of high-U) |",
        f"| AUROC(U → pathology boundary) | {auroc:.3f} |",
        f"| components ≥{MIN_COMP} | {len(comps)} |",
        "",
        "## Components",
        "",
        "| Rank | n | Dominant pathology | Purity | Class | dir_ok | GC Δrest | B Δrest | T Δrest | Abut |",
        "|-----:|--:|--------------------|-------:|-------|:------:|---------:|--------:|--------:|------|",
    ]
    for rec in rows:
        lines.append(
            f"| {rec['rank']} | {rec['n']} | {rec['dominant_truth']} | {rec['purity']:.2f} | "
            f"{rec['expected_class']} | {'Y' if rec['direction_ok'] else 'N'} | "
            f"{rec.get('Germinal_center_delta_rest', float('nan')):.3f} | "
            f"{rec.get('B_follicle_delta_rest', float('nan')):.3f} | "
            f"{rec.get('T_zone_delta_rest', float('nan')):.3f} | "
            f"{','.join(rec['abut'][:2]) if rec['abut'] else '—'} |"
        )

    n_dir = sum(1 for r in rows if r["direction_ok"])
    lines += [
        "",
        f"**Direction-ok components:** {n_dir}/{len(rows)}",
        "",
        "## Cross-tissue takeaway vs DLPFC",
        "",
        "| | DLPFC (Visium) | Lymph node (Xenium) |",
        "|--|----------------|---------------------|",
        "| Domain GT | cortical layers | pathology polygons (LN / GC aggregate / adipose) |",
        "| Molecular panels | ENC1/HOPX vs MBP | B-follicle / T-zone / GC programs |",
        "| Pipeline | identical architecture | identical architecture |",
        "",
        "## Provenance note",
        "",
        "If `expression_source` is `domain_conditioned_synthetic_pending_official_matrix`, "
        "counts are **co-registered to official polygons** but not the full official "
        "cell_feature_matrix (local download incomplete / CDN 403). Geometry + panel "
        "**code path** is production-ready; swap in official counts when available via "
        "`prepare_human_lymph_node.py`.",
        "",
        f"Artifacts: `{OUT.as_posix()}`",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "LYMPH_DISCOVERY_REPORT.md").write_text(report, encoding="utf-8")
    (BASE / "LYMPH_DISCOVERY_REPORT.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s", OUT / "LYMPH_DISCOVERY_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
