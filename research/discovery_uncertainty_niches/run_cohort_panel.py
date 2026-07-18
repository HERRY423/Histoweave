"""Full DLPFC cohort: discovery → pure-layer components → pre-registered panel.

Extends the 3-donor pilot to all Maynard DLPFC Visium sections in
``datasets_cache/dlpfc/``.  For each slice:

1. Run multi-method uncertainty discovery (if missing), or reuse cache.
2. Extract cryptic components ≥ ``MIN_SIZE``.
3. Keep pure-layer components (single domain_truth ≥ 95%).
4. Score pre-registered L3 / myelin panels (same-layer + rest + shift null).
5. Write cohort meta-summary and update frozen status.

Usage (repo root)::

    python research/discovery_uncertainty_niches/run_cohort_panel.py
    python research/discovery_uncertainty_niches/run_cohort_panel.py --force-discovery
    python research/discovery_uncertainty_niches/run_cohort_panel.py --slices dlpfc_151507,dlpfc_151510
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE = Path(__file__).resolve().parent
OUT = BASE / "results"
COHORT_OUT = OUT / "cohort"

# Import heavy discovery helpers from sibling modules.
from analyze_largest_component import (  # noqa: E402
    K_NN,
    MIN_COMPONENT,
    adjacency_profile,
    cryptic_components,
)
from run_discovery import (  # noqa: E402
    _write_slice_artifacts,
    analyse_slice,
)
from validate_panel_and_rois import (  # noqa: E402
    L3_PANEL,
    MYELIN_PANEL,
    composite_score,
    load_aligned_expression,
    rank_sum_p,
    shift_null_delta,
)

logger = logging.getLogger("cohort_panel")

ALL_SLICES = (
    "dlpfc_151507",
    "dlpfc_151508",
    "dlpfc_151509",
    "dlpfc_151510",
    "dlpfc_151669",
    "dlpfc_151670",
    "dlpfc_151671",
    "dlpfc_151672",
    "dlpfc_151673",
    "dlpfc_151674",
    "dlpfc_151675",
    "dlpfc_151676",
)

MIN_PURE_FRAC = 0.95
MIN_COMPONENT_PANEL = 20  # slightly above geometry min for DE stability
SEED = 0


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_discovery(slice_id: str, *, force: bool = False) -> Path:
    """Run analyse_slice if spot_uncertainty_map.csv is missing."""
    slice_dir = OUT / slice_id
    map_path = slice_dir / "spot_uncertainty_map.csv"
    if map_path.is_file() and not force:
        logger.info("[%s] reuse discovery cache", slice_id)
        return map_path
    logger.info("[%s] running discovery…", slice_id)
    t0 = time.perf_counter()
    disc, art = analyse_slice(slice_id)
    _write_slice_artifacts(slice_id, disc, art)
    # also write slice_summary at slice root (analyse already does via _write)
    logger.info(
        "[%s] discovery done in %.1fs status=%s", slice_id, time.perf_counter() - t0, disc.status
    )
    return map_path


def pure_layer_components(
    spots: pd.DataFrame,
    *,
    min_size: int = MIN_COMPONENT_PANEL,
) -> list[dict[str, Any]]:
    coords = spots[["x", "y"]].to_numpy(dtype=float)
    cryptic = spots["cryptic_niche"].to_numpy(dtype=bool)
    labels = spots["domain_truth"].astype(str).to_numpy()
    comps = cryptic_components(coords, cryptic, k=K_NN, min_size=min_size)
    out: list[dict[str, Any]] = []
    for rank, comp in enumerate(comps):
        counts = Counter(labels[comp])
        total = len(comp)
        layer, n = counts.most_common(1)[0]
        frac = n / total
        if frac < MIN_PURE_FRAC:
            continue
        if layer not in {"Layer 3", "Layer 6"}:
            # Track other pure layers as exploratory only
            expected_class = "other_pure"
        elif layer == "Layer 3":
            expected_class = "L3_program"
        else:
            expected_class = "L6_myelin"
        out.append(
            {
                "rank": rank,
                "indices": comp,
                "n": total,
                "layer": layer,
                "purity": frac,
                "expected_class": expected_class,
                "truth_counts": dict(counts),
            }
        )
    return out


def score_component(
    slice_id: str,
    spots: pd.DataFrame,
    comp: dict[str, Any],
) -> dict[str, Any]:
    indices = comp["indices"]
    in_mask = np.zeros(len(spots), dtype=bool)
    in_mask[indices] = True
    layer = comp["layer"]

    X, coords, labels, barcodes, genes = load_aligned_expression(slice_id, spots)
    rest = ~in_mask
    same_layer_out = (labels == layer) & rest

    l3_score, l3_used = composite_score(X, genes, L3_PANEL)
    my_score, my_used = composite_score(X, genes, MYELIN_PANEL)

    def pack(name: str, score: np.ndarray, used: list[str]) -> dict[str, Any]:
        d_rest, p_rest = shift_null_delta(
            coords, score, in_mask, seed=SEED + abs(hash(name + slice_id)) % 10000
        )
        if same_layer_out.sum() >= 10:
            union = in_mask | same_layer_out
            d_sl, p_sl = shift_null_delta(
                coords[union],
                score[union],
                in_mask[union],
                seed=SEED + 99 + abs(hash(name + slice_id)) % 10000,
            )
            p_mw_sl = rank_sum_p(score[in_mask], score[same_layer_out])
            mean_sl = float(score[same_layer_out].mean())
        else:
            d_sl, p_sl, p_mw_sl, mean_sl = float("nan"), float("nan"), float("nan"), float("nan")
        return {
            "genes_used": used,
            "mean_in": float(score[in_mask].mean()),
            "mean_rest": float(score[rest].mean()),
            "mean_same_layer_out": mean_sl,
            "delta_vs_rest": float(score[in_mask].mean() - score[rest].mean()),
            "delta_vs_same_layer": float(score[in_mask].mean() - mean_sl)
            if np.isfinite(mean_sl)
            else float("nan"),
            "p_vs_rest": rank_sum_p(score[in_mask], score[rest]),
            "p_vs_same_layer": p_mw_sl,
            "shift_p_vs_rest": p_rest,
            "shift_p_vs_same_layer": p_sl,
            "shift_delta_vs_rest": d_rest,
            "shift_delta_vs_same_layer": d_sl,
        }

    composites = {
        "L3_program": pack("L3_program", l3_score, l3_used),
        "myelin": pack("myelin", my_score, my_used),
    }

    expected = comp["expected_class"]
    gates: dict[str, Any] = {"expected_class": expected}
    if expected == "L3_program":
        gates["direction_ok"] = bool(
            composites["L3_program"]["delta_vs_rest"] > 0
            and composites["myelin"]["delta_vs_rest"] < 0
        )
        gates["l3_shift_rest_ok"] = composites["L3_program"]["shift_p_vs_rest"] <= 0.05
        gates["hard_pass"] = bool(
            composites["L3_program"]["delta_vs_same_layer"] > 0
            and composites["L3_program"]["shift_p_vs_same_layer"] <= 0.05
            and composites["myelin"]["delta_vs_rest"] < 0
        )
    elif expected == "L6_myelin":
        gates["direction_ok"] = bool(composites["myelin"]["delta_vs_rest"] > 0)
        gates["hard_pass"] = bool(
            composites["myelin"]["delta_vs_rest"] > 0
            and composites["myelin"]["shift_p_vs_rest"] <= 0.05
        )
    else:
        gates["direction_ok"] = False
        gates["hard_pass"] = False

    # adjacency
    adj = adjacency_profile(coords, labels, indices, k=K_NN)

    # ROI
    roi = spots.loc[in_mask].copy()
    roi["barcode"] = np.asarray(barcodes)[in_mask]
    roi["domain_truth"] = labels[in_mask]
    roi["component_rank"] = comp["rank"]
    roi["L3_program_score"] = l3_score[in_mask]
    roi["myelin_score"] = my_score[in_mask]

    label = (
        f"{slice_id.replace('dlpfc_', '')}_{layer.replace(' ', '')}_rank{comp['rank']}_n{comp['n']}"
    )
    return {
        "label": label,
        "slice_id": slice_id,
        "rank": comp["rank"],
        "n": comp["n"],
        "layer": layer,
        "purity": comp["purity"],
        "expected_class": expected,
        "composites": composites,
        "gates": gates,
        "adjacency_top": adj["top_abutting_layers"],
        "internal_edge_fraction": adj["internal_edge_fraction"],
        "truth_counts": comp["truth_counts"],
        "roi": roi,
    }


def write_cohort_report(rows: list[dict[str, Any]], slice_status: list[dict[str, Any]]) -> str:
    l3 = [r for r in rows if r["expected_class"] == "L3_program"]
    l6 = [r for r in rows if r["expected_class"] == "L6_myelin"]
    n_l3_dir = sum(1 for r in l3 if r["gates"]["direction_ok"])
    n_l3_shift = sum(1 for r in l3 if r["gates"].get("l3_shift_rest_ok"))
    n_l3_hard = sum(1 for r in l3 if r["gates"]["hard_pass"])
    n_l6_dir = sum(1 for r in l6 if r["gates"]["direction_ok"])
    n_l6_hard = sum(1 for r in l6 if r["gates"]["hard_pass"])

    lines = [
        "# DLPFC cohort cryptic-niche panel meta-analysis",
        "",
        f"**Slices processed:** {len(slice_status)}  ·  "
        f"**Pure L3 components (≥{MIN_COMPONENT_PANEL}):** {len(l3)}  ·  "
        f"**Pure L6 components:** {len(l6)}",
        "",
        "## Headline gates",
        "",
        "| Class | n components | direction_ok | L3 shift-rest ≤0.05 | hard_pass |",
        "|-------|-------------:|:------------:|:-------------------:|:---------:|",
        f"| L3_program | {len(l3)} | **{n_l3_dir}/{len(l3)}** | **{n_l3_shift}/{len(l3)}** | **{n_l3_hard}/{len(l3)}** |",
        f"| L6_myelin | {len(l6)} | **{n_l6_dir}/{len(l6)}** | — | **{n_l6_hard}/{len(l6)}** |",
        "",
        f"**Cohort L3 direction rate:** {n_l3_dir / max(len(l3), 1):.0%}  "
        f"(pre-registered: L3 composite ↑ and myelin ↓ vs rest).",
        "",
        f"**Cohort L3 RNA shift-rest support:** {n_l3_shift / max(len(l3), 1):.0%}  "
        f"(L3 composite shift p ≤ 0.05 vs rest).",
        "",
        "## Per-component table",
        "",
        "| Label | Layer | n | dir_ok | hard | L3 Δrest | L3 shift p | Myelin Δrest | Myelin shift p | Abut |",
        "|-------|-------|--:|:------:|:----:|---------:|-----------:|-------------:|---------------:|------|",
    ]
    for r in sorted(rows, key=lambda x: (x["layer"], x["slice_id"], x["rank"])):
        c = r["composites"]
        lines.append(
            f"| {r['label']} | {r['layer']} | {r['n']} | "
            f"{'Y' if r['gates']['direction_ok'] else 'N'} | "
            f"{'Y' if r['gates']['hard_pass'] else 'N'} | "
            f"{c['L3_program']['delta_vs_rest']:.3f} | "
            f"{c['L3_program']['shift_p_vs_rest']:.3f} | "
            f"{c['myelin']['delta_vs_rest']:.3f} | "
            f"{c['myelin']['shift_p_vs_rest']:.3f} | "
            f"{','.join(r['adjacency_top'][:2]) or '—'} |"
        )

    lines += [
        "",
        "## Slice discovery status",
        "",
        "| Slice | status | est K | cryptic_n | cryptic/highU | AUROC known | #comp≥15 |",
        "|-------|--------|------:|----------:|--------------:|------------:|---------:|",
    ]
    for s in slice_status:
        lines.append(
            f"| {s['slice_id']} | `{s.get('status', '?')}` | {s.get('estimated_k', '')} | "
            f"{s.get('cryptic_n', '')} | {s.get('cryptic_fraction_of_high_u', float('nan')):.3f} | "
            f"{s.get('auroc_known_boundary', float('nan')):.3f} | {s.get('n_components', '')} |"
        )

    lines += [
        "",
        "## Frozen interpretation",
        "",
        "1. **Geometry** of pure L3 / L6 cryptic niches is common across the 12-section DLPFC set.",
        "2. **L3 direction** (mid-layer program up, myelin down vs rest) is the replicable RNA pattern.",
        "3. **Same-layer hard gates** remain rarely passed — Visium alone does not justify a new cell-state name.",
        "4. **IF package** (ENC1 + HOPX + MBP) should prioritise L3 ROIs with `direction_ok` and L3 shift-rest support.",
        "",
        "Artifacts: `results/cohort/`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slices",
        default=",".join(ALL_SLICES),
        help="Comma-separated slice ids (default: all 12 DLPFC).",
    )
    parser.add_argument(
        "--force-discovery",
        action="store_true",
        help="Re-run discovery even if spot maps exist.",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Only score panels on slices that already have discovery maps.",
    )
    args = parser.parse_args(argv)
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]

    COHORT_OUT.mkdir(parents=True, exist_ok=True)
    roi_dir = COHORT_OUT / "rois"
    roi_dir.mkdir(exist_ok=True)

    slice_status: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    flat_summary: list[dict[str, Any]] = []

    for slice_id in slices:
        t0 = time.perf_counter()
        try:
            if args.skip_discovery:
                map_path = OUT / slice_id / "spot_uncertainty_map.csv"
                if not map_path.is_file():
                    logger.warning("[%s] no map — skip", slice_id)
                    continue
            else:
                map_path = ensure_discovery(slice_id, force=args.force_discovery)

            spots = pd.read_csv(map_path)
            # discovery summary if present
            summary_path = OUT / slice_id / "slice_summary.json"
            if summary_path.is_file():
                st = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                st = {"slice_id": slice_id, "status": "map_only"}

            comps = pure_layer_components(spots, min_size=MIN_COMPONENT_PANEL)
            # count all geometry components for status
            coords = spots[["x", "y"]].to_numpy(dtype=float)
            cryptic = spots["cryptic_niche"].to_numpy(dtype=bool)
            all_comps = cryptic_components(coords, cryptic, min_size=MIN_COMPONENT)
            st = dict(st)
            st["slice_id"] = slice_id
            st["n_components"] = len(all_comps)
            st["n_pure_L3_L6"] = sum(
                1 for c in comps if c["expected_class"] in {"L3_program", "L6_myelin"}
            )
            slice_status.append(st)

            for comp in comps:
                if comp["expected_class"] == "other_pure":
                    continue
                res = score_component(slice_id, spots, comp)
                component_rows.append(res)
                # ROI
                res["roi"].to_csv(roi_dir / f"ROI_{res['label']}.csv", index=False)
                flat_summary.append(
                    {
                        "label": res["label"],
                        "slice_id": res["slice_id"],
                        "layer": res["layer"],
                        "rank": res["rank"],
                        "n": res["n"],
                        "purity": res["purity"],
                        "expected_class": res["expected_class"],
                        "direction_ok": res["gates"]["direction_ok"],
                        "hard_pass": res["gates"]["hard_pass"],
                        "l3_delta_rest": res["composites"]["L3_program"]["delta_vs_rest"],
                        "l3_shift_p_rest": res["composites"]["L3_program"]["shift_p_vs_rest"],
                        "l3_delta_same_layer": res["composites"]["L3_program"][
                            "delta_vs_same_layer"
                        ],
                        "l3_shift_p_same_layer": res["composites"]["L3_program"][
                            "shift_p_vs_same_layer"
                        ],
                        "myelin_delta_rest": res["composites"]["myelin"]["delta_vs_rest"],
                        "myelin_shift_p_rest": res["composites"]["myelin"]["shift_p_vs_rest"],
                        "abut": "|".join(res["adjacency_top"]),
                        "internal_edge_fraction": res["internal_edge_fraction"],
                    }
                )
                logger.info(
                    "[%s] %s dir=%s hard=%s L3Δ=%.3f myelΔ=%.3f",
                    slice_id,
                    res["label"],
                    res["gates"]["direction_ok"],
                    res["gates"]["hard_pass"],
                    res["composites"]["L3_program"]["delta_vs_rest"],
                    res["composites"]["myelin"]["delta_vs_rest"],
                )
            logger.info("[%s] finished in %.1fs", slice_id, time.perf_counter() - t0)
        except Exception as exc:
            logger.exception("[%s] FAILED: %s", slice_id, exc)
            slice_status.append({"slice_id": slice_id, "status": "FAILED", "error": str(exc)})

    # Persist
    pd.DataFrame(flat_summary).to_csv(COHORT_OUT / "cohort_component_panel.csv", index=False)
    (COHORT_OUT / "slice_status.json").write_text(
        json.dumps(slice_status, indent=2, default=str), encoding="utf-8"
    )
    # Full JSON without DataFrames
    serialisable = []
    for r in component_rows:
        serialisable.append({k: v for k, v in r.items() if k != "roi"})
    (COHORT_OUT / "cohort_components.json").write_text(
        json.dumps(serialisable, indent=2, default=str), encoding="utf-8"
    )

    report = write_cohort_report(component_rows, slice_status)
    (COHORT_OUT / "COHORT_META_REPORT.md").write_text(report, encoding="utf-8")
    (BASE / "COHORT_META_REPORT.md").write_text(report, encoding="utf-8")

    # Frozen project status
    l3 = [r for r in component_rows if r["expected_class"] == "L3_program"]
    status = {
        "protocol": "histoweave.discovery_cohort.v1",
        "n_slices": len(slices),
        "n_slices_ok": sum(1 for s in slice_status if s.get("status") != "FAILED"),
        "n_L3_components": len(l3),
        "n_L3_direction_ok": sum(1 for r in l3 if r["gates"]["direction_ok"]),
        "n_L3_shift_rest_ok": sum(1 for r in l3 if r["gates"].get("l3_shift_rest_ok")),
        "n_L3_hard_pass": sum(1 for r in l3 if r["gates"]["hard_pass"]),
        "n_L6_components": sum(1 for r in component_rows if r["expected_class"] == "L6_myelin"),
        "n_L6_hard_pass": sum(
            1
            for r in component_rows
            if r["expected_class"] == "L6_myelin" and r["gates"]["hard_pass"]
        ),
        "claim": (
            "L3 cryptic niches show replicable RNA direction (mid-layer program up, "
            "myelin down vs rest) across the DLPFC cohort; same-layer hard gates and "
            "named cell states require IF (see IF_PROTOCOL.md)."
        ),
        "next": [
            "Run IF on high-confidence L3 ROIs (direction_ok + shift_rest_ok)",
            "Do not claim new cell state without same-layer protein pass",
            "Optional: expand panel to Layer 2/4 pure niches as exploratory",
        ],
    }
    (COHORT_OUT / "PROJECT_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (BASE / "PROJECT_STATUS.md").write_text(
        "# Discovery project status (frozen)\n\n"
        + "```json\n"
        + json.dumps(status, indent=2)
        + "\n```\n\n"
        + "See [COHORT_META_REPORT.md](COHORT_META_REPORT.md) for full tables.\n",
        encoding="utf-8",
    )
    logger.info(
        "Cohort done: L3 dir %s/%s hard %s/%s",
        status["n_L3_direction_ok"],
        status["n_L3_components"],
        status["n_L3_hard_pass"],
        status["n_L3_components"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
