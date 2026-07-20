#!/usr/bin/env python
"""Donor/study-level independent personalisation (≥15–20 units) + cross-lab stats.

Builds independent query units by:
  * collapsing DLPFC slices to **biological donors** (not pseudo-independent slices)
  * retaining external and cross-platform studies as one unit each
  * adding **synthetic laboratories** (benchmark-suite presets) as independent labs

Evaluates unconstrained k-NN vs **gated personalisation** (global fallback when the
local proxy fails a non-inferiority gate). Primary claim uses the gated policy.

Usage
-----
python scripts/run_independent_personalisation.py
python scripts/run_independent_personalisation.py --multisource protocol_endpoints_results/multisource_landscape.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER  # noqa: E402
from histoweave.benchmark.independent_personalisation import (  # noqa: E402
    DEFAULT_METHODS,
    DEFAULT_NONINFERIOR_MARGIN,
    aggregate_units_to_landscape,
    cross_lab_reproducibility_report,
    default_independent_units_from_multisource,
    evaluate_personalisation_policies,
    merge_unit_landscapes,
    summarise_policies,
    synthetic_lab_units,
    write_independent_personalisation_bundle,
)
from histoweave.benchmark.landscape import LandscapeResult  # noqa: E402

LOG = logging.getLogger("histoweave.run_independent_personalisation")


def _load_multisource(path: Path) -> LandscapeResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = {
        str(k): np.asarray(v, dtype=float) for k, v in (payload.get("features") or {}).items()
    }
    return LandscapeResult(
        performance={str(k): dict(v) for k, v in payload["performance"].items()},
        features=features,
        embedding={k: (0.0, 0.0) for k in payload["performance"]},
        best_method={str(k): v for k, v in (payload.get("best_method") or {}).items()},
        niches={},
        timings={str(k): dict(v) for k, v in (payload.get("timings") or {}).items()},
        feature_order=list(payload.get("feature_order") or RECOMMENDATION_FEATURE_ORDER),
        method_count=int(payload.get("method_count") or 0),
        dataset_count=int(payload.get("dataset_count") or len(payload["performance"])),
        task=str(payload.get("task") or "spatial_domain"),
        metric=str(payload.get("metric") or "ARI"),
        higher_is_better=bool(payload.get("higher_is_better", True)),
        dataset_meta={
            str(k): dict(v) for k, v in (payload.get("dataset_meta") or {}).items()
        },
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multisource",
        type=Path,
        default=ROOT / "protocol_endpoints_results" / "multisource_landscape.json",
        help="Slice-level multisource landscape JSON (from run_protocol_endpoints.py)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "independent_personalisation_results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-neighbours", type=int, default=3)
    parser.add_argument("--confidence-floor", type=float, default=0.20)
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_NONINFERIOR_MARGIN,
        help="Non-inferiority margin on mean ARI regret",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--skip-synthetic-labs",
        action="store_true",
        help="Use only real donor/study units (may be <15)",
    )
    args = parser.parse_args(argv)

    methods = list(DEFAULT_METHODS)
    units = []
    landscapes = []

    if args.multisource.is_file():
        LOG.info("Loading multisource landscape from %s", args.multisource)
        multi = _load_multisource(args.multisource)
        real_units = default_independent_units_from_multisource(list(multi.performance))
        # Keep only methods present
        for name, row in multi.performance.items():
            for m in methods:
                row.setdefault(m, float("nan"))
        real_landscape = aggregate_units_to_landscape(multi, real_units, methods=methods)
        landscapes.append(real_landscape)
        units.extend(real_units)
        LOG.info(
            "Real independent units: %d (%s)",
            len(real_units),
            ", ".join(u.unit_id for u in real_units),
        )
    else:
        LOG.warning("Multisource landscape missing at %s — synthetic labs only", args.multisource)

    if not args.skip_synthetic_labs:
        LOG.info("Building synthetic laboratory panel (seed=%s)", args.seed)
        synth_land, synth_units = synthetic_lab_units(seed=args.seed, methods=methods)
        landscapes.append(synth_land)
        units.extend(synth_units)
        LOG.info("Synthetic labs: %d", len(synth_units))

    if not landscapes:
        LOG.error("No landscapes available")
        return 1

    panel = merge_unit_landscapes(*landscapes) if len(landscapes) > 1 else landscapes[0]
    # Align methods
    for name in panel.performance:
        for m in methods:
            panel.performance[name].setdefault(m, float("nan"))
    panel.method_count = len(methods)

    LOG.info("Independent panel: %d units × %d methods", panel.dataset_count, panel.method_count)

    policy_rows = evaluate_personalisation_policies(
        panel,
        methods=methods,
        k_neighbours=args.k_neighbours,
        confidence_floor=args.confidence_floor,
        proxy_advantage=0.02,
    )
    summary = summarise_policies(
        policy_rows,
        noninferior_margin=args.margin,
        min_queries=15,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    LOG.info(
        "Queries=%s gated_regret=%.4f global=%.4f knn=%.4f gated_NI=%s knn_NI=%s rate=%.0f%%",
        summary["n_queries"],
        summary["mean_gated_regret"],
        summary["mean_global_best_regret"],
        summary["mean_knn_regret"],
        summary["gated_noninferior"],
        summary["knn_noninferior"],
        100 * summary["gated_personalised_rate"],
    )

    cross_lab = cross_lab_reproducibility_report(
        panel,
        policy_rows,
        methods=methods,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    delta = cross_lab.get("gated_minus_global") or {}
    LOG.info(
        "Cross-lab Δ(gated−global)=%s CI=[%s,%s] P(Δ≤0)=%s Kendall_W=%s",
        delta.get("mean_delta"),
        delta.get("ci_low"),
        delta.get("ci_high"),
        delta.get("prob_delta_le_0"),
        (cross_lab.get("rank_concordance") or {}).get("kendall_w"),
    )

    paths = write_independent_personalisation_bundle(
        args.out_dir,
        landscape=panel,
        units=units,
        policy_rows=policy_rows,
        summary=summary,
        cross_lab=cross_lab,
    )
    LOG.info("Wrote %s", {k: str(v) for k, v in paths.items()})

    # Exit 0 even if unconstrained k-NN is inferior — gated non-inferiority is primary.
    if not summary.get("meets_query_target"):
        LOG.warning("Independent query target not met (n=%s)", summary.get("n_queries"))
    if not summary.get("primary_noninferior"):
        LOG.warning("Primary gated policy is NOT non-inferior under margin=%s", args.margin)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
