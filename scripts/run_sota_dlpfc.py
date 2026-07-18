#!/usr/bin/env python
"""Probe and/or run the official SOTA DLPFC reproduction grid (P2).

Examples
--------
# Probe only (no heavy training)
python scripts/run_sota_dlpfc.py --dry-run

# Run every available backend (skips missing installs)
python scripts/run_sota_dlpfc.py --methods banksy_py,spagcn

# Then merge into the recommender knowledge base
python scripts/build_merged_landscape.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.sota_pipeline import (  # noqa: E402
    env_contract,
    run_sota_benchmark,
)

LOG = logging.getLogger("histoweave.run_sota_dlpfc")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default="banksy_py,spagcn,graphst,stagate,bayesspace",
        help="Comma-separated method list",
    )
    parser.add_argument(
        "--slices",
        default="151673,151674,151507,151669,151670",
        help="Comma-separated DLPFC slice ids",
    )
    parser.add_argument("--seeds", default="42,1,2", help="Comma-separated seeds")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "5x15_spatial_aware",
        help="Directory for sota_benchmark_long.csv and throughput JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe backends and write a skipped status grid (no training)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore per-cell checkpoints and re-run",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Attempt methods even when probe fails (records explicit failures)",
    )
    parser.add_argument(
        "--show-contract",
        action="store_true",
        help="Print the environment contract JSON and exit",
    )
    args = parser.parse_args(argv)

    if args.show_contract:
        import json

        sys.stdout.write(json.dumps(env_contract(), indent=2) + "\n")
        return 0

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    report = run_sota_benchmark(
        methods=methods,
        slices=slices,
        seeds=seeds,
        repo_root=ROOT,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        force=args.force,
        only_available=not args.all_methods,
    )
    summary = report.throughput_summary()
    LOG.info(
        "SOTA grid done dry_run=%s success=%s failed=%s skipped=%s available=%s",
        summary["dry_run"],
        summary["n_success"],
        summary["n_failed"],
        summary["n_skipped"],
        summary["available_methods"],
    )
    LOG.info("artifacts under %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
