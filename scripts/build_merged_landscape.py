#!/usr/bin/env python
"""Build a task-contracted DLPFC landscape from on-disk benchmark CSVs.

Example
-------
python scripts/build_merged_landscape.py \\
    --out figure3_results/landscape_dlpfc_merged.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.landscape_io import (  # noqa: E402
    build_dlpfc_merged_landscape,
    validate_landscape_contracts,
    write_landscape_json,
)

LOG = logging.getLogger("histoweave.build_merged_landscape")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figure3_results" / "landscape_dlpfc_merged.json",
        help="Output knowledge-base JSON path",
    )
    parser.add_argument("--baseline-csv", type=Path, default=None)
    parser.add_argument("--sota-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    landscape = build_dlpfc_merged_landscape(
        baseline_csv=args.baseline_csv,
        sota_csv=args.sota_csv,
        repo_root=ROOT,
    )
    problems = validate_landscape_contracts(landscape)
    for problem in problems:
        LOG.warning("contract: %s", problem)

    path = write_landscape_json(landscape, args.out)
    LOG.info(
        "Wrote %s (%s datasets x %s methods, task=%s)",
        path,
        landscape.dataset_count,
        landscape.method_count,
        landscape.task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
