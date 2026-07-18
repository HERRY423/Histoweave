"""Driver: map the failure boundary of every runnable method on every axis.

Runs the full adversarial study (all three tasks, all sweep axes, N replicate
seeds), detects each method's acceptable->unacceptable boundary at ``tau``,
and writes the tidy long table plus the Safe Operating Cards (CSV / JSON / MD).

The study engine now lives in the installed package
(``histoweave.benchmark.failure_boundary``); this script is a thin, path-robust
wrapper so the exact study can be reproduced from a checkout. The same run is
available from the CLI:

    histoweave benchmark-boundary --seeds 5 --tau 0.7 --out <dir>

Usage
-----
    python run_failure_mapping.py --seeds 5 --tau 0.7 --out results

Backend-gated methods that cannot run in the current environment
(``banksy`` container, ``cell2location``, ``nnsvg``, ``spatialde``) are
auto-detected and recorded as "not evaluated (backend unavailable)" rather
than silently dropped.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from histoweave.benchmark.failure_boundary import (
    DEFAULT_TAU,
    run_boundary_study,
    write_study_outputs,
)


def _log(message: object) -> None:
    """Emit a progress line through standard logging (repo logging contract)."""
    logging.getLogger(__name__).info("%s", message)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5, help="number of replicate seeds")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="acceptability threshold")
    ap.add_argument("--out", type=str, default="results", help="output directory")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    here = Path(__file__).resolve().parent
    out = (here / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    result = run_boundary_study(tau=args.tau, n_seeds=args.seeds, progress=True)
    paths = write_study_outputs(result, out)

    cards = result.cards_dataframe()
    _log(f"[write] {paths['long_csv']}  ({len(result.long_rows)} rows)")
    _log(f"[write] {paths['cards_csv']}  ({len(cards)} cards)")
    _log(f"[write] {paths['cards_json']}")
    _log(f"[write] {paths['cards_md']}")
    counts = cards["verdict"].value_counts().to_dict()
    _log(f"verdicts: {counts}")
    _log("DONE.")


if __name__ == "__main__":
    main()
