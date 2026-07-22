"""Run ONE external SOTA method for a SUBSET of slices (all seeds), with per-cell checkpoints.

Usage:
    HISTOWEAVE_LOCAL_DATA=/workspace/Histoweave \
    HISTOWEAVE_<METHOD>_PYTHON=/path/to/venv/bin/python \
    python run_method_slices.py <method> <checkpoint_dir> <sid1,sid2,...>

Same idempotent per-cell checkpoint behavior as run_one_method.py, but only
iterates the comma-separated slice list passed as argv[3]. Lets two machines
split one slow method (e.g. GraphST) across disjoint slice sets.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sota_runner import checkpoint_metadata, run_sota_cell  # noqa: E402

SEEDS = [42, 1, 2]
N_DOMAINS = {"151673": 7, "151674": 7, "151507": 7, "151669": 8, "151670": 5}


def main() -> None:
    method = sys.argv[1]
    ckpt_dir = Path(sys.argv[2])
    slices = [s.strip() for s in sys.argv[3].split(",") if s.strip()]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n = len(slices) * len(SEEDS)
    i = 0
    for sid in slices:
        for seed in SEEDS:
            i += 1
            ari, secs = run_sota_cell(
                method,
                sid,
                seed,
                N_DOMAINS[sid],
                benchmark_dir=HERE,
                checkpoint_dir=ckpt_dir,
            )
            status, err = checkpoint_metadata(method, sid, seed, checkpoint_dir=ckpt_dir)
            elapsed = time.time() - t_start
            _log(
                f"[{method}:{','.join(slices)}] {i}/{n} sid={sid} seed={seed} -> "
                f"ARI={ari if ari == ari else 'nan'} status={status} "
                f"cell={secs:.0f}s wall={elapsed / 60:.1f}min "
                f"{'ERR=' + str(err) if err else ''}"
            )
    total_min = round((time.time() - t_start) / 60, 1)
    _log(f"\n[{method}:{','.join(slices)}] DONE total={total_min}min")


if __name__ == "__main__":
    main()
