"""Run all (slice x seed) cells for ONE external SOTA method with per-cell checkpoints.

Usage:
    HISTOWEAVE_LOCAL_DATA=/workspace/Histoweave \
    HISTOWEAVE_<METHOD>_PYTHON=/path/to/venv/bin/python \
    python run_one_method.py <method> <checkpoint_dir>

Reuses the repo's isolated-process runner (sota_runner.run_sota_cell), which
writes checkpoints/{method}__{sid}__seed{seed}.json with ari/seconds/status/error.
Idempotent: existing checkpoints are skipped.
"""

from __future__ import annotations

import json
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

SLICES = ["151673", "151674", "151507", "151669", "151670"]
SEEDS = [42, 1, 2]
# n_domains per slice (from bundle domain_truth.nunique(), verified)
N_DOMAINS = {"151673": 7, "151674": 7, "151507": 7, "151669": 8, "151670": 5}


def main() -> None:
    method = sys.argv[1]
    ckpt_dir = Path(sys.argv[2])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rows = []
    n = len(SLICES) * len(SEEDS)
    i = 0
    for sid in SLICES:
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
            rows.append(
                {
                    "method": method,
                    "sid": sid,
                    "seed": seed,
                    "ari": ari,
                    "seconds": secs,
                    "status": status,
                    "error": err,
                }
            )
            elapsed = time.time() - t_start
            _log(
                f"[{method}] {i}/{n} sid={sid} seed={seed} -> "
                f"ARI={ari if ari == ari else 'nan'} status={status} "
                f"cell={secs:.0f}s wall={elapsed / 60:.1f}min "
                f"{'ERR=' + str(err) if err else ''}"
            )
    ok = sum(1 for r in rows if r["status"] == "success")
    aris = [r["ari"] for r in rows if r["status"] == "success" and r["ari"] == r["ari"]]
    summary = {
        "method": method,
        "n_cells": n,
        "n_success": ok,
        "mean_ari": (sum(aris) / len(aris)) if aris else None,
        "total_min": round((time.time() - t_start) / 60, 1),
        "rows": rows,
    }
    (ckpt_dir / f"_summary_{method}.json").write_text(json.dumps(summary, indent=2))
    _log(
        f"\n[{method}] DONE: {ok}/{n} success, mean_ari="
        f"{summary['mean_ari']}, total={summary['total_min']}min"
    )


if __name__ == "__main__":
    main()
