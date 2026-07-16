"""Isolated-process runner for incompatible official SOTA backends."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_sota_cell(
    method: str,
    sid: str,
    seed: int,
    n_domains: int,
    *,
    benchmark_dir: Path,
    checkpoint_dir: Path,
) -> tuple[float, float]:
    """Run one external backend with its method-specific Python interpreter."""
    ckpt = checkpoint_dir / f"{method}__{sid}__seed{seed}.json"
    if ckpt.exists():
        cached = json.loads(ckpt.read_text())
        ari = cached.get("ari")
        return (float(ari) if ari is not None else float("nan")), cached["seconds"]

    python = os.environ.get(f"HISTOWEAVE_{method.upper()}_PYTHON", sys.executable)
    src = benchmark_dir.parent / "src"
    script = (
        "import sys, json, time, numpy as np\n"
        f"sys.path.insert(0, {json.dumps(str(benchmark_dir))})\n"
        f"sys.path.insert(0, {json.dumps(str(src))})\n"
        "from experiment_5x15_methods import load_slice, _adapter_labels\n"
        "from sklearn.metrics import adjusted_rand_score\n"
        f"sid={json.dumps(sid)}; method={json.dumps(method)}; seed={seed}; "
        f"n_domains={n_domains}; ckpt_path={json.dumps(str(ckpt))}\n"
        "tab, _ = load_slice(sid); t0 = time.time()\n"
        "try:\n"
        "    labels = _adapter_labels(method, tab, seed, n_domains)\n"
        "    truth = tab.obs['domain_truth'].astype(str).values\n"
        "    ari = float(adjusted_rand_score(truth, labels))\n"
        "    payload = {'ari': ari if np.isfinite(ari) else None, "
        "'seconds': time.time() - t0}\n"
        "except Exception as exc:\n"
        "    payload = {'ari': None, 'seconds': time.time() - t0, "
        "'error': str(exc)[:400]}\n"
        "with open(ckpt_path, 'w') as f: json.dump(payload, f)\n"
        "sys.stdout.write(json.dumps(payload) + '\\n')\n"
    )
    started = time.time()
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(src), str(benchmark_dir))),
        "HISTOWEAVE_LOCAL_DATA": os.environ.get(
            "HISTOWEAVE_LOCAL_DATA", str(benchmark_dir.parent)
        ),
    }
    try:
        result = subprocess.run(
            [python, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("HISTOWEAVE_SOTA_TIMEOUT", "7200")),
            env=env,
        )
        if result.returncode != 0 or not ckpt.exists():
            error = (result.stderr or "no output").strip().splitlines()[-1]
            ckpt.write_text(
                json.dumps(
                    {"ari": None, "seconds": time.time() - started, "error": error[:400]}
                )
            )
    except subprocess.TimeoutExpired:
        ckpt.write_text(
            json.dumps({"ari": None, "seconds": time.time() - started, "error": "timeout"})
        )
    payload = json.loads(ckpt.read_text())
    ari = payload.get("ari")
    return (float(ari) if ari is not None else float("nan")), payload["seconds"]


def checkpoint_metadata(
    method: str, sid: str, seed: int, *, checkpoint_dir: Path
) -> tuple[str, str | None]:
    """Return the explicit success/failure fields for one completed cell."""
    payload = json.loads(
        (checkpoint_dir / f"{method}__{sid}__seed{seed}.json").read_text()
    )
    error = payload.get("error")
    return ("failed" if error or payload.get("ari") is None else "success"), error
