"""Merge GraphST/STAGATE checkpoint ARIs into the official real-ARI JSON + CSV."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("merge_real_ari_checkpoints")

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "5x15_spatial_aware" / "checkpoints"
OUT = Path(__file__).resolve().parent / "results" / "graphst_stagate_real_ari.json"
CSV = ROOT / "5x15_spatial_aware" / "sota_benchmark_long.csv"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rows: list[dict] = []
    for p in sorted(CKPT.glob("sota_graphst__*.json")) + sorted(CKPT.glob("sota_stagate__*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        parts = p.stem.split("__")
        d.setdefault("method", parts[0].replace("sota_", ""))
        d.setdefault("dataset", parts[1])
        d.setdefault("seed", int(parts[2].replace("seed", "")))
        if d.get("status") != "success" or d.get("ari") is None:
            continue
        d["backend"] = d.get("backend") or f"official_{d['method']}"
        d["family"] = "sota"
        d["config"] = d["method"]
        d["oracle_k"] = True
        rows.append(d)

    def summarize(method: str) -> dict:
        ok = [r for r in rows if r["method"] == method]
        per: dict[str, list[float]] = {}
        for r in ok:
            per.setdefault(str(r["dataset"]), []).append(float(r["ari"]))
        per_mean = {k: float(np.mean(v)) for k, v in per.items()}
        return {
            "n_cells": len(ok),
            "n_success": len(ok),
            "mean_ari": float(np.mean([r["ari"] for r in ok])) if ok else None,
            "std_ari": float(np.std([r["ari"] for r in ok], ddof=0)) if ok else None,
            "per_dataset_mean_ari": per_mean,
            "mean_seconds": float(np.mean([float(r.get("seconds") or 0) for r in ok]))
            if ok
            else None,
            "statuses": ["success"] if ok else [],
            "errors": [],
        }

    payload = {
        "protocol": "histoweave.sota_dlpfc.v1",
        "backend_mode": "official_real",
        "methods": ["graphst", "stagate"],
        "slices": ["151673", "151674", "151507", "151669", "151670"],
        "seeds": [42, 1, 2],
        "max_obs": 1000,
        "graphst_epochs": 120,
        "stagate_epochs": 150,
        "rows": rows,
        "summary": {
            "graphst": summarize("graphst"),
            "stagate": summarize("stagate"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("%s", json.dumps(payload["summary"], indent=2))

    fieldnames = [
        "dataset",
        "method",
        "seed",
        "ari",
        "seconds",
        "status",
        "error",
        "n_domains_truth",
        "n_obs",
        "oracle_k",
        "family",
        "config",
    ]
    existing = list(csv.DictReader(CSV.open(encoding="utf-8"))) if CSV.is_file() else []
    keep = [r for r in existing if r.get("method") not in {"graphst", "stagate"}]
    for r in rows:
        keep.append({k: r.get(k, "") for k in fieldnames})
    with CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in keep:
            w.writerow(r)
    logger.info("wrote %s and %s (%s rows)", OUT, CSV, len(keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
