"""Assemble real STAGATE/GraphST/BayesSpace results into the DLPFC landscape.

Steps (all reproducible, no hand-editing of numbers):
  1. Read per-cell SOTA checkpoints (checkpoints/{method}__{sid}__seed{seed}.json)
     written by sota_runner, for methods stagate/graphst/bayesspace.
  2. Emit a long CSV `sota_benchmark_long.csv` (dataset,method,seed,ari,seconds,status,error).
  3. Load the committed landscape (figure3_results/landscape_dlpfc_merged.json) which
     already holds the validated spagcn(0.317)/banksy_py(0.223)/15 sklearn@sw methods,
     build a SOTA-only landscape from the CSV (seed-averaged per slice), and MERGE
     (prefer_later overlays the 3 new methods without perturbing existing ones).
  4. Write the merged landscape back + a sha256 manifest and a plain per-method
     mean table (sota_method_means.csv) for the README pointers.

Usage:
    python build_sota_and_merge.py <checkpoint_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from histoweave.benchmark.landscape_io import (  # noqa: E402
    attach_dataset_meta,
    dlpfc_slice_name_map,
    landscape_from_long_csv,
    merge_landscapes,
    meta_from_registry,
    write_landscape_json,
)
from histoweave.benchmark.recommend import _load_knowledge_base  # noqa: E402
from histoweave.benchmark.task_contract import AnalysisTask  # noqa: E402

SLICES = ["151673", "151674", "151507", "151669", "151670"]
SEEDS = [42, 1, 2]
SOTA_METHODS = ["stagate", "graphst", "bayesspace"]
LANDSCAPE_JSON = ROOT / "figure3_results" / "landscape_dlpfc_merged.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def gather_rows(ckpt_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for method in SOTA_METHODS:
        for sid in SLICES:
            for seed in SEEDS:
                p = ckpt_dir / f"{method}__{sid}__seed{seed}.json"
                if not p.exists():
                    rows.append(
                        {
                            "dataset": sid,
                            "method": method,
                            "seed": seed,
                            "ari": "",
                            "seconds": "",
                            "status": "missing",
                            "error": "no checkpoint",
                        }
                    )
                    continue
                d = json.loads(p.read_text())
                ari = d.get("ari")
                err = d.get("error")
                status = "error" if (err or ari is None) else "success"
                rows.append(
                    {
                        "dataset": sid,
                        "method": method,
                        "seed": seed,
                        "ari": "" if ari is None else ari,
                        "seconds": d.get("seconds", ""),
                        "status": status,
                        "error": err or "",
                    }
                )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = gather_rows(args.ckpt_dir)

    # ---- 2. write long CSV --------------------------------------------------
    sota_csv = HERE / "sota_benchmark_long.csv"
    fields = ["dataset", "method", "seed", "ari", "seconds", "status", "error"]
    with sota_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    _log(f"[write] {sota_csv} ({len(rows)} rows)")

    # ---- per-method mean table (README pointers) ----------------------------
    means = {}
    for method in SOTA_METHODS:
        per_slice = {}
        for sid in SLICES:
            vals = [
                r["ari"]
                for r in rows
                if r["method"] == method
                and r["dataset"] == sid
                and r["status"] == "success"
                and r["ari"] != ""
            ]
            per_slice[sid] = float(np.mean(vals)) if vals else None
        ok = [v for v in per_slice.values() if v is not None]
        means[method] = {
            "per_slice_mean": per_slice,
            "grand_mean": float(np.mean(ok)) if ok else None,
            "n_cells_success": sum(
                1 for r in rows if r["method"] == method and r["status"] == "success"
            ),
            "n_cells_total": len(SLICES) * len(SEEDS),
        }
    means_csv = HERE / "sota_method_means.csv"
    with means_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", *SLICES, "grand_mean", "n_success", "n_total"])
        for m in SOTA_METHODS:
            mm = means[m]
            w.writerow(
                [
                    m,
                    *[
                        (
                            ""
                            if mm["per_slice_mean"][s] is None
                            else round(mm["per_slice_mean"][s], 4)
                        )
                        for s in SLICES
                    ],
                    "" if mm["grand_mean"] is None else round(mm["grand_mean"], 4),
                    mm["n_cells_success"],
                    mm["n_cells_total"],
                ]
            )
    _log(f"[write] {means_csv}")
    for m in SOTA_METHODS:
        ps = {
            k: (round(v, 3) if v is not None else None)
            for k, v in means[m]["per_slice_mean"].items()
        }
        _log(
            f"    {m:>11s}: grand_mean={means[m]['grand_mean']}  "
            f"success={means[m]['n_cells_success']}/{means[m]['n_cells_total']}  per_slice={ps}"
        )

    if args.dry_run:
        _log("[dry-run] not touching landscape JSON")
        return

    # ---- 3. load committed landscape + build SOTA landscape + merge ---------
    if not LANDSCAPE_JSON.exists():
        raise FileNotFoundError(LANDSCAPE_JSON)
    base = _load_knowledge_base(LANDSCAPE_JSON)
    _log(f"[load] existing landscape: {base.method_count} methods x {base.dataset_count} datasets")

    has_finite = any(r["status"] == "success" and r["ari"] != "" for r in rows)
    if not has_finite:
        _log("[warn] no finite SOTA cells; landscape left unchanged")
        return

    sota = landscape_from_long_csv(
        sota_csv,
        task=AnalysisTask.SPATIAL_DOMAIN,
        prefer_config_as_method=False,
        method_col="method",
    )
    attach_dataset_meta(
        sota,
        meta_from_registry(sota.performance.keys(), name_map=dlpfc_slice_name_map()),
        overwrite=True,
    )
    merged = merge_landscapes(base, sota, task=AnalysisTask.SPATIAL_DOMAIN.value)
    all_methods = sorted({m for row in merged.performance.values() for m in row})
    _log(
        f"[merge] merged landscape: {merged.method_count} methods x {merged.dataset_count} datasets"
    )
    _log(f"[merge] methods now: {all_methods}")

    write_landscape_json(merged, LANDSCAPE_JSON)
    _log(f"[write] {LANDSCAPE_JSON}")

    # ---- 3b. full performance matrix (slices x ALL methods) for the heatmap --
    # The committed performance_matrix_mean.csv is sklearn-only; the heatmap
    # needs every method (sklearn@sw + spagcn + banksy_py + the 3 SOTA). Extract
    # the complete matrix straight from the merged landscape so nothing is hand
    # typed. Rows = biological slice order, cols = methods sorted by mean ARI.
    perf = merged.performance  # perf[dataset][method] -> ari
    all_cols = sorted({m for row in perf.values() for m in row})
    slice_order = [s for s in SLICES if s in perf] + [s for s in perf if s not in SLICES]
    full_rows = []
    for sid in slice_order:
        full_rows.append({"dataset": sid, **{m: perf[sid].get(m, "") for m in all_cols}})
    full_csv = HERE / "performance_matrix_mean_full.csv"
    with full_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", *all_cols])
        w.writeheader()
        w.writerows(full_rows)
    n_methods = len(all_cols)
    _log(f"[write] {full_csv} ({len(slice_order)} slices x {n_methods} methods)")

    # ---- 4. manifest --------------------------------------------------------
    manifest = {
        "protocol": "histoweave.landscape.dlpfc_spatial_sota.v2",
        "sota_methods_added": SOTA_METHODS,
        "sota_method_means": means,
        "artifacts": [],
    }
    for p in [sota_csv, means_csv, LANDSCAPE_JSON, full_csv]:
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = str(p)
        manifest["artifacts"].append({"path": rel, "sha256": _sha(p), "bytes": p.stat().st_size})
    man_path = HERE / "sota_merge_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    _log(f"[write] {man_path}")


if __name__ == "__main__":
    main()
