"""Swap official Xenium counts → re-run discovery → reassess panel Δ & AUROC.

Workflow
--------
1. Snapshot current ``results/slice_summary.json`` + ``components_panel.csv``
   as *before* (usually synthetic expression).
2. Rebuild bundle preferring official matrix (download if needed).
3. Re-run ``run_discovery_ln.py``.
4. Optionally re-run ``analyze_gc_components.py``.
5. Write ``results/OFFICIAL_SWAP_COMPARISON.md`` + ``swap_comparison.json``.

Usage
-----
::

    python research/discovery_xenium_lymph/swap_and_rerun.py
    python research/discovery_xenium_lymph/swap_and_rerun.py --skip-gc
    python research/discovery_xenium_lymph/swap_and_rerun.py --force-synthetic  # control arm
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "results"

logger = logging.getLogger("swap_and_rerun")


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _run(script: Path, extra: list[str] | None = None) -> int:
    cmd = [sys.executable, str(script), *(extra or [])]
    logger.info("RUN %s", " ".join(cmd))
    return int(subprocess.run(cmd, cwd=str(ROOT), check=False).returncode)


def _load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(tag: str) -> Path:
    snap = OUT / f"_snapshot_{tag}"
    snap.mkdir(parents=True, exist_ok=True)
    for name in ("slice_summary.json", "components_panel.csv", "LYMPH_DISCOVERY_REPORT.md"):
        src = OUT / name
        if src.is_file():
            shutil.copy2(src, snap / name)
    return snap


def _panel_metrics(comp_csv: Path) -> dict[str, Any]:
    if not comp_csv.is_file():
        return {}
    df = pd.read_csv(comp_csv)
    out: dict[str, Any] = {
        "n_components": int(len(df)),
        "n_direction_ok": int(
            df["direction_ok"].astype(str).str.lower().isin(["true", "1", "yes"]).sum()
        )
        if "direction_ok" in df.columns
        else None,
    }
    for col, key in (
        ("Germinal_center_delta_rest", "gc_delta_rest_mean"),
        ("B_follicle_delta_rest", "b_delta_rest_mean"),
        ("T_zone_delta_rest", "t_delta_rest_mean"),
        ("Germinal_center_delta_same_layer", "gc_delta_same_mean"),
        ("B_follicle_delta_same_layer", "b_delta_same_mean"),
        ("T_zone_delta_same_layer", "t_delta_same_mean"),
    ):
        if col in df.columns:
            out[key] = float(pd.to_numeric(df[col], errors="coerce").mean())
            out[key.replace("_mean", "_max")] = float(pd.to_numeric(df[col], errors="coerce").max())
    if "n" in df.columns:
        out["largest_n"] = int(pd.to_numeric(df["n"], errors="coerce").max())
    return out


def _compare(before: dict, after: dict, before_panel: dict, after_panel: dict) -> str:
    lines = [
        "# Official matrix swap — panel Δ & AUROC reassessment",
        "",
        "## Expression provenance",
        "",
        "| Arm | expression_source | n_obs |",
        "|-----|-------------------|------:|",
        f"| Before | `{before.get('expression_source', 'n/a')}` | {before.get('n_obs', '—')} |",
        f"| After  | `{after.get('expression_source', 'n/a')}` | {after.get('n_obs', '—')} |",
        "",
        "## Geometry / uncertainty",
        "",
        "| Metric | Before | After | Δ |",
        "|--------|-------:|------:|--:|",
    ]
    for key, label in (
        ("auroc_known_boundary", "AUROC(U → pathology boundary)"),
        ("high_u_n", "high-U cells"),
        ("cryptic_n", "cryptic cells"),
        ("cryptic_fraction_of_high_u", "cryptic / high-U"),
        ("estimated_k", "estimated K"),
        ("ensemble_k", "ensemble K"),
    ):
        b, a = before.get(key), after.get(key)
        try:
            delta = float(a) - float(b) if b is not None and a is not None else float("nan")
            lines.append(f"| {label} | {b} | {a} | {delta:+.4g} |")
        except (TypeError, ValueError):
            lines.append(f"| {label} | {b} | {a} | — |")

    lines += [
        "",
        "## Panel Δ (component means / max)",
        "",
        "| Metric | Before | After | Δ |",
        "|--------|-------:|------:|--:|",
    ]
    keys = [
        "n_components",
        "n_direction_ok",
        "largest_n",
        "gc_delta_rest_mean",
        "gc_delta_rest_max",
        "b_delta_rest_mean",
        "t_delta_rest_mean",
        "gc_delta_same_mean",
        "b_delta_same_mean",
        "t_delta_same_mean",
    ]
    for key in keys:
        b, a = before_panel.get(key), after_panel.get(key)
        try:
            if b is None or a is None:
                lines.append(f"| {key} | {b} | {a} | — |")
            else:
                delta = float(a) - float(b)
                lines.append(f"| {key} | {b} | {a} | {delta:+.4g} |")
        except (TypeError, ValueError):
            lines.append(f"| {key} | {b} | {a} | — |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- **AUROC rise** toward pathology boundaries after official counts → "
        "uncertainty better tracks real molecular/spatial transitions.",
        "- **Panel Δ rest stable/positive** under official counts → lymphoid programs "
        "are not synthetic artefacts.",
        "- **same-domain Δ near 0 / hard fails** can persist (DLPFC L3 parallel): "
        "intra-domain cryptic niches remain the harder claim.",
        "- If after still has synthetic `expression_source`, download failed; "
        "see prepare_bundle logs.",
        "",
        f"Snapshots: `{(OUT / '_snapshot_before').as_posix()}` → "
        f"`{(OUT / '_snapshot_after').as_posix()}`",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-gc", action="store_true", help="Skip GC deep-dive after discovery")
    parser.add_argument("--max-cells", type=int, default=15000)
    parser.add_argument("--official-matrix", type=Path, default=None)
    parser.add_argument("--official-cells", type=Path, default=None)
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    before_sum = _load_summary(OUT / "slice_summary.json")
    before_panel = _panel_metrics(OUT / "components_panel.csv")
    _snapshot("before")

    prep_extra = ["--max-cells", str(args.max_cells)]
    if args.force_synthetic:
        prep_extra.append("--force-synthetic")
    if args.no_download:
        prep_extra.append("--no-download")
    if args.official_matrix:
        prep_extra.extend(["--official-matrix", str(args.official_matrix)])
    if args.official_cells:
        prep_extra.extend(["--official-cells", str(args.official_cells)])

    rc = _run(BASE / "prepare_bundle.py", prep_extra)
    if rc != 0:
        logger.error("prepare_bundle failed rc=%s", rc)
        return rc

    rc = _run(BASE / "run_discovery_ln.py")
    if rc != 0:
        logger.error("discovery failed rc=%s", rc)
        return rc

    after_sum = _load_summary(OUT / "slice_summary.json")
    after_panel = _panel_metrics(OUT / "components_panel.csv")
    _snapshot("after")

    comparison = {
        "before_summary": before_sum,
        "after_summary": after_sum,
        "before_panel": before_panel,
        "after_panel": after_panel,
    }
    (OUT / "swap_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    report = _compare(before_sum, after_sum, before_panel, after_panel)
    (OUT / "OFFICIAL_SWAP_COMPARISON.md").write_text(report, encoding="utf-8")
    (BASE / "OFFICIAL_SWAP_COMPARISON.md").write_text(report, encoding="utf-8")
    logger.info("Wrote %s", OUT / "OFFICIAL_SWAP_COMPARISON.md")

    if not args.skip_gc:
        rc_gc = _run(BASE / "analyze_gc_components.py")
        if rc_gc != 0:
            logger.warning("GC deep-dive returned %s", rc_gc)
            return rc_gc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
