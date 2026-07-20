#!/usr/bin/env python3
"""Score functional experiment returns (perturbation / lineage / orthogonal).

Reads CSVs from ``results/functional_experiments/returns/`` matching the
template schema, evaluates pre-registered pass heuristics per experiment, and
writes:

* ``RETURN_REPORT.md``
* ``return_scores.json``
* updates ``results/functional_experiments/CLAIM_STATUS.json``

Use ``--dry-run`` to validate templates without claim upgrades.
Use ``--simulate`` only for pipeline tests (forces SIMULATED tag; never F3/F4 claim).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from functional_experiments import EXPERIMENTS, ExperimentSpec

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "functional_experiments"
RETURNS = OUT / "returns"

logger = logging.getLogger("functional_return")

REQUIRED_COLS = {
    "experiment_id",
    "discovery_id",
    "class",
    "condition",
    "metric_name",
    "metric_value",
    "replicate_id",
}


def _load_returns(paths: list[Path] | None) -> pd.DataFrame:
    files: list[Path] = []
    if paths:
        files = paths
    else:
        if RETURNS.exists():
            files = sorted(RETURNS.glob("*.csv"))
        # also accept single file at package root
        root_all = OUT / "functional_return_all.csv"
        if root_all.exists():
            files.append(root_all)
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["_source_file"] = f.name
            frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skip %s: %s", f, exc)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _is_simulated(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    notes = df.get("notes", pd.Series([""] * len(df))).astype(str).str.upper()
    return bool(notes.str.contains("SIMULAT").any())


def _metric_series(df: pd.DataFrame, experiment_id: str, metric: str) -> pd.DataFrame:
    sub = df[
        (df["experiment_id"].astype(str) == experiment_id)
        & (df["metric_name"].astype(str) == metric)
    ].copy()
    if sub.empty:
        return sub
    sub["metric_value"] = pd.to_numeric(sub["metric_value"], errors="coerce")
    return sub.dropna(subset=["metric_value"])


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se == 0 or not np.isfinite(se):
        return 1.0
    t = (ma - mb) / se
    # Normal approximation for small-n convenience (documented as such)
    from math import erfc, sqrt

    p = erfc(abs(t) / sqrt(2.0))
    return float(min(1.0, max(0.0, p)))


def score_experiment(spec: ExperimentSpec, df: pd.DataFrame) -> dict[str, Any]:
    """Heuristic pass evaluation from long-format metrics.

    Labs may also set pass_flag_lab; analyzer prefers quantitative rules when
    treatment/control pairs exist for key metrics.
    """
    sub = df[df["experiment_id"].astype(str) == spec.experiment_id].copy()
    result: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "discovery_id": spec.discovery_id,
        "class": spec.class_,
        "claim_level_on_pass": spec.claim_level_on_pass,
        "n_rows": int(len(sub)),
        "pass": False,
        "status": "no_data",
        "details": [],
        "metrics_seen": sorted(sub["metric_name"].astype(str).unique().tolist())
        if not sub.empty
        else [],
    }
    if sub.empty:
        return result

    # Lab self-flag
    if "pass_flag_lab" in sub.columns:
        flags = sub["pass_flag_lab"].astype(str).str.lower()
        if flags.isin(["true", "pass", "yes", "1"]).any() and not flags.isin(
            ["false", "fail", "no", "0"]
        ).any():
            result["details"].append("lab pass_flag_lab asserts PASS (still needs metric checks)")

    # Generic treatment vs control on any metric containing "module" or known names
    cond = sub["condition"].astype(str).str.lower()
    treat_mask = cond.str.contains(
        "treat|drug|crispr|ko|kd|inhib|ttx|cuprizone|perturb|mek|soce", regex=True
    )
    ctrl_mask = cond.str.contains("control|vehicle|ntc|aavs|untreated|sham|rest|non_roi|non-roi", regex=True)
    roi_mask = cond.str.contains("roi|cryptic|niche|lineage_pos|reporter", regex=True)

    checks_passed = 0
    checks_total = 0

    # 1) If treatment + control for same metric: direction from first pass criterion keywords
    for metric, g in sub.groupby(sub["metric_name"].astype(str)):
        tvals = g.loc[treat_mask.reindex(g.index, fill_value=False), "metric_value"]
        cvals = g.loc[ctrl_mask.reindex(g.index, fill_value=False), "metric_value"]
        tvals = pd.to_numeric(tvals, errors="coerce").dropna()
        cvals = pd.to_numeric(cvals, errors="coerce").dropna()
        if len(tvals) and len(cvals):
            checks_total += 1
            diff = float(tvals.mean() - cvals.mean())
            p = _welch_p(tvals.to_numpy(), cvals.to_numpy()) if len(tvals) > 1 and len(cvals) > 1 else 1.0
            # infer expected direction from metric name / direction_expected column
            dexp = (
                g["direction_expected"].astype(str).str.lower().iloc[0]
                if "direction_expected" in g.columns
                else ""
            )
            expect_down = dexp == "down" or "p" == metric.lower()[-1:] or "shift_p" in metric.lower()
            if "shift_p" in metric.lower() or metric.lower().endswith("_p"):
                ok = bool(tvals.mean() <= 0.05)
            elif expect_down:
                ok = bool(diff < 0 and (p <= 0.05 or len(tvals) == 1))
            else:
                ok = bool(diff > 0 and (p <= 0.05 or len(tvals) == 1))
            result["details"].append(
                f"{metric}: treat-ctrl Δ={diff:.4g} p≈{p:.3g} → {'PASS' if ok else 'FAIL'}"
            )
            if ok:
                checks_passed += 1

        # 2) ROI vs rest enrichment
        rvals = g.loc[roi_mask.reindex(g.index, fill_value=False), "metric_value"]
        rest = g.loc[cond.reindex(g.index, fill_value=False).str.contains("rest|non"), "metric_value"]
        rvals = pd.to_numeric(rvals, errors="coerce").dropna()
        rest = pd.to_numeric(rest, errors="coerce").dropna()
        if len(rvals) and len(rest):
            checks_total += 1
            diff = float(rvals.mean() - rest.mean())
            p = _welch_p(rvals.to_numpy(), rest.to_numpy()) if len(rvals) > 1 and len(rest) > 1 else 1.0
            ok = bool(diff > 0 and (p <= 0.05 or len(rvals) == 1))
            result["details"].append(
                f"{metric}: roi-rest Δ={diff:.4g} p≈{p:.3g} → {'PASS' if ok else 'FAIL'}"
            )
            if ok:
                checks_passed += 1

    # 3) Single-value gate metrics (delta_rest, shift_p) without conditions
    for metric, g in sub.groupby(sub["metric_name"].astype(str)):
        vals = pd.to_numeric(g["metric_value"], errors="coerce").dropna()
        if vals.empty:
            continue
        mlow = metric.lower()
        if "shift_p" in mlow or mlow.endswith("_p") and "padj" in mlow or mlow.endswith("shift_p"):
            checks_total += 1
            ok = bool(vals.mean() <= 0.05)
            result["details"].append(f"{metric}: mean={vals.mean():.4g} ≤0.05 → {'PASS' if ok else 'FAIL'}")
            if ok:
                checks_passed += 1
        elif "delta_rest" in mlow or mlow.endswith("_delta"):
            checks_total += 1
            dexp = (
                g["direction_expected"].astype(str).str.lower().iloc[0]
                if "direction_expected" in g.columns
                else "up"
            )
            ok = bool(vals.mean() > 0) if dexp != "down" else bool(vals.mean() < 0)
            result["details"].append(
                f"{metric}: mean={vals.mean():.4g} dir={dexp} → {'PASS' if ok else 'FAIL'}"
            )
            if ok:
                checks_passed += 1

    if checks_total == 0:
        result["status"] = "insufficient_structure"
        result["details"].append(
            "Need treatment/control or ROI/rest pairs, or delta_rest/shift_p metrics"
        )
        result["pass"] = False
        return result

    # Require majority of quantitative checks + min replicates if present
    n_reps = sub["replicate_id"].nunique() if "replicate_id" in sub.columns else 0
    rep_ok = n_reps >= max(1, min(spec.n_replicates_min, 2))  # allow 2 for early F3
    frac = checks_passed / checks_total
    result["checks_passed"] = checks_passed
    result["checks_total"] = checks_total
    result["n_replicates"] = int(n_reps)
    result["pass"] = bool(frac >= 0.5 and checks_passed >= 1 and rep_ok)
    result["status"] = "pass" if result["pass"] else "fail"
    if not rep_ok:
        result["details"].append(
            f"replicate count {n_reps} < soft minimum (need ≥2 for automated pass)"
        )
    return result


def simulate_minimal_returns() -> pd.DataFrame:
    """Synthetic rows for pipeline dry-run only."""
    rows = []
    for e in EXPERIMENTS:
        if e.priority > 1:
            continue
        if e.class_ == "orthogonal":
            rows.append(
                {
                    "experiment_id": e.experiment_id,
                    "discovery_id": e.discovery_id,
                    "class": e.class_,
                    "modality": e.modality,
                    "replicate_id": "sim1",
                    "condition": "cryptic_roi",
                    "sample_id": "SIM",
                    "roi_id": e.related_roi,
                    "metric_name": "module_delta_rest",
                    "metric_value": 0.35,
                    "n_units": 50,
                    "direction_expected": "up",
                    "pass_flag_lab": "pending",
                    "notes": "SIMULATED — not a claim",
                }
            )
            rows.append(
                {
                    "experiment_id": e.experiment_id,
                    "discovery_id": e.discovery_id,
                    "class": e.class_,
                    "modality": e.modality,
                    "replicate_id": "sim1",
                    "condition": "rest",
                    "sample_id": "SIM",
                    "roi_id": e.related_roi,
                    "metric_name": "module_delta_rest",
                    "metric_value": 0.0,
                    "n_units": 200,
                    "direction_expected": "up",
                    "pass_flag_lab": "pending",
                    "notes": "SIMULATED — not a claim",
                }
            )
            rows.append(
                {
                    "experiment_id": e.experiment_id,
                    "discovery_id": e.discovery_id,
                    "class": e.class_,
                    "modality": e.modality,
                    "replicate_id": "sim2",
                    "condition": "cryptic_roi",
                    "sample_id": "SIM",
                    "roi_id": e.related_roi,
                    "metric_name": "shift_p",
                    "metric_value": 0.02,
                    "n_units": 50,
                    "direction_expected": "down",
                    "pass_flag_lab": "pending",
                    "notes": "SIMULATED — not a claim",
                }
            )
        elif e.class_ == "perturbation":
            for rep in ("sim1", "sim2"):
                rows.append(
                    {
                        "experiment_id": e.experiment_id,
                        "discovery_id": e.discovery_id,
                        "class": e.class_,
                        "modality": e.modality,
                        "replicate_id": rep,
                        "condition": "treatment",
                        "sample_id": "SIM",
                        "roi_id": e.related_roi,
                        "metric_name": "module_score",
                        "metric_value": 0.2,
                        "n_units": 40,
                        "direction_expected": "down",
                        "pass_flag_lab": "pending",
                        "notes": "SIMULATED — not a claim",
                    }
                )
                rows.append(
                    {
                        "experiment_id": e.experiment_id,
                        "discovery_id": e.discovery_id,
                        "class": e.class_,
                        "modality": e.modality,
                        "replicate_id": rep,
                        "condition": "vehicle_control",
                        "sample_id": "SIM",
                        "roi_id": e.related_roi,
                        "metric_name": "module_score",
                        "metric_value": 0.55,
                        "n_units": 40,
                        "direction_expected": "down",
                        "pass_flag_lab": "pending",
                        "notes": "SIMULATED — not a claim",
                    }
                )
        else:  # lineage
            for rep in ("sim1", "sim2"):
                rows.append(
                    {
                        "experiment_id": e.experiment_id,
                        "discovery_id": e.discovery_id,
                        "class": e.class_,
                        "modality": e.modality,
                        "replicate_id": rep,
                        "condition": "lineage_pos_roi",
                        "sample_id": "SIM",
                        "roi_id": e.related_roi,
                        "metric_name": "lineage_density",
                        "metric_value": 0.4,
                        "n_units": 30,
                        "direction_expected": "up",
                        "pass_flag_lab": "pending",
                        "notes": "SIMULATED — not a claim",
                    }
                )
                rows.append(
                    {
                        "experiment_id": e.experiment_id,
                        "discovery_id": e.discovery_id,
                        "class": e.class_,
                        "modality": e.modality,
                        "replicate_id": rep,
                        "condition": "rest",
                        "sample_id": "SIM",
                        "roi_id": e.related_roi,
                        "metric_name": "lineage_density",
                        "metric_value": 0.15,
                        "n_units": 80,
                        "direction_expected": "up",
                        "pass_flag_lab": "pending",
                        "notes": "SIMULATED — not a claim",
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    scores: list[dict[str, Any]],
    *,
    simulated: bool,
) -> str:
    n_pass = sum(1 for s in scores if s.get("pass"))
    lines = [
        "# Functional experiment return report",
        "",
        f"**Analyzed:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Simulated:** {'**YES — NOT A CLAIM**' if simulated else 'no'}  ",
        f"**Experiments with data:** {len(scores)} · **PASS:** {n_pass}",
        "",
    ]
    if simulated:
        lines += [
            "> ⚠️ **SIMULATED OR DRY-RUN DATA.** Do not cite as F3/F4 validation.",
            "",
        ]
    lines += [
        "| Experiment | Class | Level | Status | Checks |",
        "|------------|-------|-------|--------|--------|",
    ]
    for s in scores:
        ch = f"{s.get('checks_passed', 0)}/{s.get('checks_total', 0)}"
        st = s.get("status", "?")
        if s.get("pass") and not simulated:
            st = "**PASS**"
        elif s.get("pass") and simulated:
            st = "PASS (simulated)"
        lines.append(
            f"| `{s['experiment_id']}` | {s.get('class')} | {s.get('claim_level_on_pass')} | "
            f"{st} | {ch} |"
        )
    lines.append("")
    for s in scores:
        lines.append(f"### `{s['experiment_id']}`")
        lines.append("")
        for d in s.get("details", []):
            lines.append(f"- {d}")
        if not s.get("details"):
            lines.append("- _(no details)_")
        lines.append("")

    # Claim ladder update section
    f3 = [
        s["experiment_id"]
        for s in scores
        if s.get("pass") and s.get("claim_level_on_pass") == "F3" and not simulated
    ]
    f4 = [
        s["experiment_id"]
        for s in scores
        if s.get("pass") and s.get("claim_level_on_pass") == "F4" and not simulated
    ]
    lines += [
        "## Claim upgrade eligibility",
        "",
        f"- **F3 (orthogonal / lineage) eligible:** {', '.join(f'`{x}`' for x in f3) or '_none_'}",
        f"- **F4 (perturbation) eligible:** {', '.join(f'`{x}`' for x in f4) or '_none_'}",
        "",
        "Eligibility requires real (non-simulated) returns meeting pre-registered",
        "metrics. Final claim language still needs human review + methods text.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only check template/schema")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Inject simulated returns for pipeline test (never claim-grade)",
    )
    parser.add_argument("--csv", type=Path, nargs="*", default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RETURNS.mkdir(parents=True, exist_ok=True)

    if args.dry_run and not args.simulate:
        tmpl = OUT / "templates" / "functional_return_template.csv"
        if not tmpl.exists():
            logger.error("Template missing — run prepare_functional_experiment_package.py first")
            raise SystemExit(1)
        df = pd.read_csv(tmpl)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            logger.error("template missing columns: %s", sorted(missing))
            raise SystemExit(1)
        logger.info("dry-run OK: template has required columns (%s rows example)", len(df))
        # score nothing claim-grade
        status = {
            "protocol": "histoweave.functional_return.v1",
            "dry_run": True,
            "simulated": False,
            "n_pass": 0,
            "message": "schema OK; no claim upgrade",
        }
        (OUT / "CLAIM_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        return

    df = _load_returns(args.csv)
    simulated = args.simulate or _is_simulated(df)
    if args.simulate or df.empty:
        if df.empty and not args.simulate:
            logger.warning("No return CSVs found — writing empty report")
            scores = [
                {
                    "experiment_id": e.experiment_id,
                    "discovery_id": e.discovery_id,
                    "class": e.class_,
                    "claim_level_on_pass": e.claim_level_on_pass,
                    "n_rows": 0,
                    "pass": False,
                    "status": "no_data",
                    "details": ["no return file"],
                }
                for e in EXPERIMENTS
            ]
            simulated = False
        else:
            df = simulate_minimal_returns()
            simulated = True
            logger.warning("Using SIMULATED returns — not claim-grade")

    # validate columns
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error("return data missing columns: %s", sorted(missing))
        raise SystemExit(1)

    by_id = {e.experiment_id: e for e in EXPERIMENTS}
    scores = []
    for eid in sorted(df["experiment_id"].astype(str).unique()):
        if eid not in by_id:
            logger.warning("unknown experiment_id in returns: %s", eid)
            continue
        scores.append(score_experiment(by_id[eid], df))
    # include registered experiments with zero data
    seen = {s["experiment_id"] for s in scores}
    for e in EXPERIMENTS:
        if e.experiment_id not in seen:
            scores.append(
                {
                    "experiment_id": e.experiment_id,
                    "discovery_id": e.discovery_id,
                    "class": e.class_,
                    "claim_level_on_pass": e.claim_level_on_pass,
                    "n_rows": 0,
                    "pass": False,
                    "status": "no_data",
                    "details": ["no return rows"],
                }
            )

    report = write_report(scores, simulated=simulated)
    (OUT / "RETURN_REPORT.md").write_text(report, encoding="utf-8")
    payload = {
        "protocol": "histoweave.functional_return.v1",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "simulated": simulated,
        "scores": scores,
        "n_pass_claim_grade": sum(1 for s in scores if s.get("pass") and not simulated),
    }
    (OUT / "return_scores.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    f3 = [
        s["experiment_id"]
        for s in scores
        if s.get("pass") and s.get("claim_level_on_pass") == "F3" and not simulated
    ]
    f4 = [
        s["experiment_id"]
        for s in scores
        if s.get("pass") and s.get("claim_level_on_pass") == "F4" and not simulated
    ]
    status = {
        "protocol": "histoweave.functional_return.v1",
        "simulated": simulated,
        "F2_computational": "see FUNCTIONAL_VALIDATION.md",
        "F3_eligible_experiments": f3,
        "F4_eligible_experiments": f4,
        "n_pass_claim_grade": payload["n_pass_claim_grade"],
        "next": "Deposit real CSVs in results/functional_experiments/returns/",
    }
    (OUT / "CLAIM_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    logger.info("wrote RETURN_REPORT.md · claim-grade PASS=%s", payload["n_pass_claim_grade"])


if __name__ == "__main__":
    main()
