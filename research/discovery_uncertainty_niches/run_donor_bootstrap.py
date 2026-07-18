"""Donor-stratified bootstrap CI for direction_ok L3 cryptic components.

Reads ``results/cohort/cohort_component_panel.csv`` (from run_cohort_panel.py)
and writes CI tables + a short markdown report.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.donor_bootstrap import (  # noqa: E402
    donor_stratified_bootstrap_l3,
    load_cohort_panel,
)

logger = logging.getLogger("donor_bootstrap")
BASE = Path(__file__).resolve().parent
DEFAULT_CSV = BASE / "results" / "cohort" / "cohort_component_panel.csv"
OUT = BASE / "results" / "cohort"


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def write_report(result) -> str:
    p = result.point
    ci = result.ci
    lines = [
        "# Donor-stratified bootstrap CI — L3 direction_ok components",
        "",
        f"**Protocol:** `{result.protocol}`  ·  **n_boot:** {result.n_boot}  ·  "
        f"**components:** {result.n_components}  ·  **donors:** {result.n_donors} "
        f"({', '.join(result.donors)})",
        "",
        f"**Filter:** `{result.filter}`",
        "",
        "## Point estimates (donor-equal weight)",
        "",
        f"| Metric | Point | {int(result.ci_level * 100)}% CI (donor-stratified) |",
        "|--------|------:|----:|",
        f"| L3 Δ vs rest | {p['l3_delta_rest']:.4f} | "
        f"[{ci['l3_delta_rest']['ci_low']:.4f}, {ci['l3_delta_rest']['ci_high']:.4f}] |",
        f"| Myelin Δ vs rest | {p['myelin_delta_rest']:.4f} | "
        f"[{ci['myelin_delta_rest']['ci_low']:.4f}, {ci['myelin_delta_rest']['ci_high']:.4f}] |",
        f"| Direction rate | {p['direction_rate']:.3f} | "
        f"[{ci['direction_rate']['ci_low']:.3f}, {ci['direction_rate']['ci_high']:.3f}] |",
        "",
        "## Per-donor means (observed)",
        "",
        "| Donor | n_comp | n_spots | L3 Δrest | Myelin Δrest | dir rate |",
        "|-------|-------:|-------:|---------:|-------------:|---------:|",
    ]
    for donor, m in result.donor_means.items():
        lines.append(
            f"| {donor} | {m['n_components']} | {m['n_spots']} | "
            f"{m['l3_delta_rest']:.4f} | {m['myelin_delta_rest']:.4f} | "
            f"{m['direction_rate']:.3f} |"
        )
    u = result.unstratified
    lines += [
        "",
        "## Unstratified component bootstrap (comparison only)",
        "",
        "| Metric | mean | CI |",
        "|--------|-----:|----|",
        f"| L3 Δ | {u['l3_delta_rest']['mean']:.4f} | "
        f"[{u['l3_delta_rest']['ci_low']:.4f}, {u['l3_delta_rest']['ci_high']:.4f}] |",
        f"| Myelin Δ | {u['myelin_delta_rest']['mean']:.4f} | "
        f"[{u['myelin_delta_rest']['ci_low']:.4f}, {u['myelin_delta_rest']['ci_high']:.4f}] |",
        "",
        "## Interpretation",
        "",
    ]
    l3_ci = ci["l3_delta_rest"]
    my_ci = ci["myelin_delta_rest"]
    if l3_ci["ci_low"] > 0 and my_ci["ci_high"] < 0:
        lines.append(
            "- **Donor-stratified 95% CIs exclude 0 in the pre-registered directions** "
            "(L3 Δ > 0 and myelin Δ < 0) → cohort direction is robust to within-donor "
            "component resampling."
        )
    elif l3_ci["ci_low"] > 0:
        lines.append(
            "- L3 Δ CI is entirely positive; myelin CI still touches/crosses 0 under "
            "donor stratification — treat myelin depletion as directionally consistent "
            "but less tightly estimated."
        )
    else:
        lines.append(
            "- At least one primary CI includes 0 under donor stratification — "
            "direction is frequent but effect-size precision is limited."
        )
    lines += [
        "- Unstratified CIs are typically **narrower** (anticonservative).",
        "- Same-layer hard gates remain separate (see cohort meta-report).",
        "",
    ]
    for note in result.notes:
        lines.append(f"- _{note}_")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    _setup()
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    if not csv_path.is_file():
        logger.error("missing %s — run run_cohort_panel.py first", csv_path)
        return 1
    frame = load_cohort_panel(csv_path)
    result = donor_stratified_bootstrap_l3(frame, n_boot=2000, seed=0, ci_level=0.95)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    (OUT / "donor_bootstrap_l3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = write_report(result)
    (OUT / "DONOR_BOOTSTRAP_L3.md").write_text(report, encoding="utf-8")
    (BASE / "DONOR_BOOTSTRAP_L3.md").write_text(report, encoding="utf-8")
    logger.info(
        "L3 Δrest=%.4f CI[%.4f, %.4f]  myelin=%.4f CI[%.4f, %.4f]  n_comp=%s n_donor=%s",
        result.point["l3_delta_rest"],
        result.ci["l3_delta_rest"]["ci_low"],
        result.ci["l3_delta_rest"]["ci_high"],
        result.point["myelin_delta_rest"],
        result.ci["myelin_delta_rest"]["ci_low"],
        result.ci["myelin_delta_rest"]["ci_high"],
        result.n_components,
        result.n_donors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
