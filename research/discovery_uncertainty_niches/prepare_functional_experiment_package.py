#!/usr/bin/env python3
"""Build F3/F4 functional experiment package (perturbation / lineage / orthogonal).

Outputs under ``results/functional_experiments/``:

* ``registry.json`` — frozen experiment catalogue
* ``EXPERIMENT_MATRIX.csv`` — one row per experiment
* ``templates/functional_return_template.csv`` — return schema for labs
* ``templates/`` per-class README snippets
* ``LAB_HANDOFF.md`` — hand-off for core facilities
* ``FUNCTIONAL_EXPERIMENTS.md`` — full protocol narrative (also track root)

Does **not** invent wet-lab results.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from functional_experiments import EXPERIMENTS, registry_summary

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "functional_experiments"
TEMPLATES = OUT / "templates"

logger = logging.getLogger("functional_exp_package")


RETURN_COLUMNS = [
    "experiment_id",
    "discovery_id",
    "class",  # perturbation | lineage | orthogonal
    "modality",
    "replicate_id",
    "condition",  # treatment / control / lineage_pos / …
    "sample_id",
    "roi_id",
    "metric_name",  # e.g. MBP_delta_rest, plasticity_module, KCNN4_protein
    "metric_value",
    "n_units",  # cells/spots/animals contributing to metric
    "direction_expected",  # up | down | enrich
    "pass_flag_lab",  # optional lab self-call: true/false/pending
    "notes",
]


def write_templates() -> None:
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    # Empty template with header + example rows (commented style via notes)
    examples = [
        {
            "experiment_id": "D1_orthogonal_merfish_xenium_myelin",
            "discovery_id": "D1_L6_myelin",
            "class": "orthogonal",
            "modality": "xenium",
            "replicate_id": "rep1",
            "condition": "cryptic_roi",
            "sample_id": "donorX_section1",
            "roi_id": "deep_layer_comp0",
            "metric_name": "myelin_delta_rest",
            "metric_value": 0.42,
            "n_units": 120,
            "direction_expected": "up",
            "pass_flag_lab": "pending",
            "notes": "EXAMPLE — replace with real return; delete example rows",
        },
        {
            "experiment_id": "D1_orthogonal_merfish_xenium_myelin",
            "discovery_id": "D1_L6_myelin",
            "class": "orthogonal",
            "modality": "xenium",
            "replicate_id": "rep1",
            "condition": "rest",
            "sample_id": "donorX_section1",
            "roi_id": "deep_layer_comp0",
            "metric_name": "myelin_shift_p",
            "metric_value": 0.01,
            "n_units": 120,
            "direction_expected": "down",  # p should be small
            "pass_flag_lab": "pending",
            "notes": "EXAMPLE shift p; lower is stronger",
        },
        {
            "experiment_id": "D3_drug_ca_mek_inhibitors",
            "discovery_id": "D3_LN_ca2",
            "class": "perturbation",
            "modality": "drug",
            "replicate_id": "rep1",
            "condition": "mek_inhibitor",
            "sample_id": "tonsil_explant_01",
            "roi_id": "ca2_niche",
            "metric_name": "ca2_module_score",
            "metric_value": 0.12,
            "n_units": 800,
            "direction_expected": "down",
            "pass_flag_lab": "pending",
            "notes": "EXAMPLE vs vehicle row required",
        },
    ]
    pd.DataFrame(examples, columns=RETURN_COLUMNS).to_csv(
        TEMPLATES / "functional_return_template.csv", index=False
    )
    (TEMPLATES / "README_RETURN.md").write_text(
        """# Functional experiment return format

Place filled CSVs in ``results/functional_experiments/returns/`` named:

```
RETURN_<experiment_id>_<date>.csv
```

or a single ``functional_return_all.csv`` concatenating all rows.

## Required columns

"""
        + "\n".join(f"- `{c}`" for c in RETURN_COLUMNS)
        + """

## Rules

1. Always include **control** and **treatment** (or ROI vs non-ROI) conditions.
2. One metric per row; multiple metrics → multiple rows sharing replicate_id.
3. Do not put SCGB/SAA as primary metrics for D1/D2 (see FUNCTIONAL_VALIDATION.md artifacts).
4. Run analyzer:

```bash
python research/discovery_uncertainty_niches/analyze_functional_return.py
# dry-run schema check:
python research/discovery_uncertainty_niches/analyze_functional_return.py --dry-run
```
""",
        encoding="utf-8",
    )


def write_matrix() -> pd.DataFrame:
    rows = [e.to_dict() for e in EXPERIMENTS]
    # flatten tuples for CSV
    flat = []
    for r in rows:
        flat.append(
            {
                "experiment_id": r["experiment_id"],
                "discovery_id": r["discovery_id"],
                "class": r["class"],
                "modality": r["modality"],
                "priority": r["priority"],
                "claim_level_on_pass": r["claim_level_on_pass"],
                "title": r["title"],
                "system": r["system"],
                "targets": "|".join(r["targets"]),
                "related_roi": r["related_roi"],
                "n_replicates_min": r["n_replicates_min"],
                "primary_contrast": r["primary_contrast"],
                "pass_criteria": " || ".join(r["pass_criteria"]),
            }
        )
    df = pd.DataFrame(flat)
    df.to_csv(OUT / "EXPERIMENT_MATRIX.csv", index=False)
    return df


def write_lab_handoff() -> None:
    lines = [
        "# Functional experiment lab hand-off (F3 / F4)",
        "",
        f"**Protocol:** histoweave.functional_experiments.v1  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "",
        "This package specifies **perturbation (CRISPR/drug)**, **lineage tracing**,",
        "and **orthogonal platform** experiments to validate cryptic states D1–D3.",
        "It does not contain invented results.",
        "",
        "## Priority experiments (start here)",
        "",
        "| Priority | ID | Class | Discovery | Claim if pass |",
        "|---------:|----|-------|-----------|---------------|",
    ]
    for e in sorted(EXPERIMENTS, key=lambda x: (x.priority, x.experiment_id)):
        if e.priority > 1:
            continue
        lines.append(
            f"| {e.priority} | `{e.experiment_id}` | {e.class_} | {e.discovery_id} | {e.claim_level_on_pass} |"
        )
    lines += [
        "",
        "## Per-discovery menu",
        "",
    ]
    for disc in ("D1_L6_myelin", "D2_L3_plasticity", "D3_LN_ca2"):
        lines.append(f"### {disc}")
        lines.append("")
        for e in EXPERIMENTS:
            if e.discovery_id != disc:
                continue
            lines.append(f"#### `{e.experiment_id}` ({e.class_} / {e.modality})")
            lines.append("")
            lines.append(f"- **Title:** {e.title}")
            lines.append(f"- **Hypothesis:** {e.hypothesis}")
            lines.append(f"- **System:** {e.system}")
            lines.append(f"- **Targets:** {', '.join(e.targets)}")
            lines.append(f"- **Contrast:** {e.primary_contrast}")
            lines.append(f"- **Min replicates:** {e.n_replicates_min}")
            lines.append(f"- **Pass criteria:**")
            for c in e.pass_criteria:
                lines.append(f"  - {c}")
            lines.append(f"- **ROI link:** `{e.related_roi}`")
            if e.notes:
                lines.append(f"- **Notes:** {e.notes}")
            lines.append("")
    lines += [
        "## Return path",
        "",
        "1. Fill `templates/functional_return_template.csv`.",
        "2. Save under `results/functional_experiments/returns/`.",
        "3. `python research/discovery_uncertainty_niches/analyze_functional_return.py`",
        "",
        "## Safety / ethics",
        "",
        "- CRISPR and in vivo lineage work require local IACUC/IRB and biosafety approval.",
        "- Drug concentrations are **design placeholders** — finalize with pharmacology SOP.",
        "- Human LN/tonsil tissue: informed consent and de-identification required.",
        "",
    ]
    (OUT / "LAB_HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")


def write_full_markdown(df: pd.DataFrame) -> None:
    summary = registry_summary()
    by_class = summary["by_class"]
    lines = [
        "# Functional experiments — perturbation, lineage, orthogonal platforms",
        "",
        f"**Protocol:** `{summary['protocol']}`  ",
        f"**n experiments:** {summary['n_experiments']}  ",
        f"**Classes:** perturbation={by_class.get('perturbation', 0)}, "
        f"lineage={by_class.get('lineage', 0)}, "
        f"orthogonal={by_class.get('orthogonal', 0)}",
        "",
        "> **Scope.** Pre-registered **F3/F4** experiments that can upgrade",
        "> computational F2 cryptic-state claims. No wet-lab outcomes are claimed",
        "> until returns pass `analyze_functional_return.py`.",
        "",
        "---",
        "",
        "## How this upgrades claim levels",
        "",
        "| Level | Evidence class | Typical experiment |",
        "|------:|----------------|--------------------|",
        "| F2 | Computational dual-axis | modules on DE (done) |",
        "| **F3** | Orthogonal assay / lineage map | MERFISH/Xenium/CODEX; lineage density |",
        "| **F4** | Perturbation causality | CRISPR; drug; demyelination model |",
        "",
        "```",
        "F2 computational map",
        "        │",
        "        ├─► Orthogonal platform (F3) ──► same program, new assay",
        "        ├─► Lineage tracing (F3) ─────► cell-of-origin / descendant map",
        "        └─► CRISPR / drug (F4) ───────► necessity / disease mechanism",
        "```",
        "",
        "---",
        "",
        "## Experiment matrix (summary)",
        "",
        "| ID | Disc | Class | Modality | Prio | Level |",
        "|----|------|-------|----------|-----:|-------|",
    ]
    for e in sorted(EXPERIMENTS, key=lambda x: (x.discovery_id, x.priority, x.experiment_id)):
        lines.append(
            f"| `{e.experiment_id}` | {e.discovery_id} | {e.class_} | "
            f"{e.modality} | {e.priority} | {e.claim_level_on_pass} |"
        )

    lines += [
        "",
        "Full table: `results/functional_experiments/EXPERIMENT_MATRIX.csv`.",
        "",
        "---",
        "",
        "## D1 — L6 myelin microcompartment",
        "",
        "| Class | Experiment | Key targets | Pass gist |",
        "|-------|------------|-------------|-----------|",
        "| Orthogonal | MERFISH/Xenium myelin | MBP PLP1 MOBP | myelin Δrest>0, shift p≤0.05 on new platform |",
        "| Lineage | OPC reporter | PDGFRA/OLIG2 | lineage density ↑ in ROI |",
        "| CRISPR | MYRF/OLIG2/SOX10 CRISPRi | oligo drivers | myelin program ↓, layer ID intact |",
        "| Drug | Cuprizone/LPC demyelination | myelin integrity | niche myelin program shrinks then recovers |",
        "",
        "## D2 — L3 plasticity microcompartment",
        "",
        "| Class | Experiment | Key targets | Pass gist |",
        "|-------|------------|-------------|-----------|",
        "| Orthogonal | multiome/snRNA+spatial | ENC1 HOPX GAP43 | mid-layer state maps into L3 ROI |",
        "| Lineage | HOPX-CreERT2 | HOPX | lineage enriched in L3 cryptic ROI |",
        "| CRISPR | ENC1/HOPX CRISPRi | plasticity | module ↓ |",
        "| Drug | TTX / experience | activity | module moves in pre-registered direction |",
        "",
        "## D3 — LN Ca²⁺ micro-niche",
        "",
        "| Class | Experiment | Key targets | Pass gist |",
        "|-------|------------|-------------|-----------|",
        "| Orthogonal | CODEX / 2nd Xenium | KCNN4 ORAI3 BCL6 | protein or 2nd donor same-domain ↑; not GC |",
        "| Lineage | CITE-seq / immune barcode | lineage + Ca²⁺ | lineage composition non-random |",
        "| CRISPR | KCNN4/ORAI3 KO | Ca²⁺ | module ↓; GC counter holds |",
        "| Drug | SOCE / MEK inhibitors | Ca²⁺ MAPK | module ↓ at pre-registered dose |",
        "",
        "---",
        "",
        "## Detailed registry",
        "",
    ]
    for e in EXPERIMENTS:
        lines.append(f"### `{e.experiment_id}`")
        lines.append("")
        lines.append(f"- **Discovery:** `{e.discovery_id}`")
        lines.append(f"- **Class / modality:** {e.class_} / {e.modality}")
        lines.append(f"- **Claim on pass:** {e.claim_level_on_pass} (priority {e.priority})")
        lines.append(f"- **Title:** {e.title}")
        lines.append(f"- **Hypothesis:** {e.hypothesis}")
        lines.append(f"- **System:** {e.system}")
        lines.append(f"- **Targets:** {', '.join(f'`{t}`' for t in e.targets)}")
        lines.append(f"- **Readouts:** {'; '.join(e.readouts)}")
        lines.append(f"- **Contrast:** {e.primary_contrast}")
        lines.append(f"- **Controls:** {', '.join(e.controls) if e.controls else '—'}")
        lines.append(f"- **Min n:** {e.n_replicates_min}")
        lines.append("- **Pass criteria:**")
        for c in e.pass_criteria:
            lines.append(f"  - {c}")
        lines.append(f"- **Related ROI:** `{e.related_roi}`")
        if e.notes:
            lines.append(f"- **Notes:** {e.notes}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Return analysis",
        "",
        "```bash",
        "# Build / refresh this package",
        "python research/discovery_uncertainty_niches/prepare_functional_experiment_package.py",
        "",
        "# Schema check (no data)",
        "python research/discovery_uncertainty_niches/analyze_functional_return.py --dry-run",
        "",
        "# Score real returns",
        "python research/discovery_uncertainty_niches/analyze_functional_return.py",
        "```",
        "",
        "Analyzer writes `results/functional_experiments/RETURN_REPORT.md` and",
        "updates claim status JSON. Simulated data must be labelled",
        "`notes` containing `SIMULATED` or use `--simulate` (explicitly non-claim).",
        "",
        "---",
        "",
        "## Artifact & stats continuity",
        "",
        "- SCGB/SAA are **not** valid primary readouts (see FUNCTIONAL_VALIDATION.md).",
        "- D3 GC counter remains a non-enrichment control; orthogonal CODEX must show",
        "  BCL6 not GC-like high in the niche.",
        "- All pass criteria are pre-registered in `functional_experiments.py`.",
        "",
        "## Honesty banner",
        "",
        "* Package ≠ completed F3/F4 validation.",
        "* CRISPR/drug parameters require local biosafety and pharmacology approval.",
        "* Do not cite empty returns or `--simulate` as causal proof.",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "FUNCTIONAL_EXPERIMENTS.md").write_text(text, encoding="utf-8")
    (BASE / "FUNCTIONAL_EXPERIMENTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "returns").mkdir(exist_ok=True)

    reg = {
        "protocol": "histoweave.functional_experiments.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": registry_summary(),
        "experiments": [e.to_dict() for e in EXPERIMENTS],
        "return_columns": RETURN_COLUMNS,
    }
    (OUT / "registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    logger.info("wrote registry.json (%s experiments)", len(EXPERIMENTS))

    df = write_matrix()
    logger.info("wrote EXPERIMENT_MATRIX.csv (%s rows)", len(df))
    write_templates()
    write_lab_handoff()
    write_full_markdown(df)
    logger.info("wrote FUNCTIONAL_EXPERIMENTS.md + LAB_HANDOFF.md")


if __name__ == "__main__":
    main()
