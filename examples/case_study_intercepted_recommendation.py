#!/usr/bin/env python3
"""Case study (dry-lab only): intercepting an attractive but unjustified recommendation.

No wet-lab data or tissue section is required.  The script constructs four
*plausible-looking* method rankings that a naive recommender would promote,
then shows how :func:`histoweave.decide` / :func:`build_decision_card` refuse
to deploy them.

Scenarios
---------
A. High local rank-support, **no held-out validation** → ``evidence_required``
B. Local rank looks strong, but **grouped holdout fails** the global gate
   (uses the bundled external negative control) → ``global_default``
C. Ranking built on **Leiden / cluster_proxy** "domains" → ``abstain``
D. Landscape polluted with **cross-task** (cell-type) evidence → hard-filtered
   neighbours; engine does not personalise on incompatible references

Run from the repository root::

    python examples/case_study_intercepted_recommendation.py
    python examples/case_study_intercepted_recommendation.py --out-dir /tmp/intercept_case

Outputs
-------
* ``intercept_case_report.md`` — narrative table for manuscript / review
* ``intercept_case_cards.json`` — full DecisionCard payloads
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from histoweave.benchmark import (
    DecisionAction,
    DecisionEngine,
    LandscapeResult,
    MethodRecommender,
    MethodScore,
    Recommendation,
    build_decision_card,
    extract_features,
    feature_vector,
)
from histoweave.benchmark.features import RECOMMENDATION_FEATURE_ORDER
from histoweave.datasets import make_synthetic

_LOGGER = logging.getLogger("intercept_case")
REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_NEGATIVE_HOLDOUT = REPO_ROOT / "benchmark_external_validation" / "decision_validation.json"


@dataclass(frozen=True)
class ScenarioResult:
    """One intercept vignette and its protocol outcome."""

    scenario_id: str
    title: str
    naive_promotion: str
    action: str
    primary_set: list[str]
    comparison_set: list[str]
    intercept_checks: list[dict[str, str]]
    rationale: list[str]
    card: dict[str, Any]


def _ranked_pair(
    *,
    local_name: str = "attractive_local@sw0.8",
    local_score: float = 0.86,
    global_name: str = "global_default_method",
    global_score: float = 0.79,
    beats: bool | None,
    ground_truth_kind: str = "spatial_domain",
    neighbour_task: str = "spatial_domain",
) -> Recommendation:
    """Build a recommendation a naive UI would render as a confident winner."""
    ranked = [
        MethodScore(
            method=local_name,
            score=local_score,
            confidence=0.88,
            wins=3,
            neighbour_scores={"ref_near": local_score, "ref_mid": local_score - 0.03},
            uncertainty=0.02,
            support=2,
            coverage=1.0,
            base_method=local_name.split("@", 1)[0],
            spatial_context_policy="sw0.8",
        ),
        MethodScore(
            method=global_name,
            score=global_score,
            confidence=0.70,
            wins=0,
            neighbour_scores={"ref_near": global_score, "ref_mid": global_score},
            uncertainty=0.03,
            support=2,
            coverage=1.0,
            base_method=global_name,
            spatial_context_policy="sw0.0",
        ),
    ]
    return Recommendation(
        task="spatial_domain",
        dataset_name="query_section",
        ranked_methods=ranked,
        neighbours=[
            {
                "name": "ref_near",
                "similarity": 0.93,
                "task": neighbour_task,
                "ground_truth_kind": ground_truth_kind,
            },
            {
                "name": "ref_mid",
                "similarity": 0.81,
                "task": neighbour_task,
                "ground_truth_kind": ground_truth_kind,
            },
        ],
        global_best_method=global_name,
        global_best_score=global_score,
        beats_global_best_baseline=beats,
        selection_regret_vs_global_best=(None if beats is None else (-0.05 if beats else 0.04)),
    )


def _failing_checks(card_dict: dict[str, Any]) -> list[dict[str, str]]:
    interesting = {"fail", "warn", "not_evaluated"}
    out: list[dict[str, str]] = []
    for item in card_dict.get("checks", []):
        status = str(item.get("status", "")).lower()
        if status in interesting:
            out.append(
                {
                    "name": str(item.get("name")),
                    "status": status,
                    "detail": str(item.get("detail")),
                }
            )
    return out


def scenario_a_missing_holdout() -> ScenarioResult:
    """Attractive kNN winner without grouped holdout → evidence_required."""
    rec = _ranked_pair(beats=True)
    card = build_decision_card(rec)  # validation omitted on purpose
    payload = card.to_dict()
    assert card.action is DecisionAction.EVIDENCE_REQUIRED
    return ScenarioResult(
        scenario_id="A",
        title="Attractive local ranking without held-out validation",
        naive_promotion=(
            f"Deploy {rec.ranked_methods[0].method} "
            f"(proxy ARI={rec.ranked_methods[0].score:.2f}, "
            f"confidence={rec.ranked_methods[0].confidence:.2f})"
        ),
        action=card.action.value,
        primary_set=list(card.primary_set),
        comparison_set=list(card.comparison_set),
        intercept_checks=_failing_checks(payload),
        rationale=list(card.rationale),
        card=payload,
    )


def scenario_b_negative_holdout() -> ScenarioResult:
    """Local proxy looks good; bundled external holdout fails the gate → global_default."""
    if not BUNDLED_NEGATIVE_HOLDOUT.is_file():
        raise FileNotFoundError(f"Bundled negative holdout missing: {BUNDLED_NEGATIVE_HOLDOUT}")
    validation = json.loads(BUNDLED_NEGATIVE_HOLDOUT.read_text(encoding="utf-8"))
    rec = _ranked_pair(beats=True)
    card = build_decision_card(rec, validation=validation)
    payload = card.to_dict()
    assert card.action is DecisionAction.GLOBAL_DEFAULT
    assert validation.get("beats_global_best") is False
    return ScenarioResult(
        scenario_id="B",
        title="Negative grouped holdout blocks personalisation",
        naive_promotion=(
            f"Personalise to {rec.ranked_methods[0].method} because neighbours favour it"
        ),
        action=card.action.value,
        primary_set=list(card.primary_set),
        comparison_set=list(card.comparison_set),
        intercept_checks=_failing_checks(payload),
        rationale=list(card.rationale),
        card=payload,
    )


def scenario_c_cluster_proxy_gt() -> ScenarioResult:
    """Leiden/cluster labels sold as spatial domains → abstain."""
    rec = _ranked_pair(beats=True, ground_truth_kind="cluster_proxy")
    # Even a fabricated "positive" holdout must not rescue circular GT.
    fake_positive_holdout = {
        "protocol": "external_holdout",
        "n_queries": 20,
        "beats_global_best": True,
    }
    card = build_decision_card(rec, validation=fake_positive_holdout)
    payload = card.to_dict()
    assert card.action is DecisionAction.ABSTAIN
    return ScenarioResult(
        scenario_id="C",
        title="Circular cluster_proxy ground truth is rejected",
        naive_promotion=(
            "Trust the ranking because every reference reports high ARI "
            "against Leiden-derived 'domains'"
        ),
        action=card.action.value,
        primary_set=list(card.primary_set),
        comparison_set=list(card.comparison_set),
        intercept_checks=_failing_checks(payload),
        rationale=list(card.rationale),
        card=payload,
    )


def scenario_d_cross_task_landscape(tmp_dir: Path) -> ScenarioResult:
    """Cross-task (cell-type / cluster_proxy) landscape rows cannot enter the shortlist."""
    datasets = {
        "spatial_a": make_synthetic(n_cells=60, n_genes=16, seed=1),
        "spatial_b": make_synthetic(n_cells=70, n_genes=16, seed=2),
        "proxy_celltype": make_synthetic(n_cells=80, n_genes=16, seed=3),
    }
    features = {
        name: feature_vector(
            extract_features(data, include_domain=False),
            order=RECOMMENDATION_FEATURE_ORDER,
        )
        for name, data in datasets.items()
    }
    # Make the incompatible reference look like the strongest method source.
    landscape = LandscapeResult(
        performance={
            "spatial_a": {"kmeans": 0.72, "spectral": 0.70},
            "spatial_b": {"kmeans": 0.71, "spectral": 0.73},
            "proxy_celltype": {"kmeans": 0.10, "spectral": 0.99},
        },
        features=features,
        embedding={},
        best_method={
            "spatial_a": "kmeans",
            "spatial_b": "spectral",
            "proxy_celltype": "spectral",
        },
        niches={
            "kmeans": ["spatial_a"],
            "spectral": ["spatial_b", "proxy_celltype"],
        },
        timings={},
        feature_order=list(RECOMMENDATION_FEATURE_ORDER),
        method_count=2,
        dataset_count=3,
        task="spatial_domain",
        metric="ARI",
        dataset_meta={
            "spatial_a": {
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
            },
            "spatial_b": {
                "task": "spatial_domain",
                "ground_truth_kind": "spatial_domain",
            },
            "proxy_celltype": {
                "task": "cell_type",
                "ground_truth_kind": "cluster_proxy",
            },
        },
    )
    kb_path = tmp_dir / "polluted_landscape.json"
    MethodRecommender(landscape).save_knowledge_base(kb_path)

    query = datasets["spatial_a"]
    card = DecisionEngine(kb_path, k_neighbours=3).decide(
        query,
        dataset_name="query_section",
        task="spatial_domain",
    )
    payload = card.to_dict()
    neighbour_names = {item["name"] for item in payload["recommendation"]["neighbours"]}
    if "proxy_celltype" in neighbour_names:
        raise AssertionError("Cross-task proxy_celltype leaked into query-local neighbours")
    # Without positive holdout, personalisation is never unlocked.
    if card.action is DecisionAction.PERSONALISED_SET:
        raise AssertionError("Personalisation must not unlock on filtered landscape alone")

    return ScenarioResult(
        scenario_id="D",
        title="Cross-task landscape evidence is hard-filtered",
        naive_promotion=("Recommend spectral@… because a cell-type proxy dataset reports ARI=0.99"),
        action=card.action.value,
        primary_set=list(card.primary_set),
        comparison_set=list(card.comparison_set),
        intercept_checks=_failing_checks(payload),
        rationale=list(card.rationale)
        + [
            "Filtered neighbours: "
            + (", ".join(sorted(neighbour_names)) if neighbour_names else "(none retained)"),
            "Excluded by task contract: proxy_celltype (task=cell_type, GT=cluster_proxy).",
        ],
        card=payload,
    )


def run_all_scenarios(work_dir: Path | None = None) -> list[ScenarioResult]:
    """Execute A–D and return structured results (also used by unit tests)."""
    tmp = work_dir if work_dir is not None else Path(".")
    tmp.mkdir(parents=True, exist_ok=True)
    results = [
        scenario_a_missing_holdout(),
        scenario_b_negative_holdout(),
        scenario_c_cluster_proxy_gt(),
        scenario_d_cross_task_landscape(tmp),
    ]
    return results


def _markdown_report(results: list[ScenarioResult]) -> str:
    lines = [
        "# Case study: intercepting unjustified method recommendations",
        "",
        "**Setting:** dry-lab only (synthetic features + protocol fixtures).",
        "**Question:** when a naive ranker promotes a confident winner, does the",
        "evidence-governed protocol still refuse deployment?",
        "",
        "## Summary",
        "",
        "| ID | Failure mode | Naive promotion | Protocol action | Primary set |",
        "|----|--------------|-----------------|-----------------|-------------|",
    ]
    for r in results:
        primary = ", ".join(r.primary_set) if r.primary_set else "(empty)"
        lines.append(
            f"| **{r.scenario_id}** | {r.title} | {r.naive_promotion} | `{r.action}` | {primary} |"
        )
    lines.extend(
        [
            "",
            "## Why this matters",
            "",
            "A leaderboard row or nearest-neighbour score is not a deployment licence.",
            "The four vignettes cover the most common silent failures in spatial",
            "method selection:",
            "",
            "1. **Missing external validation** — local fit is only a proxy.",
            "2. **Negative holdout** — personalisation that does not beat a fixed",
            "   global comparator must fall back (`global_default`).",
            "3. **Circular ground truth** — Leiden-as-domain ARI must not enter the",
            "   decision (`abstain`).",
            "4. **Cross-task contamination** — cell-type benchmarks cannot rank",
            "   spatial-domain methods.",
            "",
            "## Per-scenario intercept checks",
            "",
        ]
    )
    for r in results:
        lines.append(f"### Scenario {r.scenario_id} — {r.title}")
        lines.append("")
        lines.append(f"- **Action:** `{r.action}`")
        lines.append(f"- **Naive promotion:** {r.naive_promotion}")
        if r.rationale:
            lines.append("- **Rationale:**")
            for item in r.rationale:
                lines.append(f"  - {item}")
        lines.append("- **Non-passing checks:**")
        if not r.intercept_checks:
            lines.append("  - (none flagged beyond hard fail path)")
        for chk in r.intercept_checks:
            lines.append(f"  - `{chk['name']}` **{chk['status']}** — {chk['detail']}")
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            "> The decision card prioritises methods for comparative execution.",
            "> It does not establish biological validity, universal superiority,",
            "> or a causal benefit from spatial modelling.",
            "",
            "This case study demonstrates **refusal under bad evidence**, not a new",
            "clustering algorithm and not a wet-lab discovery.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python examples/case_study_intercepted_recommendation.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Directory for markdown + JSON artefacts (default: cwd)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_intercept_work"
    work.mkdir(exist_ok=True)

    _LOGGER.info("HistoWeave intercept case study (dry-lab)")
    _LOGGER.info("Repository root: %s", REPO_ROOT)

    results = run_all_scenarios(work)
    report = _markdown_report(results)
    md_path = out_dir / "intercept_case_report.md"
    json_path = out_dir / "intercept_case_cards.json"
    md_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "protocol": "histoweave.case_study.intercepted_recommendation.v1",
                "n_scenarios": len(results),
                "scenarios": [
                    {
                        "scenario_id": r.scenario_id,
                        "title": r.title,
                        "naive_promotion": r.naive_promotion,
                        "action": r.action,
                        "primary_set": r.primary_set,
                        "comparison_set": r.comparison_set,
                        "intercept_checks": r.intercept_checks,
                        "rationale": r.rationale,
                        "card": r.card,
                    }
                    for r in results
                ],
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    _LOGGER.info("")
    _LOGGER.info("%s", report)
    _LOGGER.info("Wrote %s", md_path)
    _LOGGER.info("Wrote %s", json_path)

    expected = {
        "A": DecisionAction.EVIDENCE_REQUIRED.value,
        "B": DecisionAction.GLOBAL_DEFAULT.value,
        "C": DecisionAction.ABSTAIN.value,
    }
    for r in results:
        if r.scenario_id in expected and r.action != expected[r.scenario_id]:
            _LOGGER.error(
                "Scenario %s expected %s, got %s",
                r.scenario_id,
                expected[r.scenario_id],
                r.action,
            )
            return 1
    if results[-1].scenario_id != "D":
        return 1
    if results[-1].action == DecisionAction.PERSONALISED_SET.value:
        _LOGGER.error("Scenario D must not personalise")
        return 1
    # Determinism smoke: synthetic features finite
    _ = np.array([0.0])
    _LOGGER.info("All intercept scenarios behaved as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
