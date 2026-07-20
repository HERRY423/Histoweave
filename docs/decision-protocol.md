# Evidence-governed decision protocol

HistoWeave has one scientific question:

> Given an explicit spatial-analysis task and incomplete benchmark evidence,
> which method set is justified, and when should the workflow fall back or
> abstain?

The answer is an executable evidence contract, not a universal recommender.
Plugins, containers, reports, digital twins, ISUS, and failure fingerprints are
infrastructure or evidence producers. They are not parallel headline claims.

## Core contribution

The protocol contributes three linked capabilities.

1. **Executable evidence admissibility.** Analysis task, ground-truth meaning,
   oracle-K status, metric direction, failure status, and resource provenance
   are checked before a score can influence a decision. Cross-task and
   self-supervised evidence is excluded rather than softly down-weighted.
2. **Evidence-limited set decisions.** The system compares a query-local ranking
   with a fixed global-default comparator and, when matched objective data are
   available, returns only non-dominated configurations. It does not collapse
   incompatible objectives into a claimed universal winner.
3. **Structured fallback and abstention.** A local ranking is not promoted to a
   personalised action without grouped held-out validation. Missing or negative
   evidence produces `global_default`, `evidence_required`, or `abstain`.

The novelty is the executable coupling of these rules. Pareto sorting, nearest
neighbours, and mutual-information estimation are established components and
are not claimed as new mathematical inventions.

## Evidence roles

| Evidence | Role in the protocol | What it cannot establish |
|---|---|---|
| Task contract and dataset metadata | Hard pre-execution admissibility gate | Biological validity of a result |
| Reference-neighbour ranking | Pre-execution candidate-generation proxy | Held-out superiority on the query |
| Grouped held-out validation | Generalisation gate for personalisation | Universal superiority outside its scope |
| Pareto objective table | Matched set-valued trade-off output | A uniquely correct method or value function |
| Failure fingerprint | Synthetic stress-test warning | Real-dataset failure probability |
| ISUS | Post-hoc, label-conditioned spatial-information descriptor | Target-free prediction of method gain |
| Post-run coherence/consensus | Comparative execution diagnostic | Ground-truth biological correctness |

These roles are emitted in every `DecisionCard.evidence_roles` record so that a
post-hoc diagnostic cannot silently become a pre-execution selector.

## Decision rule

For a declared task and query dataset, the protocol applies the following rule
in order:

1. Reject circular, cluster-proxy, or cross-task reference evidence.
   Cross-modal domain tasks (RNA / protein / chromatin partitions) are
   related but **not** admissible for each other; `virtual_st` is isolated
   from all domain-partition rankings (see
   [multimodal & virtual ST](multimodal-virtual-st.md)).
2. Require finite query-local candidates with predeclared minimum support.
3. Treat the recommender's `confidence` field as an uncalibrated rank-support
   heuristic, not a probability.
4. If the local proxy does not beat the global-default proxy, return
   `global_default`.
5. If independent grouped held-out validation is missing, return
   `evidence_required` even when the local proxy appears favourable.
6. If matched Pareto evidence is supplied, require an exact configuration-level
   intersection; `method@sw0.0` and `method@sw0.8` are different decisions.
7. Return `personalised_set` only after all gates pass. Return `abstain` when no
   task-valid evidence remains.

The default thresholds are serialized with the output. Changing them creates a
different decision policy and must be reported as such.

## Python API

```python
import logging

import histoweave as hw

logger = logging.getLogger(__name__)

card = hw.decide(
    data,
    knowledge_base="landscape.json",
    dataset_name="query_section",
    task="spatial_domain",
    platform="visium",
    # Optional, role-specific evidence:
    pareto=pareto_report,
    validation=grouped_holdout_summary,
    isus_domain_key="expert_region",  # post-hoc only
)

logger.info("decision action: %s", card.action)
logger.info("primary set: %s", card.primary_set)
logger.info("comparison set: %s", card.comparison_set)
logger.info("%s", card.summary())
```

## CLI

```bash
histoweave decide \
  --in query.ttab \
  --knowledge-base landscape.json \
  --task spatial_domain \
  --dataset-name query_section \
  --platform visium \
  --validation grouped_holdout.json \
  --pareto-report pareto.json \
  --out decision_card.json --json
```

Omit optional evidence rather than fabricating it. The resulting checks will be
`not_evaluated`, and the action will remain conservative.

Grouped held-out evidence uses a deliberately small contract:

```json
{
  "protocol": "external_holdout",
  "n_queries": 5,
  "beats_global_best": false
}
```

The bundled external validation is normalised to this schema in
`benchmark_external_validation/decision_validation.json`. It records the
current negative result and therefore forces `global_default`; it is not a demo
fabricated to unlock personalisation.

## Claim boundary

Every card carries the following boundary:

> The decision card prioritises methods for comparative execution. It does not
> establish biological validity, universal superiority, or a causal benefit
> from spatial modelling.

ISUS requires trusted domain labels and therefore cannot answer whether an
unlabelled query should use a spatial method before any method is run. Optional
coordinate-shuffle nulls (`n_null`) supply Monte Carlo p-values and Z-scores for
residual spatial MI; when present, primary bands use that permutation evidence
instead of the subjective absolute cut-offs 0.1/0.3. A separate gain map
(`fit_isus_gain_calibration` / `histoweave isus --calibrate`) binds ISUS to
observed spatial ARI gain from `benchmark_long.csv`, but the current five-slice
DLPFC calibration remains `underpowered`/`unsupported` (Spearman rho ≈ -0.30).
ISUS remains a supplementary post-hoc audit, never a pre-execution selector.

Likewise, the current external recommendation result ties the global-default
regret, while the broader cross-platform result is worse than the global
default. Those negative results motivate the fallback gate; they do not support
a superiority claim.

## Falsifiable evaluation

The paper-level contribution should be evaluated with predeclared endpoints:

| Claim | Primary endpoint | Required design |
|---|---|---|
| Invalid evidence is blocked | Incompatible-evidence admission rate = 0 | Adversarial task/GT/oracle-K contract corpus |
| Set decisions avoid dominated choices | Dominated-selection rate = 0 | Matched objectives at the same sample size and hardware |
| Abstention improves safety | Selective regret versus coverage | Leave-one-study-out, not cell-level resampling |
| Personalisation adds value | Regret non-inferior or superior to global-best | At least 15-20 independent study/donor queries |
| Pareto membership is stable | Frontier inclusion probability | Donor/bootstrap and compute-replicate perturbations |
| Oracle-K must not silently inflate SOTA | Mean ARI(oracle) − mean ARI(estimate) on dual-track long tables | ≥2 SOTA methods × ≥5 slices; both tracks reported |

Runnable implementations live in
`histoweave.benchmark.protocol_endpoints` and the operator script
`scripts/run_protocol_endpoints.py`. Reference multi-source results (20
study/slice queries, selective regret–coverage, Pareto stability,
unified-resource SOTA comparison, and **oracle-K leakage**) are archived under
`protocol_endpoints_results/`. Dual-track SOTA cells live in
`non_oracle_k_sota/` (`histoweave.non_oracle_k_sota.v1` → endpoint
`histoweave.oracle_k_leakage.v1`).

**Donor/study-level personalisation (stricter independence).** Slice-level LOO
overcounts DLPFC sections from the same donor. The independent panel collapses
slices to biological donors, keeps external/cross-platform studies as one unit
each, and expands **real** public corpora until ≥15 independent study units
(see `scripts/expand_real_independent_studies.py`). Evaluation uses a **gated**
policy (personalise only when the local proxy clears a non-inferiority gate;
otherwise global-default). Artefacts:
`independent_personalisation_results/`
(`histoweave.benchmark.independent_personalisation`).

**Relative to Squidpy / SpatialData.** Analysis grammar and data models remain
with those libraries; HistoWeave's irreplaceable layer is the evidence-governed
decision workflow (task/GT gates, abstention, fail-closed SOTA, study-level
statistics). See [vs Squidpy / SpatialData](vs-squidpy-spatialdata.md).

Until those endpoints are met, the defensible software contribution is safe,
auditable decision support—not automated discovery of a universally best method.
