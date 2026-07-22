# Case study: intercepting unjustified recommendations (dry-lab)

**Purpose.** Provide a *Bioinformatics*-style, wet-lab-free analysis vignette that
shows the evidence-governed decision protocol **refusing** confident but invalid
method promotions.

**Runnable artefact.**

```bash
# from repository root
python examples/case_study_intercepted_recommendation.py --out-dir .
# → intercept_case_report.md
# → intercept_case_cards.json
```

Protocol string: `histoweave.case_study.intercepted_recommendation.v1`.

---

## Scientific question

> When a naive ranker (nearest-neighbour landscape / leaderboard screenshot)
> promotes a high-scoring method, **what evidence is still required** before that
> promotion becomes a justified deployment action?

The answer is a `DecisionCard` action, not another ARI point estimate:

| Action | Meaning |
|--------|---------|
| `personalised_set` | Held-out + baseline gates passed; optional Pareto set |
| `global_default` | Local ranking lost the fixed comparator (or holdout failed) |
| `evidence_required` | Attractive proxy, but insufficient external evidence |
| `abstain` | Task-valid evidence is missing or circular |

This case study demonstrates the **three refusal modes** (plus cross-task
filtering). It does **not** claim a new clustering algorithm or a protein-validated
biological niche.

---

## Design (no tissue required)

All vignettes use either:

1. synthetic `Recommendation` objects with realistic-looking scores, or  
2. a tiny synthetic landscape + `DecisionEngine` (scenario D), or  
3. the **bundled negative external holdout**
   (`benchmark_external_validation/decision_validation.json`) — a real frozen
   control that records `beats_global_best: false`.

No H5AD download, no imaging, no wet-lab return.

---

## Scenario matrix

| ID | Failure mode | What a naive tool would do | Protocol action | Primary set |
|----|--------------|----------------------------|-----------------|-------------|
| **A** | Missing grouped holdout | Deploy the local kNN winner | `evidence_required` | (empty) |
| **B** | Negative external holdout | Personalise because neighbours look good | `global_default` | global comparator |
| **C** | Circular GT (`cluster_proxy` / Leiden-as-domain) | Trust high ARI on self-cluster labels | `abstain` | (empty) |
| **D** | Cross-task pollution (`cell_type` ARI in a domain landscape) | Promote the method that “wins” on the proxy dataset | hard-filter + no `personalised_set` | not personalised |

### A — Attractive ranking is not held-out proof

- Local method: proxy score 0.86, confidence 0.88, `beats_global_best_baseline=True`
- Validation: **omitted**
- Gate that fires: `heldout_validation = not_evaluated` → action **`evidence_required`**
- Lesson: reference-neighbour advantage is only a **candidate-generation proxy**.

### B — Frozen negative control forces fallback

- Same attractive local ranking as A
- Validation: bundled `decision_validation.json` (`beats_global_best: false`)
- Gate that fires: `heldout_validation = fail` → action **`global_default`**
- Lesson: negative results are **product features**; they cannot be edited away to
  unlock personalisation demos.

### C — Circular ground truth hard-fails

- Neighbours declare `ground_truth_kind=cluster_proxy`
- Even a fabricated *positive* holdout is supplied
- Gate that fires: `task_compatibility = fail` → action **`abstain`**
- Lesson: high ARI against Leiden-as-domain labels is **not** spatial-domain evidence.

### D — Cross-task landscape rows never enter the shortlist

- Knowledge base mixes two spatial-domain references with one `cell_type` /
  `cluster_proxy` reference that has the best numeric ARI (0.99)
- `DecisionEngine` hard-filters incompatible neighbours
- Without positive holdout, action is never `personalised_set`
- Lesson: soft down-weighting of incompatible tasks is **rejected** by design.

---

## How to cite this vignette in a manuscript

**Suggested Results sentence.**

> In a dry-lab intercept case study
> (`histoweave.case_study.intercepted_recommendation.v1`), four attractive but
> unjustified promotions were refused: missing holdout yielded
> `evidence_required`; a bundled negative external control forced
> `global_default`; circular cluster-proxy labels produced `abstain`; and
> cross-task cell-type evidence was excluded from the spatial-domain neighbour
> set (Supplementary Table / `intercept_case_report.md`).

**What not to claim.**

- That the protocol improves ARI on a real tissue section (not tested here).
- That personalisation beats a global default (contradicted by scenario B and the
  broader independent-panel results).
- That abstention equals biological correctness of any method.

---

## Relation to the decision protocol

Full rule set: [Evidence-governed decision protocol](decision-protocol.md).

| Protocol endpoint idea | Covered here? |
|------------------------|---------------|
| Invalid evidence blocked | **C**, **D** |
| Personalisation needs grouped holdout | **A**, **B** |
| Negative holdout → global default | **B** |
| Oracle-K leakage | No (see `non_oracle_k_sota/`) |
| Study-level non-inferiority panel | No (see independent personalisation artefacts) |

---

## Implementation map

| Piece | Location |
|-------|----------|
| Runnable script | [`examples/case_study_intercepted_recommendation.py`](https://github.com/HERRY423/Histoweave/blob/main/examples/case_study_intercepted_recommendation.py) |
| Unit lock | `tests/test_case_study_intercepted_recommendation.py` |
| Negative holdout fixture | `benchmark_external_validation/decision_validation.json` |
| Core engine | `histoweave.benchmark.decision` / `histoweave.decide` |
