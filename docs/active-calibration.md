# Active-learning recommender calibration

When the landscape recommender **fails to beat the global-best baseline**
(`beats_global_best_baseline=False`), personalisation is not yet justified.
HistoWeave then emits an **evidence-acquisition todo list**: specific
`dataset × method` pairs to run next, ranked by expected information gain (EIG).

This is not marketing — it is an honest admission that more benchmark evidence
is needed, plus a prioritised plan to get it.

## When it fires

`MethodRecommender.recommend()` always computes baseline diagnostics:

* `global_best_method` — best mean score across the knowledge base
* `selection_regret_vs_global_best` — ARI (or metric) lost vs always picking that default
* `beats_global_best_baseline` — *True* only when the neighbour-weighted pick is
  **strictly better** than the global default

If the flag is `False` (tie or worse), active calibration attaches:

```json
{
  "evidence_todo": [
    {
      "dataset": "ref_c",
      "method": "beta",
      "expected_information_gain": 0.42,
      "reason": "missing score; near query (sim=0.81); recommendation frontier",
      "priority": 1
    }
  ],
  "calibration": { "needed": true, "protocol": "histoweave.active_calibration.v1", ... }
}
```

## EIG heuristic

For each candidate pair `(d, m)`:

```
EIG(d, m) = similarity(query, d)
            × method_importance(m)
            × novelty(d, m)
            × (1 + decision_relevance(d, m))
```

| Term | Meaning |
|------|---------|
| **similarity** | k-NN neighbour weight of dataset `d` to the query |
| **method_importance** | Higher for top-k ranked methods, global-best, high-uncertainty configs |
| **novelty** | 1 if performance cell is missing/NaN; 0 if already filled |
| **decision_relevance** | Boost when filling the cell could revise top-1 vs global-best |

Only **missing** cells are proposed (re-runs are left to multi-seed landscape
protocols).  Tasks are sorted by EIG descending.

## CLI

```bash
# Embedded in recommend when baseline is not beaten
histoweave recommend --in sample.ttab \
  --knowledge-base landscape.json

# Explicit calibration plan
histoweave calibrate-recommender \
  --in sample.ttab \
  --knowledge-base landscape.json \
  --top 10 \
  --out calibration.json
```

## Python API

```python
import logging

from histoweave.benchmark import MethodRecommender, propose_evidence_acquisition

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

recommender = MethodRecommender("landscape.json")
rec = recommender.recommend(data)
if rec.beats_global_best_baseline is False:
    _LOGGER.info("Evidence todo:")
    for item in rec.evidence_todo:
        _LOGGER.info(
            "  %s. %s × %s  EIG=%s — %s",
            item["priority"],
            item["dataset"],
            item["method"],
            item["expected_information_gain"],
            item["reason"],
        )

plan = propose_evidence_acquisition(recommender, rec, top_n=10)
_LOGGER.info("%s", plan.summary())
```

## How to use the todo list

1. Run each listed method on the listed reference dataset (or regenerate the
   landscape cell if the dataset is local).
2. Write the score back into the knowledge-base JSON
   (`performance[dataset][method] = score`).
3. Re-run `histoweave recommend` — regret and `beats_global_best_baseline`
   should improve as coverage holes close near the query.
4. If the todo is empty but baseline still fails, the KB has full coverage but
   the wrong *neighbours*: add new reference datasets close to the query in
   feature space.

## Relationship to digital twins & AutoML

| Tool | Role |
|------|------|
| Digital twin | Proxy ranking on a synthetic match of *this* sample |
| AutoML | Run top-k on the real sample + Pareto report |
| Active calibration | Tell you which *landscape cells* to fill so future recommendations improve |
