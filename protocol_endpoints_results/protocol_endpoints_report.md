# Protocol endpoints summary

Falsifiable evaluation bundle aligned with `docs/decision-protocol.md`.

## Personalisation (study-grouped holdout)

- Queries: **20** (target ≥20: met)
- Mean selection regret: **0.0468**
- Mean global-best regret: **0.0290**
- Beats global-best: **False**
- Top-1 / Top-3: **35.0%** / **60.0%**

## Selective regret–coverage

- Recommended policy: **always_global_default**
- Recommended confidence threshold: **None**
- Coverage at threshold: **0.0**
- Hybrid mean regret: **0.028961942166314713**

## Pareto stability

- Datasets: **5**
- Bootstrap resamples: **200**

## SOTA under unified resources

- Accepted cells: **210**
- Rejected cells: **0**
- Top method (mean ARI): **spagcn** (ARI=0.3171446309976544, s=18.870426666666667)

- Mean selection regret does not beat the global-best comparator (fallback / global_default remains justified).
