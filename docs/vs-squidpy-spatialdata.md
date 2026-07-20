# HistoWeave vs Squidpy / SpatialData — irreplaceable workflow differences

This page is the **positioning contract** for manuscript and reviewer language.
It states what Squidpy and SpatialData already own, and what HistoWeave adds that
those libraries do **not** replace.

## One-line split of labour

| Layer | Owner | Question answered |
|-------|--------|-------------------|
| **Data model & IO** | **SpatialData** (+ spatialdata-io) | How are multi-modal spatial objects stored, transformed, and shared? |
| **Spatial analysis grammar** | **Squidpy** (and Giotto, scanpy, …) | How do I compute graphs, statistics, and visualizations on a table? |
| **Evidence-governed method decision** | **HistoWeave** | Given a *declared task* and incomplete benchmark evidence, which method *set* is justified—and when must the workflow fall back or abstain? |

HistoWeave is **complementary**: it can ingest SpatialData/AnnData, wrap the same
upstream algorithms Squidpy users already know, and emit decision cards that
analysis toolboxes do not compute.

## What Squidpy / SpatialData already do well

- **SpatialData**: unified multi-table/image/shape representation, transformations,
  lazy IO, community exchange format.
- **Squidpy**: neighbourhood graphs, spatial statistics (e.g. Moran's I),
  co-occurrence, image features, rich plotting on AnnData/SpatialData.
- Interactive exploration and notebook-first analysis.

HistoWeave **does not claim** to replace those capabilities.

## What only HistoWeave's workflow forces (irreplaceable delta)

These steps are *executable gates*, not documentation suggestions:

1. **Task & ground-truth admissibility**  
   Cross-task evidence, circular labels (Leiden-as-domain-GT), and undeclared
   oracle-K are rejected before scores enter a decision. Squidpy will happily
   plot whatever labels you pass; SpatialData will store them. Neither refuses
   the *aggregation* of incompatible evidence into a method ranking.

2. **Non-oracle K by default**  
   Domain count is estimated unless `allow_oracle_k=True` with documented
   ablation notes. Analysis APIs typically leave K as a free parameter without
   a leakage contract.

3. **Set-valued decisions with fallback / abstention**  
   Output is not a forced singleton winner. Actions include
   `personalised_set`, `global_default`, `evidence_required`, `abstain`.
   Squidpy has no equivalent decision automaton.

4. **Grouped held-out validation as a hard gate**  
   Personalisation is not promoted without study/donor-level holdout evidence.
   Negative results (k-NN worse than global-best) *keep* the gate closed.

5. **Fail-closed SOTA backends**  
   Missing SpaGCN/GraphST/… raises an explicit failure; a toy substitute is
   never silently scored under a SOTA name. Toolbox installs fail at import
   time, but do not encode this into a multi-method comparison contract.

6. **Resource-matched comparison & Pareto set**  
   Accuracy–runtime–memory–robustness frontiers under a shared budget, with
   bootstrap frontier inclusion. Analysis suites optimise interactive use, not
   predeclared multi-objective deployment sets.

7. **Donor/study-level uncertainty**  
   Personalisation and rank statistics resample **independent study units**,
   not cells. Squidpy bootstrap helpers (where present) address measurement
   noise, not method-selection generalisation.

## Workflow that is not a Squidpy pipeline

```text
SpatialData / AnnData / HistoWeave SpatialTable
        │
        ▼
 TaskContract  ── reject invalid GT / task / oracle-K
        │
        ▼
 Reference landscape (multi-study, maturity-tiered plugins)
        │
        ▼
 Candidate generation (k-NN proxy) ── not a claim of superiority
        │
        ▼
 Grouped holdout gate ── missing / negative → global_default | abstain
        │
        ▼
 Optional Pareto intersection ── configuration-exact, resource-matched
        │
        ▼
 DecisionCard + comparison panel + HTML report (provenance, evidence roles)
```

A Squidpy notebook typically stops at *compute + plot*. The dashed boxes above
are HistoWeave's product surface.

## What we deliberately do **not** claim

| Claim | Status |
|-------|--------|
| HistoWeave is a better graph library than Squidpy | **No** |
| HistoWeave replaces SpatialData as the community data model | **No** (wrapper / consumer) |
| Unconstrained k-NN personalisation always beats global-best | **No** (often false; gate exists for this reason) |
| A decision card proves biological correctness | **No** (comparative execution priority only) |

## Manuscript wording (paste-ready)

> Squidpy and SpatialData provide the analysis grammar and data model for spatial
> omics. HistoWeave does not reimplement that stack. Its contribution is an
> evidence-governed decision workflow that analysis toolboxes leave to the user:
> typed task/ground-truth contracts, non-oracle domain-count defaults, fail-closed
> SOTA comparison under shared resources, and set-valued method decisions that
> fall back or abstain when grouped held-out validation does not support
> personalisation. Independent donor/study-level evaluations are reported with
> study-bootstrap confidence intervals and rank concordance; negative
> personalisation results remain first-class outputs that keep the global-default
> gate closed.

## Related docs

- [Decision protocol](decision-protocol.md)
- [Benchmark comparison / reviewer response](benchmark-comparison-reviewer-response.md)
- [Architecture](architecture.md)
- Independent personalisation artefacts: `independent_personalisation_results/`
