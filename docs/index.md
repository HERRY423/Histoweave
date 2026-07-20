# HistoWeave

**Orchestration & evaluation for reproducible spatial transcriptomics.**

The bottleneck in spatial transcriptomics has moved. Generating spatial data is no longer
rate-limiting — turning coordinates and counts into *reproducible, interoperable,
benchmarked* biological insight is. The analytical layer is fragmented across languages,
formats proliferate, pipelines are bespoke, and method proliferation has outpaced
consensus on validation.

HistoWeave is the **connective tissue**: a unified data substrate, containerized
pipelines, typed plugins over existing R & Python methods, and a continuous benchmarking
layer that converts method proliferation into **managed selection under uncertainty**.
It is explicitly **complementary to scverse and Bioconductor** — the distribution,
orchestration, and evaluation layer on top of them, not a competing method zoo.

## Core scientific stance

- There is **no universal best method**.
- **Method × spatial-context policy** often dominates preprocessing choices.
- Spatial-domain recovery and cell-type recovery are **different tasks** and must not
  share ground-truth semantics (Leiden-as-domain-GT is rejected by task contracts).
- RNA, protein, and chromatin domain partitions are **same-family but non-transferable**
  for method ranking; H&E→expression (**virtual ST**) is a separate task scored on
  measured expression (see [multimodal & virtual ST](multimodal-virtual-st.md)).
- Recommendations report **regret vs global-best** and emit warnings when the knowledge
  base is too narrow — negative results are product features.

## Where to go next

- **[Quickstart](quickstart.md)** — run the pipeline and generate a report in five minutes.
- **[Core decision protocol](decision-protocol.md)** — evidence-governed method sets.
- **[Case study: intercept bad recommendations](case-study-intercepted-recommendation.md)** — dry-lab vignette (`evidence_required` / `global_default` / `abstain`).
- **[Reference artefacts](reference-artefacts.md)** — which evaluation summaries are in git vs local-only.
- **[Multimodal tasks & virtual ST](multimodal-virtual-st.md)** — cross-modal rules + H&E→ST.
- **[Method selection](method-selection.md)** — task-aware guidance for analysts.
- **[Architecture](architecture.md)** — the six-layer stack and why each layer exists.
- **[Concepts](concepts.md)** — the data model, plugins, provenance, and benchmarking.
- **[Roadmap](roadmap.md)** — phased plan toward a stable platform release.

!!! note "v0.1.0-beta"
    Built-in methods include field-standard wrappers (beta/production/validated) and an
    explicit research incubator (`weave_*`). Heavy optional backends fail closed rather
    than substituting toy algorithms.
