# HistoWeave

**An open-source orchestration & evaluation platform for reproducible spatial
transcriptomics analysis.**

The bottleneck in spatial transcriptomics has moved. Generating spatial data is no longer
rate-limiting — turning coordinates and counts into *reproducible, interoperable,
benchmarked* biological insight at scale is. The analytical layer is fragmented across two
language ecosystems, formats proliferate, pipelines are bespoke, and method proliferation
has outpaced any consensus on validation.

HistoWeave is the **connective tissue**: a unified data substrate, scalable containerized
pipelines, a plugin interface that wraps existing R & Python methods behind stable APIs,
and a continuous benchmarking harness that converts method proliferation into *guided
method selection*. It is explicitly **complementary to scverse and Bioconductor** — the
distribution, orchestration, and evaluation layer on top of them, not a competing method
zoo.

## Where to go next

- **[Quickstart](quickstart.md)** — run the pipeline and generate a report in five minutes.
- **[Architecture](architecture.md)** — the six-layer stack and why each layer exists.
- **[Concepts](concepts.md)** — the data model, plugins, provenance, and benchmarking.
- **[Roadmap](roadmap.md)** — phased plan from scaffold to v1.0.

!!! warning "Pre-alpha scaffold"
    This is the Phase-0 walking skeleton. The architecture runs end-to-end on synthetic
    data; the built-in methods are simple reference implementations and vendor readers are
    stubs that activate with the `spatial` extra.
