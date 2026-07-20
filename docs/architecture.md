# Architecture

HistoWeave is a **six-layer implementation stack governed by one decision plane**.
Each layer has a stable interface so that components
(methods, executors, viewers) are swappable and the volatile frontier never destabilizes
the core. Data flows *up* the stack (raw vendor output → canonical objects → orchestrated
methods → evaluated results → visual reports); control and provenance flow *down* (a
workflow definition pins method versions, container digests, and parameters for every run).

```
┌─────────────────────────────────────────────────────────────────────┐
│ 6 · Visualization & reporting   report/    → Vitessce, napari        │
├─────────────────────────────────────────────────────────────────────┤
│ 5 · Benchmarking & evaluation   benchmark/ → Open Problems-style CI   │
├─────────────────────────────────────────────────────────────────────┤
│ 4 · Method / plugin layer       plugins/   → wrapped R & Py methods   │
├─────────────────────────────────────────────────────────────────────┤
│ 3 · Workflow / compute layer    workflow/  → Nextflow / Snakemake     │
├─────────────────────────────────────────────────────────────────────┤
│ 2 · Data & storage layer        data/      → SpatialData / OME-Zarr   │
├─────────────────────────────────────────────────────────────────────┤
│ 1 · Ingestion / adapters        io/        → spatialdata-io           │
└─────────────────────────────────────────────────────────────────────┘
```

## 1 · Ingestion / adapters (`histoweave.io`)
Vendor & community format readers convert raw outputs (Visium/HD, Xenium, CosMx,
MERSCOPE, Stereo-seq, GeoMx) into canonical objects. In this scaffold the `Reader`
interface is defined and the concrete readers are honest stubs that activate when
`spatialdata-io` is installed (`pip install -e ".[spatial]"`).

## 2 · Data & storage (`histoweave.data`)
The single most consequential decision is to standardize on **SpatialData / OME-Zarr** as
the canonical representation (with AnnData/MuData for the tabular molecular layer). The
scaffold ships `SpatialTable`, an AnnData-shaped container that mirrors the same mental
model (`X`/`obs`/`var`/`obsm`/`uns`) and provides bridges to/from AnnData. Everything
downstream is written against `SpatialTable`, so swapping in a SpatialData-backed
implementation later is an internal change, not an API break. Every object carries
structured **provenance**.

## 3 · Workflow / compute (`histoweave.workflow` + `workflows/nextflow`)
Analyses are declarative lists of plugin steps. The scaffold's in-process `run_pipeline`
threads a single object through the steps and captures a `RunManifest`. The same
declarative definition is what a **Nextflow DSL2** backend (nf-core conventions, one
container per process) consumes to run unchanged on laptop → Slurm → Kubernetes → cloud.

## 4 · Method / plugin layer (`histoweave.plugins`)
Each analysis step is defined by a **typed interface** (`Method` + `MethodSpec`): declared
category, inputs/outputs, parameters, and assumptions. Concrete methods are plugins.
Python methods are wrapped natively; R/Bioconductor methods are wrapped as containerized
steps — so the R↔Python divide is an implementation detail, not a user problem. A
machine-readable **registry** records each plugin's metadata and benchmark standing.

## 5 · Evidence and decision plane (`histoweave.benchmark` + `histoweave.decision`)

This is the submission-facing core. Benchmark modules produce typed evidence;
`histoweave.decision` controls what that evidence may justify. Task and
ground-truth semantics are hard admissibility gates. Candidate generation,
Pareto tables, failure fingerprints, ISUS, and grouped held-out validation retain
different declared evidence roles in the final `DecisionCard`.

The decision path is deliberately auditable:

    raw bundle
      -> 13 target-free features
      -> reference-fit imputation and standardization
      -> hard task / ground-truth admissibility filter
      -> k-nearest compatible reference datasets
      -> reference-neighbour candidate proxy + global-default comparison
      -> grouped held-out validation gate
      -> matched non-dominated method set, global fallback, evidence request, or abstention

The knowledge-base schema stores task, metric, feature order, per-dataset performance,
method versions/timings, and missing evaluations. Ground-truth labels are excluded
from retrieval but their semantics remain mandatory evidence metadata. A catalogue
must pass dataset- or study-grouped held-out evaluation before the protocol permits
`personalised_set`; fit on retrieved reference neighbours is only a proxy.

## 6 · Visualization & reporting (`histoweave.report`)
Every pipeline emits a **self-contained, versioned HTML report** (QC, spatial maps,
annotation, provenance). Interactive exploration is delegated to mature viewers —
Vitessce (browser) and napari-spatialdata (desktop) — rather than rebuilt.

## Indicative technology stack

| Concern | Choice |
|--------|--------|
| Core language | Python 3.11+ |
| Data model / storage | SpatialData, OME-Zarr, AnnData/MuData, Zarr |
| Out-of-core compute | Dask (+ optional GPU via RAPIDS) |
| Workflow engine | Nextflow (nf-core conventions) |
| Containers | Docker / Apptainer |
| R interoperability | Containerized R steps; light language bridges |
| Packaging | PyPI + conda-forge / Bioconda |
| CI / testing | GitHub Actions, pytest, tiny canonical datasets |
| Docs | MkDocs / Sphinx + executable notebooks |
| Visualization | Vitessce, napari-spatialdata |
