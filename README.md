# HistoWeave

**Open-source orchestration & evaluation platform for reproducible spatial transcriptomics.**

HistoWeave is the connective tissue the spatial transcriptomics field lacks — a unified
**distribution, orchestration, and evaluation layer** that sits on top of scverse and
Bioconductor. Instead of adding yet another method to the zoo, it wraps 51 existing
methods across every analysis category behind a single plugin interface, benchmarks
them on structured reference datasets, and provides a **data-driven recommendation
engine** that suggests the best method for new data without requiring ground truth.

---

## Status

**v0.1.0-beta** — 51 methods, 11/11 analysis categories, 242 tests, 4 reference datasets,
Vitessce interactive reporting, deployable recommendation engine.  Method wrappers
exercise the real upstream libraries behind mocked integration tests; R-side methods
require the `histoweave-r` container image (ghcr.io).  See [docs/roadmap.md](docs/roadmap.md).

## Install

```bash
# PyPI (core — NumPy + pandas + Jinja2, runs anywhere)
pip install histoweave-spatial

# conda-forge (pending feedstock approval)
conda install -c conda-forge histoweave-spatial

# With real-data ingestion (10x H5, Parquet, spatialdata-io)
pip install "histoweave-spatial[io,spatial]"

# Method-specific extras
pip install "histoweave-spatial[scanpy,cell2location,scanvi,cellpose2,spatialde,liana,celltypist,deep-learning]"

# All extras + development tools
pip install "histoweave-spatial[all]"
```

> Python **3.11+** required. Conda package available once the
> [conda-forge feedstock](https://github.com/conda-forge/staged-recipes) PR lands.

## Quick start — 60 seconds to a report

```python
import histoweave as ts

# Synthetic demo: 600 cells, 3 ground-truth domains
data = ts.datasets.make_synthetic(seed=0)
result = ts.run_pipeline(data)          # QC → normalize → kmeans → annotate
ts.build_report(result, "report.html")  # Self-contained HTML with interactive Vitessce viewer
```

Or from the command line:

```bash
histoweave run --demo --out report.html
open report.html
```

## Real data workflow

```bash
# 1. Ingest vendor data into a portable bundle
histoweave ingest --input /path/to/spaceranger/outs --assay visium --out sample.ttab

# 2. Run analysis pipeline with domain count
histoweave step --in sample.ttab --category qc --method basic_qc
histoweave step --in sample.ttab --category normalization --method sctransform
histoweave step --in sample.ttab --category domain_detection --method banksy \
    --param n_domains=7 --param lambda_param=0.8

# 3. Generate interactive report
histoweave run --in sample.ttab --out report.html
```

## Method recommendation — HistoWeave's killer feature

```bash
# Reproduce the Figure 3 pilot: 10 methods x 3 datasets
histoweave benchmark --suite figure3 --out-dir figure3_results --seed 42

# Recommend the best method for new data (no ground truth needed)
histoweave recommend --in my_sample.ttab \
    --knowledge-base figure3_results/landscape.json

# Machine-readable output
histoweave recommend --in my_sample.ttab \
    --knowledge-base figure3_results/landscape.json \
    --json --out recommendation.json
```

The engine extracts 16 target-free spatial/expression/geometry features, finds the
*k* nearest reference datasets, and ranks methods by similarity-weighted performance
— without ever looking at domain labels or cell-type annotations on the query data.

The bundled Figure 3 protocol is a deterministic synthetic validation pilot. Its
outputs must be labelled as such; real-data, study-grouped validation is required
before making a method-selection generalization claim.

## What's in the box

| Category | Methods | Highlights |
|----------|:-------:|------------|
| **Ingestion** | 7 | Visium, Xenium, CosMx, MERSCOPE, MERFISH, Slide-seq, Stereo-seq |
| **QC** | 4 | basic, library-size, complexity, mitochondrial QC |
| **Normalization** | 8 | SCTransform, log1p, CLR, TF-IDF, arcsinh, sqrt, scaling |
| **Segmentation** | 1 | Cellpose 2 (cyto2/nuclei) |
| **Annotation** | 3 | scANVI, CellTypist, marker_score |
| **Domain detection** | 11 | BANKSY, spectral, GMM, k-means, DBSCAN, OPTICS, ... |
| **Deconvolution** | 2 | cell2location, marker_deconv |
| **SVG detection** | 4 | SpatialDE, Moran's I, Geary's C, spatial variance ratio |
| **Neighborhood** | 1 | spatial graph (networkx) |
| **CCC** | 1 | LIANA+ (16 method consensus) |
| **Integration** | 9 | ComBat + 8 PyTorch expression/image representation models |

All methods share a common plugin interface, parameter validation, and automatic
provenance recording.  New methods can be added as independently versioned packages.

## Architecture

```
 6 · Reporting        HTML + Vitessce interactive viewer (CDN, zero Python deps)
 5 · Benchmarking     Tasks, metrics, performance landscape, recommendation engine
 4 · Methods          51 plugins across 11 categories, typed capability interface
 3 · Workflow         In-process SDK runner + Nextflow DAG (Docker/Slurm/K8s)
 2 · Data             SpatialTable + sparse + provenance + AnnData ↔ SpatialData bridge
 1 · Ingestion        Native 10x H5 readers + spatialdata-io adapters
```

See [docs/architecture.md](docs/architecture.md) for details.

## Development

```bash
pip install -e ".[dev]"
pytest                         # 230 tests (~68s), --fail-under=80
ruff check .                   # zero-config linting
mypy src                       # full type coverage
```

## Real-data 5  10 benchmark

The reproducible DLPFC Visium performance landscape, raw result tables, manifest,
figures, and rerunnable scripts are in
[5x10_dlpfc_benchmark](5x10_dlpfc_benchmark/report_5x10_dlpfc_benchmark.md).
Set HISTOWEAVE_DLPFC_DATA to place downloaded slices outside the repository and
HISTOWEAVE_BENCHMARK_OUT to redirect regenerated outputs.
