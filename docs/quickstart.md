# Quickstart Guide

A 10-minute walkthrough from zero to a spatial transcriptomics analysis with method
recommendation.  CLI and Python SDK paths are shown side by side throughout.

---

## 1. Install

```bash
# Core — NumPy + pandas + Jinja2, runs anywhere
pip install histoweave-spatial

# With real-data ingestion (10x H5, Parquet, spatialdata-io)
pip install "histoweave-spatial[io,spatial,scanpy]"
```

Python **3.11+** required.

## 2. Explore available methods

```bash
histoweave list-methods
```

```
annotation               3   celltypist, marker_score, scanvi
ccc                      1   liana_plus
deconvolution            2   cell2location, marker_deconv
domain_detection        10   agglomerative, banksy, birch, dbscan, gaussian_mixture,
                             kmeans, mean_shift, minibatch_kmeans, optics, spectral
ingestion                3   stereoseq_reader, visium_reader, xenium_reader
integration              1   combat
neighborhood             1   spatial_graph
normalization            3   log1p_cp10k, r_lognorm, sctransform
qc                       1   basic_qc
segmentation             1   cellpose2
svg                      2   morans_i, spatialde
```

Filter by category:

```bash
histoweave list-methods --category domain_detection --json
```

## 3. Try the synthetic demo (no data needed)

```bash
histoweave run --demo --out demo_report.html
open demo_report.html
```

This runs QC → normalization → k-means domain detection → annotation and generates
a **self-contained HTML report** with interactive Vitessce spatial scatterplots,
expression heatmaps, and a complete provenance table — all on synthetic data.

```python
import logging

import histoweave as ts

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

data = ts.datasets.make_synthetic(seed=0)     # 600 cells, 3 ground-truth domains
result = ts.run_pipeline(data, verbose=True)   # QC → normalize → domains → annotate
ts.build_report(result, "report.html")         # HTML + Vitessce interactive viewer

# Every step is provenance-tracked (method, version, params, timing)
for step in result.uns["run_manifest"]["steps"]:
    _LOGGER.info("%s %s %s", step["category"], step["method"], step["version"])
```

## 4. Ingest real Visium data

Point at a Space Ranger `outs/` directory:

```bash
histoweave ingest --input /path/to/spaceranger/outs --assay visium --out sample.ttab
```

```python
from histoweave.io import read
data = read("visium", "/path/to/spaceranger/outs")
```

For Xenium in-situ data:

```bash
histoweave ingest --input /path/to/xenium/output --assay xenium --out sample.ttab
```

Don't have data? Generate a format-faithful fixture for testing:

```python
from histoweave.datasets import write_visium_fixture, write_xenium_fixture
write_visium_fixture("demo_visium")     # minimal Space Ranger tree with 100 spots
write_xenium_fixture("demo_xenium")     # minimal Xenium bundle with 50 cells
```

## 5. Build a custom pipeline

```python
from histoweave import run_pipeline
from histoweave.workflow import PipelineStep

steps = [
    PipelineStep("qc", "basic_qc", {"n_mads": 3.0}),
    PipelineStep("normalization", "log1p_cp10k"),
    PipelineStep("domain_detection", "spectral",
                 {"n_domains": 7, "spatial_weight": 0.3}),
    PipelineStep("annotation", "marker_score"),
]
result = run_pipeline(data, steps)
```

Or step-by-step from the CLI (each step appends to the provenance chain):

```bash
histoweave step --in sample.ttab --category qc --method basic_qc
histoweave step --in sample.ttab --category normalization --method sctransform
histoweave step --in sample.ttab --category domain_detection --method spectral \
    --param n_domains=7 --param spatial_weight=0.3
```

## 6. Benchmark methods

```bash
histoweave benchmark --task domain_detection --out benchmark.json
```

```text
RANK  METHOD              ARI     TIME(s)
----------------------------------------
1     gaussian_mixture    0.999    0.3
2     kmeans              0.991    0.2
3     minibatch_kmeans    0.990    0.2
4     birch               0.989    0.4
5     spectral            0.985    0.6
```

```python
from histoweave.benchmark import domain_detection_task, run_benchmark

result = run_benchmark(domain_detection_task())
_LOGGER.info("%s", result.best())           # the top-ranked method
for row in result.leaderboard:
    _LOGGER.info("%s %s %s", row["rank"], row["method"], row["score"])
```

## 7. Recommend methods — HistoWeave's killer feature

Recommend the best method for new data **without requiring ground truth**. The engine
extracts 16 target-free spatial, expression, and geometry features; finds the nearest
reference datasets in the benchmark landscape; and ranks methods by similarity-weighted
performance.

```bash
# Build the knowledge base once (or download a pre-built one)
histoweave benchmark --out landscape.json

# Recommend for a new sample
histoweave recommend --in new_sample.ttab --knowledge-base landscape.json
```

```text
Recommendation: new_sample [domain_detection]
RANK  METHOD              SCORE    UNCERTAINTY  SUPPORT
----------------------------------------------------------
1     spectral            0.832    0.047        3/3
2     gaussian_mixture    0.815    0.053        2/3
3     kmeans              0.791    0.062        2/3

Nearest evidence: dense_regular (similarity=0.851), small_clean (0.783)
Ensemble suggestion: Run spectral and gaussian_mixture; retain consensus regions,
flag disagreements for review.
```

Machine-readable output for workflow integration:

```bash
histoweave recommend --in sample.ttab --knowledge-base landscape.json \
    --json --out recommendation.json
```

The JSON includes feature vectors, nearest reference datasets, weighted scores,
uncertainty estimates, support/coverage statistics, and ensemble strategy.

Domain labels and cell-type annotations on the query data are **never used** for
retrieval — the recommendation is genuinely target-free.

## 8. Use advanced methods

### Deconvolution with cell2location

```python
from histoweave.plugins import create_method

# Your reference: a genes × cell_types DataFrame stored in data.uns
data.uns["cell2location_reference"] = reference_df

result = create_method(
    "deconvolution", "cell2location",
    max_epochs=30000, n_cells_per_location=8.0,
).run(data)

abundance = result.obsm["cell_abundance"]      # spots × cell types
proportions = result.obsm["proportions"]        # normalised to sum = 1
```

### Cell-cell communication with LIANA+

```python
result = create_method(
    "ccc", "liana_plus",
    groupby="cell_type",              # obs column with cell identities
    resource_name="consensus",        # LIANA ligand-receptor resource
    spatial_weighted=True,            # incorporate spatial proximity
).run(data)

interactions = result.uns["liana_res"]  # ranked LR interactions
```

### Image segmentation with Cellpose 2

```python
result = create_method(
    "segmentation", "cellpose2",
    image_key="HE",                   # image key in data.images
    model_type="cyto2",               # pretrained Cellpose model
    diameter=30.0,                    # expected cell diameter in pixels
).run(data)

masks = result.images["cellpose_masks"]  # integer label image
```

### Cell-type annotation with scANVI or CellTypist

```python
# Semi-supervised with scANVI (partial labels in obs["cell_type_seed"])
result = create_method("annotation", "scanvi",
    labels_key="cell_type_seed",
    unlabeled_category="Unknown",
).run(data)

# Pretrained CellTypist model
result = create_method("annotation", "celltypist",
    model="Immune_All_Low.pkl", majority_voting=True,
).run(data)
```

## 9. Run the Nextflow pipeline (HPC / cloud)

The same analysis runs as a portable Nextflow DAG — identical logic, containerized
execution, resumable at any step:

```bash
# Local execution (needs histoweave on PATH)
nextflow run workflows/nextflow/main.nf -profile local,test

# Containerized (Docker / Singularity)
nextflow run workflows/nextflow/main.nf -profile docker \
    --input /path/to/outs --assay visium --n_domains 8 --outdir results

# HPC (Slurm)
nextflow run workflows/nextflow/main.nf -profile slurm \
    --input /path/to/outs --assay xenium --outdir results
```

Swap `-profile` to run identically from laptop to HPC to cloud.

## 10. Write your own plugin

A method is a class implementing the `Method` interface and declaring a `MethodSpec`.
Register it with `@register`:

```python
from histoweave.plugins import Method, MethodCategory, MethodSpec, ParamSpec, register

@register
class MyDomainDetector(Method):
    spec = MethodSpec(
        name="my_method",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="Wraps <best-in-class method>.",
        params=(ParamSpec("resolution", "float", 1.0, "Clustering resolution."),),
        wraps="my_library",
    )

    def run(self, data):
        data = data.copy()
        # ... call the wrapped library, write results into data.obs ...
        return self.finalize(data)
```

Distribute it as an independently versioned package by advertising on the
`histoweave.plugins` entry-point group. See `plugin-template/` for a copy-paste template.

## Next steps

- [Method selection guide](method-selection.md) - objective-aware method and parameter choices
- [API reference](api.md) — complete method specs, data model, benchmark tasks
- [Architecture](architecture.md) — six-layer design, plugin contract, data flow
- [Concepts](concepts.md) — SpatialTable, provenance, maturity levels
- [Roadmap](roadmap.md) — planned methods, datasets, and features
