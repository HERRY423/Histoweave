# Concepts

## The canonical object: `SpatialTable`

Everything in HistoWeave flows through one object. It mirrors AnnData:

| Attribute | Holds |
|-----------|-------|
| `X` | cell/spot × gene matrix |
| `obs` | per-observation metadata (QC metrics, domain, cell_type, ground truth) |
| `var` | per-gene metadata (e.g. `mito` flag) |
| `obsm` | multi-dim per-observation arrays (`spatial` coordinates, `X_pca`) |
| `uns` | unstructured annotations, including the `provenance` chain and `run_manifest` |

Real deployments back this with SpatialData/OME-Zarr and lazy Dask chunks; the scaffold
uses a dense in-memory implementation with an `to_anndata()` / `from_anndata()` bridge.

## Methods & the plugin contract

A method implements the `Method` interface and declares a `MethodSpec`:

```python
from histoweave.plugins import Method, MethodCategory, MethodSpec, ParamSpec, register

@register
class MyQC(Method):
    spec = MethodSpec(
        name="my_qc",
        category=MethodCategory.QC,
        version="0.1.0",
        summary="What it does.",
        params=(ParamSpec("threshold", "float", 1.0, "..."),),
        wraps="scanpy",           # provenance for what's under the hood
        language="python",        # or "r" / "container"
        maturity="beta",         # experimental → beta → production → contract_validated → validated
        model_family="deep_learning",
        modalities=("expression", "image"),
    )

    def run(self, data):
        data = data.copy()        # inputs are immutable
        # ... transform, write into data.obs / data.obsm ...
        return self.finalize(data)  # stamp provenance
```

**Categories** map onto the analysis stages in the plan: `qc`, `normalization`,
`segmentation`, `annotation`, `domain_detection`, `deconvolution`, `svg`, `neighborhood`,
`ccc`, `integration`, `ingestion`.

## The registry & method selection

The registry is the machine-readable catalogue behind `histoweave list-methods` and the
leaderboards. It answers *"what methods exist for domain detection, and which lead for a
Xenium dataset of this size?"* External plugin packages register by advertising a hook on
the `histoweave.plugins` entry-point group.

Capability and release coverage are queryable rather than inferred from method names:

```python
from histoweave.plugins import method_coverage_report

coverage = method_coverage_report()
assert coverage["total_methods"] >= 50
assert coverage["counts"]["unclassified_methods"] == 0
assert coverage["unclassified_names"] == []
assert coverage["ratios"]["beta_plus"] == 1.0
assert coverage["ratios"]["production_plus"] > 0.60
```

The same registry metadata identifies deep-learning methods and methods that consume
registered image plus expression inputs.

## Provenance & reproducibility

Every transformation appends a `Provenance` record (method, version, parameters, histoweave
version, container digest, timestamp) to `uns['provenance']`. A full pipeline additionally
stores a `RunManifest` in `uns['run_manifest']`. Together they make any result auditable
and re-runnable — reproducibility is a first-class output, not a byproduct.

## Benchmarking

A `Task` bundles a method category, a dataset with ground truth, and a scoring function.
`run_benchmark` evaluates every registered method for that category and returns a
best-first leaderboard. Because a failing method scores worst rather than crashing the
run, the harness is safe to run continuously in CI and to gate releases against
performance regressions.

## Digital-twin synthetic validation

Real user uploads usually lack ground truth.  HistoWeave builds a **digital twin**: a
synthetic sample that matches the real data on 13 target-free dimensions (sparsity,
library-size distribution, Moran's I, Hopkins tendency, effective rank, …) while
planting known domain labels.  Methods are benchmarked on the twin; the ARI ranking is
returned as a **predicted ranking** for the real sample.  See [Digital-twin
validation](digital-twin.md).

## Spatial AutoML compiler

The AutoML compiler combines `histoweave ask` (LLM pipeline compiler) with the landscape
recommender: extract features → retrieve nearest reference datasets → auto-run top-3
methods → multi-objective (Pareto) comparison → full HTML report.  See [Spatial AutoML
compiler](spatial-automl.md).

## Failure fingerprint atlas

Beyond *where* a method fails (boundary mapping), HistoWeave classifies *how* it fails
using contingency structure (ARI-related, label-permutation invariant): fragmentation,
merge, noise micro-clusters, and structural collapse on hard data.  Each method gets a
4-vector fingerprint.  See [Failure fingerprints](failure-fingerprints.md).

## Active recommender calibration

When the recommender does not beat the global-best baseline, HistoWeave proposes a
prioritised **evidence-acquisition todo** of dataset×method pairs that maximise expected
information gain — a concrete plan to reduce recommendation uncertainty.  See
[Active calibration](active-calibration.md).
