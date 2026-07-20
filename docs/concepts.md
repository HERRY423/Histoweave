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

## Evidence-governed decisions

`histoweave.decide()` is the only high-level method-selection entry point. It
converts benchmark evidence into a versioned `DecisionCard` after hard task and
ground-truth checks, a global-default comparison, grouped held-out validation,
and an optional matched Pareto intersection. Its result is a method set,
fallback, evidence request, or abstention—not an asserted universal winner.

See [Evidence-governed decision protocol](decision-protocol.md).

## Multimodal tasks and virtual ST

Analysis tasks are expanded beyond RNA spatial domains:

* `spatial_protein_domain` / `spatial_chromatin_domain` — domain partitions on other
  molecular modalities (exact-match evidence only; never soft-mixed with RNA domains).
* `virtual_st` — predict spatial expression from registered H&E; score with
  mean gene Pearson against **measured** expression.

Cross-modal compatibility is an executable gate (`tasks_admissible`,
`cross_modal_relation`). Real H&E paths include
`load_visium_hne_paired()` and Space Ranger `tissue_*_image.png` ingestion.
See [Multimodal tasks & virtual ST](multimodal-virtual-st.md) and
[Tutorial 6](tutorials/06_virtual_st_he.md).

## Digital-twin synthetic validation (supporting diagnostic)

Real user uploads usually lack ground truth.  HistoWeave builds a **digital twin**: a
synthetic sample that matches the real data on 13 target-free dimensions (sparsity,
library-size distribution, Moran's I, Hopkins tendency, effective rank, …) while
planting known domain labels. Methods are benchmarked on the twin; the result is
an experimental synthetic proxy, not validated ranking transfer to the real sample. See [Digital-twin
validation](digital-twin.md).

## Spatial AutoML execution adapter

The AutoML compiler combines `histoweave ask` (LLM pipeline compiler) with the landscape
evidence layer: extract features → admit compatible references → fallback/compare/abstain
→ execute an allowed panel → canonical multi-objective comparison → HTML report. It emits
the same `decision_card.json`; it is an execution adapter, not a parallel scientific
selector. See [Spatial AutoML compiler](spatial-automl.md).

## Failure fingerprint atlas (synthetic stress-test evidence)

Beyond *where* a method fails (boundary mapping), HistoWeave classifies *how* it fails
using contingency structure (ARI-related, label-permutation invariant): fragmentation,
merge, noise micro-clusters, and structural collapse on hard data.  Each method gets a
4-vector fingerprint.  See [Failure fingerprints](failure-fingerprints.md).

## Active evidence acquisition heuristic

When the reference-neighbour proxy does not beat the global-best baseline, HistoWeave
proposes a prioritised **evidence-acquisition todo** of dataset×method pairs. Its score is
a practical priority heuristic, not calibrated expected information gain. See
[Active calibration](active-calibration.md).
