# HistoWeave

**Open-source orchestration & evaluation for reproducible spatial transcriptomics.**

HistoWeave is the connective tissue the spatial field still lacks: a unified
**distribution, orchestration, and evaluation layer** on top of scverse and
Bioconductor. It does **not** claim a universal best method. Instead it:

1. wraps field-standard and baseline methods behind one plugin contract;
2. benchmarks them under **task contracts** (spatial-domain vs cell-type, never mixed);
3. quantifies **method × spatial-context selection uncertainty**;
4. recommends configurations with explicit baselines, priors, and failure warnings;
5. produces interactive reports for multi-method review (Vitessce).

---

## Try it now

| | |
|--|--|
| **60-second demo** | `pip install "histoweave-spatial[scanpy]"` → `histoweave run --demo --out report.html` |
| **30-min workshop** | Open [`examples/workshop_30min.ipynb`](examples/workshop_30min.ipynb) (install → report → method compare) |
| **中文快速入门** | [`docs/zh/quickstart.md`](docs/zh/quickstart.md) |
| **Validation wall** | [`docs/methods/validation/`](docs/methods/validation/index.md) — multi-dataset evidence for **13 validated** methods |
| **Real SOTA ARI** | SpaGCN ≈0.32 · STAGATE ≈0.29 · GraphST ≈0.12 on the same DLPFC protocol |

```bash
pip install "histoweave-spatial[scanpy]"
histoweave run --demo --out report.html
# open report.html — self-contained HTML + provenance
```

**Why teams adopt HistoWeave**

- **One contract** for baselines and SOTA (SpaGCN / GraphST / STAGATE / RCTD / …)
- **Task contracts** block silent errors (no Leiden-as-domain-GT)
- **Fail-closed** optional backends — no toy substitute when SpaGCN is missing
- **Honest recommenders** report when they fail to beat a global-best baseline
- **Report-first** onboarding: every first run ends in a shareable HTML artifact

**Choose your path**

| You are… | Start here |
|----------|------------|
| New user / student | Demo above → [workshop notebook](examples/workshop_30min.ipynb) |
| Visium wet-lab | [Quickstart § real data](docs/quickstart.md) · [中文](docs/zh/quickstart.md) |
| Method author | [CONTRIBUTING](CONTRIBUTING.md) · maturity + `VALIDATION_EVIDENCE` |
| Benchmarking / SOTA | [validation reports](docs/methods/validation/index.md) · `research/method_validation/` |

---

## Status

**v0.1.0-beta** — plugin registry with maturity tiers (including **13 validated** methods),
first-class SOTA domain plugins (SpaGCN / GraphST / STAGATE / BayesSpace / RCTD),
task-bound recommendation engine v2, O(n log n) spatial kNN, Vitessce reporting,
Nextflow + containers.  Optional heavy backends fail closed (no silent toy
substitutes).  See [docs/roadmap.md](docs/roadmap.md).

## Community

- [Contributing guide](CONTRIBUTING.md) — plugins, maturity classification, docs regen
- [Code of Conduct](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- [Method catalog](docs/methods/catalog.md) — every registered method documented
- [中文快速入门](docs/zh/quickstart.md) · [30-min workshop](examples/workshop_30min.ipynb)

## Scientific claim (read this first)

| We claim | We do **not** claim |
|----------|---------------------|
| Method choice and spatial-context policy jointly dominate many analysis outcomes | One recommender always picks the single best method |
| Multi-method disagreement maps prioritise hard boundaries for review | Cross-platform ARI is comparable when ground-truth semantics differ |
| Task-separated landscapes reduce silent scientific errors | Leiden/self-clustering is valid domain ground truth |
| Negative recommender results (fails to beat global-best) are useful diagnostics | Top-1 accuracy on a 5-slice DLPFC LOOCV is a generalisation proof |

## Install

```bash
# PyPI (core — NumPy + pandas + Jinja2 + SpatialData, runs anywhere)
pip install histoweave-spatial

# With real-data ingestion (10x H5, Parquet, spatialdata-io)
pip install "histoweave-spatial[io,spatial]"

# Method-specific extras
pip install "histoweave-spatial[scanpy,cell2location,scanvi,cellpose2,spatialde,liana,celltypist,deep-learning]"

# All extras + development tools
pip install "histoweave-spatial[all]"
```

> Python **3.11+** required.

## Quick start — 60 seconds to a report

```python
import histoweave as ts

# Synthetic demo: 600 cells, 3 ground-truth domains
data = ts.datasets.make_synthetic(seed=0)
result = ts.run_pipeline(data)          # QC → normalize → kmeans → annotate
ts.build_report(result, "report.html")  # Self-contained HTML with Vitessce
```

```bash
histoweave run --demo --out report.html
```

## Method recommendation v2

```bash
# Landscape knowledge base (task = spatial_domain)
histoweave recommend --in my_sample.ttab \
    --knowledge-base figure3_results/landscape.json

# Prefer high spatial context for domain recovery
histoweave recommend --in my_sample.ttab \
    --knowledge-base figure3_results/landscape.json \
    --json --out recommendation.json
```

```python
from histoweave.benchmark import MethodRecommender, AnalysisTask

rec = MethodRecommender("figure3_results/landscape.json").recommend(
    data,
    task=AnalysisTask.SPATIAL_DOMAIN,
    platform="visium",
    spatial_context_policy="high",
)
print(rec.summary())
# Inspect rec.beats_global_best_baseline and rec.warnings before acting.
```

The engine extracts target-free features, applies **task + platform priors**, ranks
`method` or `method@policy` configurations, and reports regret against a
**global-best baseline**.  If personalisation does not beat that baseline, the
API says so.

## Task contracts (hard rules)

```python
from histoweave.benchmark import AnalysisTask, GroundTruthKind, TaskContract

# Valid: expert cortical layers for domain recovery
TaskContract(
    task=AnalysisTask.SPATIAL_DOMAIN,
    ground_truth_kind=GroundTruthKind.SPATIAL_DOMAIN,
    label_key="domain_truth",
    platform="visium",
).validate()

# Invalid: Leiden-as-domain-GT is rejected
TaskContract(
    task=AnalysisTask.SPATIAL_DOMAIN,
    ground_truth_kind=GroundTruthKind.SELF_SUPERVISED,
    label_key="leiden",
).validate()  # raises ValueError
```

## Phenomenology capability benchmark

The primary evaluation framework is a phenomenon-centred capability matrix rather
than a cross-task leaderboard. It covers six spatial phenomena, five observation
conditions, 54 frozen methods (all validated/production/beta releases plus the
dependency-light `marker_deconv` baseline), and five paired replicates. Applicability,
backend gaps, resource failures, and scientific failures remain separate, and summaries
are conditioned on method role/category instead of producing a misleading global rank.

```bash
# Freeze the full design without running methods
histoweave benchmark --suite phenomenology --dry-run \
    --out-dir phenomenology_plan

# Run a dependency-light smoke study
histoweave benchmark --suite phenomenology --tiny \
    --phenomena compartment --conditions clean \
    --methods basic_qc,log1p_cp10k,kmeans \
    --out-dir phenomenology_smoke
```

See the [phenomenology benchmark guide](docs/phenomenology-benchmark.md) for the
frozen applicability contracts, metrics, uncertainty analysis, and failure semantics.
## What's in the box

| Category | Highlights |
|----------|------------|
| **Ingestion** | Visium, Xenium, CosMx, MERSCOPE, MERFISH, Slide-seq, Stereo-seq |
| **Domain detection** | BANKSY (R + native), SpaGCN, GraphST, STAGATE, BayesSpace, sklearn family |
| **Deconvolution** | cell2location, RCTD (R bridge), marker baseline |
| **SVG** | SpatialDE, nnSVG, Moran's I / Geary's C |
| **Annotation / CCC / Seg** | scANVI, CellTypist, LIANA+, Cellpose 2 |
| **Research incubator** | `weave_*` experimental candidates (explicitly unvalidated) |

Maturity tiers: `experimental` → `beta` → `production` → `validated`.
Only methods with multi-dataset concordance evidence are marked **validated**.

## Architecture

```
 6 · Reporting        HTML + Vitessce interactive viewer
 5 · Benchmarking     Task contracts, landscapes, recommender v2, uncertainty maps
 4 · Methods          Typed plugins (baselines + SOTA + research incubator)
 3 · Workflow         In-process SDK + Nextflow DAG (Docker/Slurm/K8s)
 2 · Data             SpatialTable + sparse + SpatialData/AnnData bridge
 1 · Ingestion        Native 10x H5 readers + spatialdata-io adapters
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

## Real-data benchmarks

Reproducible DLPFC landscapes and reports live under:

- [5x10_dlpfc_benchmark](5x10_dlpfc_benchmark/report_5x10_dlpfc_benchmark.md)
- [5x15_spatial_aware](5x15_spatial_aware/report_5x15_spatial_aware.md)
- [7x15_cross_platform](7x15_cross_platform/report_7x15_cross_platform.md) — interpret with the proxy-label caveats in the report

Set `HISTOWEAVE_DLPFC_DATA` / `HISTOWEAVE_BENCHMARK_OUT` to redirect large artifacts.
