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
| **Validation wall** | [`docs/methods/validation/`](docs/methods/validation/index.md) — **10 scientific** + **3 contract** multi-dataset packages (**13** total) |
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

**v0.1.0** — submission freeze. Plugin registry with maturity tiers
(`experimental` → `beta` → `production` → `contract_validated` → `validated`):
**10 scientifically validated** methods and **3 contract-validated** multi-dataset
packages (**13** evidence packages total). First-class SOTA domain plugins
(SpaGCN / GraphST / STAGATE / BayesSpace / RCTD), task-bound recommendation engine
v2, Vitessce reporting, Nextflow + containers. Optional heavy backends fail closed.
See [docs/roadmap.md](docs/roadmap.md).

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

## Digital-twin validation & Spatial AutoML

When your sample has **no ground truth**, HistoWeave can still rank methods:

```bash
# 1) Feature-matched synthetic twin → predicted ranking from planted-truth ARI
histoweave digital-twin --in my_sample.ttab --out-dir digital_twin_out

# 2) Full AutoML: recommend → run top-3 → Pareto HTML report
histoweave automl "Find spatial domains for my Visium liver cancer data." \
  --in my_sample.ttab \
  --knowledge-base figure3_results/landscape.json \
  --out-dir automl_out --top 3 --platform visium
```

See [digital-twin.md](docs/digital-twin.md) and [spatial-automl.md](docs/spatial-automl.md).

## Failure fingerprints & recommender calibration

```bash
# How each method fails (4-mode fingerprint: frag / merge / noise / structural)
histoweave failure-fingerprint --methods kmeans,spectral --out-dir fingerprints

# When recommend does not beat global-best: evidence-acquisition todo list
histoweave calibrate-recommender --in sample.ttab \
  --knowledge-base figure3_results/landscape.json --out calibration.json
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

## Pareto frontier — trade-offs, not a single winner

"Which method is best?" is the wrong question when methods differ on axes that
cannot be collapsed into one number.  The Pareto recommender scores every
configuration on up to **four objectives** — accuracy (ARI, maximise), speed
(seconds, minimise), memory (GB, minimise) and robustness (bootstrap ARI
CI-width, minimise) — and reports the **non-dominated frontier**: the set of
configs that no other config beats on *every* axis.  Nothing on the frontier is
strictly worse than anything else; picking among them is a value judgement the
tool makes explicit rather than hiding behind a mean.

```bash
# Per-seed ARI + timings (enables the robustness axis); memory from the
# scalability study. Writes a report + a per-dataset SVG.
histoweave pareto \
    --benchmark-long 5x15_spatial_aware/benchmark_long.csv \
    --scaling-dir scalability_proof \
    --dataset 151673 --svg pareto_151673.svg --out pareto.json

# Accuracy + timings only (no per-seed data -> no robustness axis)
histoweave pareto --landscape figure3_results/landscape.json --json
```

```python
from histoweave.benchmark import (
    objective_tables_from_long_csv, analyze_dataset, build_report, pareto_svg,
)

tables = objective_tables_from_long_csv(
    "5x15_spatial_aware/benchmark_long.csv", scaling_dir="scalability_proof",
)
res = analyze_dataset(next(t for t in tables if t.dataset == "151673"))
print(res.frontier)                               # non-dominated configs
print(res.knee)                                    # balanced compromise pick
open("pareto_151673.svg", "w").write(pareto_svg(res))

report = build_report(tables)                      # NaN-safe .to_dict() for JSON
report.datasets["151673"]["frontier"]             # dict form for serialisation
```

The **knee** is a convenience pick — the frontier point closest to the ideal
corner in normalised objective space — but the frontier itself is the product.
On the bundled DLPFC slices the frontier is genuinely multi-config (5–9 of 15
per slice), and `agglomerative@sw0.0`/`agglomerative@sw0.8` sit on it for all
five slices, which single-winner reporting would hide.

## ISUS — should this dataset use a spatial method at all?

The recommender picks *which* method; the **Information-theoretic Spatial
Utility Score (ISUS)** answers the prior question — *whether* spatial
coordinates help this dataset at all — before any method is run:

```
ISUS = I(D; S | E) / I(D; E)
```

where `D` are domain labels, `S` spatial coordinates and `E` expression.  The
numerator is the domain information that space adds **beyond** expression
(conditional mutual information); the denominator normalises by what expression
already provides.  Both terms use the Ross (2014) kNN estimator on NumPy/SciPy
(no scikit-learn dependency).

```bash
# From a bundle / dataset with expression + spatial coords + domain labels
histoweave isus --in my_sample --domain-key domain_truth --out isus.json

# Calibrate ISUS against observed spatial ARI gain over a benchmark dir
histoweave isus --calibrate 5x15_spatial_aware --out isus_calibration.json
```

```python
from histoweave.benchmark import compute_isus_from_table, isus_band
res = compute_isus_from_table(data, domain_key="domain_truth")
print(res.isus, res.band)      # e.g. 0.077 "expression-sufficient"
```

**Interpretation bands (provisional heuristics, not a validated predictor).**
`ISUS < 0.1` — expression alone is largely sufficient; `0.1–0.3` — modest
spatial signal; `> 0.3` — spatial structure is a large fraction of the domain
information.  A coordinate-shuffle control collapses ISUS to ~0, confirming it
measures *genuine spatial structure* rather than an artifact.  **However**, on
the 5×15 DLPFC benchmark ISUS does **not** correlate with the ARI improvement
that spatial-weighting actually delivers (Spearman ρ ≈ −0.30, n = 5, not
significant): every DLPFC layer is spatially contiguous, so the numerator is
high across the board, while realised ARI gain is driven by expression
separability and the specific smoothing mechanism.  Treat ISUS as a validated
*descriptor* of spatial information content, and the bands as a starting point
to calibrate on your own data — not as a promise about a given method's gain.

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

Maturity tiers: `experimental` → `beta` → `production` → `contract_validated` → `validated`.
**validated** = multi-dataset *scientific* concordance (10 methods).  
**contract_validated** = multi-dataset *interface/mock* gates (3 methods).  
Together: **13** multi-dataset evidence packages (do not conflate the two kinds).

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
