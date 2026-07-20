# HistoWeave

**Executable evidence contracts for task-constrained method decisions in spatial
transcriptomics.**

HistoWeave answers one question: given an explicit analysis task and incomplete
benchmark evidence, **which method set is justified, and when should the workflow
fall back or abstain?**

Its core contribution is an evidence-governed decision protocol that:

1. rejects cross-task, circular, proxy-ground-truth, and silent oracle evidence;
2. treats nearest-neighbour recommendation as a candidate-generation proxy, not proof;
3. returns a matched non-dominated method set only after fixed baseline and grouped
   held-out validation gates; and
4. emits `global_default`, `evidence_required`, or `abstain` when those gates fail.

Method wrappers, workflows, reports, Pareto analysis, ISUS, failure fingerprints,
digital twins, and AutoML are supporting infrastructure or evidence producers—not
parallel headline claims. See the [decision protocol](docs/decision-protocol.md)
and [vs Squidpy / SpatialData](docs/vs-squidpy-spatialdata.md).

---

## Try it now

| | |
|--|--|
| **60-second demo** | `pip install "histoweave-spatial[scanpy]"` → `histoweave run --demo --out report.html` |
| **30-min workshop** | Open [`examples/workshop_30min.ipynb`](examples/workshop_30min.ipynb) (install → report → method compare) |
| **中文快速入门** | [`docs/zh/quickstart.md`](docs/zh/quickstart.md) |
| **Validation wall** | [`docs/methods/validation/`](docs/methods/validation/index.md) — **10 scientific** + **3 contract** packages + reference artefacts |
| **SOTA ARI (protocol-bound)** | Always check `k_policy` — Oracle-K and estimate tracks differ (see table below) |

**SOTA DLPFC numbers (must not be mixed across tracks)**

| Method | Track | Mean ARI | Protocol / source |
|--------|-------|---------:|-------------------|
| SpaGCN | **oracle-K** (historical SOTA grid) | ≈0.32 | `5x15_spatial_aware/sota_benchmark_long.csv`, `k_policy` implicit oracle |
| STAGATE | **oracle-K** (subsample max_obs=1000) | ≈0.29 | validation report `stagate.md` |
| GraphST | **oracle-K** | ≈0.12 | same SOTA protocol family |
| SpaGCN | **estimate · silhouette** | ≈0.24 | `non_oracle_k_sota/` dual-track (seed 42) |
| STAGATE | **estimate · silhouette** | ≈0.22 | same |
| SpaGCN | max slice drop oracle→estimate | **0.23** on 151673 | protocol endpoint `oracle_k_leakage` |

**Same-task same-data parallel table** (33 configs × 5 DLPFC slices; sklearn /
spatial-aware / SpaGCN+STAGATE aligned):
[`parallel_experiment_table/`](parallel_experiment_table/) —
see `report_parallel_experiment.md` (protocol
`histoweave.parallel_experiment_table.v1`).

Scientific default for new work is `k_policy=estimate`. Oracle-K is opt-in ablation only.

```bash
pip install "histoweave-spatial[scanpy]"
histoweave run --demo --out report.html
# open report.html — self-contained HTML + provenance
```

**Why teams adopt HistoWeave**

- **One executable evidence contract** for baselines and SOTA
- **Hard semantic gates** block cross-task and Leiden-as-domain-GT evidence
- **Fail-closed** optional backends — no toy substitute when SpaGCN is missing
- **Set-valued decisions** fall back or abstain instead of forcing a winner
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
packages (**13** evidence packages total). The submission-facing core is the
evidence-governed `decide()` protocol; SOTA wrappers, candidate generation,
Vitessce reporting, Nextflow, and containers support that protocol.
See [docs/roadmap.md](docs/roadmap.md).

## Community

- [Contributing guide](CONTRIBUTING.md) — plugins, maturity classification, docs regen
- [Code of Conduct](CODE_OF_CONDUCT.md) — Contributor Covenant 2.1
- [Method catalog](docs/methods/catalog.md) — every registered method documented
- [中文快速入门](docs/zh/quickstart.md) · [30-min workshop](examples/workshop_30min.ipynb)

## Scientific claim (read this first)

| We claim | We do **not** claim |
|----------|---------------------|
| Evidence semantics can be machine-checked before aggregation | Pareto sorting or kNN retrieval is itself a new algorithm |
| Cross-task, circular, and proxy-domain evidence is rejected | Soft down-weighting makes incompatible evidence valid |
| A decision may be a non-dominated set, global fallback, or abstention | A local ranking proves one method is biologically correct |
| Negative held-out results define when personalisation is unsupported | Reference-neighbour fit is independent generalisation evidence |
| ISUS describes label-conditioned spatial information post hoc | ISUS predicts method gain for an unlabelled query |

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

## Core decision protocol

```bash
histoweave decide \
  --in my_sample.ttab \
  --knowledge-base landscape.json \
  --task spatial_domain \
  --dataset-name my_sample \
  --validation benchmark_external_validation/decision_validation.json \
  --pareto-report pareto.json \
  --out decision_card.json --json
```

```python
import histoweave as hw

card = hw.decide(
    data,
    knowledge_base="landscape.json",
    dataset_name="my_sample",
    task="spatial_domain",
    validation=grouped_holdout_summary,
    pareto=pareto_report,
)
print(card.action, card.primary_set, card.comparison_set)
```

The output is a versioned `DecisionCard`, not just a ranking. It records evidence
roles, contract checks, fixed controls, claim boundaries, and one of four actions:
`personalised_set`, `global_default`, `evidence_required`, or `abstain`.
Personalisation requires grouped held-out evidence; a favourable score on the
retrieved reference neighbours is only a proxy. Full specification:
[evidence-governed decision protocol](docs/decision-protocol.md).

**Dry-lab case study (no wet lab):** four unjustified promotions are intercepted
(`evidence_required` / `global_default` / `abstain` / cross-task hard-filter) —

```bash
python examples/case_study_intercepted_recommendation.py
```

See [docs/case-study-intercepted-recommendation.md](docs/case-study-intercepted-recommendation.md).

## Experimental execution adapters (supporting infrastructure)

When a sample has **no ground truth**, these adapters can generate a comparison
panel and post-run diagnostics. They do not validate the winning biology:

```bash
# 1) Feature-matched synthetic twin → predicted ranking from planted-truth ARI
histoweave digital-twin --in my_sample.ttab --out-dir digital_twin_out

# 2) Execution adapter: decision card → comparison panel → Pareto HTML report
histoweave automl "Find spatial domains for my Visium liver cancer data." \
  --in my_sample.ttab \
  --knowledge-base figure3_results/landscape.json \
  --out-dir automl_out --top 3 --platform visium
```

See [digital-twin.md](docs/digital-twin.md) and [spatial-automl.md](docs/spatial-automl.md).

## Synthetic stress tests and evidence acquisition (supporting diagnostics)

```bash
# How each method fails (4-mode fingerprint: frag / merge / noise / structural)
histoweave failure-fingerprint --methods kmeans,spectral --out-dir fingerprints

# When recommend does not beat global-best: evidence-acquisition todo list
histoweave calibrate-recommender --in sample.ttab \
  --knowledge-base figure3_results/landscape.json --out calibration.json
```

## Reference-neighbour candidate generator (not the decision product)

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

The engine extracts target-free features, hard-filters incompatible task/label
semantics, applies a platform prior, and ranks `method` or `method@policy`
configurations. Its global-best comparison is computed on retrieved reference
neighbours, so it is a **proxy diagnostic rather than independent validation**.
Use `histoweave decide` for an actionable, fail-closed decision.

## Set-valued output layer: canonical Pareto analysis

"Which method is best?" is the wrong question when methods differ on axes that
cannot be collapsed into one number. The Pareto layer records every
configuration on up to **four objectives** — accuracy (ARI, maximise), speed
(seconds, minimise), memory (GB, minimise) and robustness (bootstrap ARI
CI-width, minimise) — and reports the **non-dominated frontier**: the set of
configs that no other config beats on *every* axis.  Nothing on the frontier is
strictly worse than anything else; picking among them is a value judgement the
tool makes explicit rather than hiding behind a mean. Pareto sorting is an
established operator; the contribution is enforcing matched evidence and using
the set within an explicit fallback/abstention protocol.

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

## Label-conditioned spatial information audit (post hoc only)

The **Information-theoretic Spatial Utility Score (ISUS)** describes how much
trusted domain labels depend on coordinates beyond expression. Because it
requires those labels, it cannot decide whether an unlabelled query should use
a spatial method before methods are run:

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

# Permutation null: p-value + Z-score; primary band uses null evidence (not 0.1/0.3)
histoweave isus --in my_sample --domain-key domain_truth --n-null 99 --out isus.json

# Bind ISUS to observed spatial ARI gain from benchmark_long.csv (gain map + audit)
histoweave isus --calibrate 5x15_spatial_aware --out isus_calibration.json

# Attach a previously fitted gain map to a new labelled sample
histoweave isus --in my_sample --gain-calibration isus_calibration.json --out isus.json
```

```python
from histoweave.benchmark import (
    assess_isus_predictor,
    attach_gain_prediction,
    compute_isus_from_table,
    extract_spatial_ari_gains_from_long,
    fit_isus_gain_calibration,
)

res = compute_isus_from_table(data, domain_key="domain_truth", n_null=99)
print(res.isus, res.band, res.p_value_i_d_s_given_e, res.z_score_i_d_s_given_e)
# band_source == "permutation_z" when n_null>0; band_heuristic still reports 0.1/0.3

gains = extract_spatial_ari_gains_from_long("5x15_spatial_aware/benchmark_long.csv")
calib = fit_isus_gain_calibration(per_slice_records)  # each row: isus + spatial_ari_gain
print(calib.slope, calib.reliability, calib.loo_rmse)
res = attach_gain_prediction(res, calib)
print(res.expected_spatial_ari_gain, res.gain_prediction_reliability)
```

**Bands and thresholds.** With ``n_null=0``, bands still use the legacy absolute
cut-offs (`ISUS_LOW=0.1`, `ISUS_HIGH=0.3`) and are flagged as subjective. With
``n_null>0``, the **primary** band comes from the coordinate-shuffle null:
not significant → `not_above_null`; significant with Z < 3 →
`modest-spatial-signal`; Z ≥ 3 → `spatial-critical`. Dataset-specific absolute
thresholds (`threshold_significant_isus`, `threshold_critical_isus`) are the
null quantile / mean+3σ on the ISUS scale for that sample; the 0.1/0.3 values
remain only under `band_heuristic`.

**Downstream gain map.** ``--calibrate`` fits
`spatial_ari_gain ≈ intercept + slope * ISUS` against `benchmark_long.csv`
(per dataset: mean over methods of best `sw>0` ARI minus `sw0.0`), with LOO
RMSE intervals and an explicit reliability flag from the Spearman audit. On the
5×15 DLPFC benchmark the map is **unsupported/low-reliability** (ρ ≈ −0.30,
n = 5): ISUS does not track realised spatial-weighting ARI gain. Treat expected
gain as an exploratory post-hoc binding, never as a pre-execution gate for an
unlabelled query.

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
