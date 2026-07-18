# Changelog

All notable changes to this project will be documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] — 2026-07-18

### Submission freeze

- Version **0.1.0** (was `0.1.0b1`): package, `CITATION.cff`, and Git tag `v0.1.0` aligned.
- **Validation ledger unified:** **10** scientifically `validated` methods + **3**
  `contract_validated` multi-dataset packages = **13** evidence packages total.
  Mock/interface gates no longer inflate the scientific `validated` count.
- New maturity tier `contract_validated` (rank between production and validated).
- Zenodo metadata (`.zenodo.json`) for DOI minting on GitHub Release.
- Release notes: `RELEASE_NOTES_v0.1.0.md`.

### Method failure fingerprint atlas

- Added `histoweave.benchmark.failure_fingerprint` — classifies *how* methods fail
  (fragmentation / merge / noise / structural) via contingency structure between planted
  truth and predictions (label-permutation invariant).
- Each method receives a 4-vector fingerprint describing degradation near its failure
  boundary; CLI `histoweave failure-fingerprint` and optional attachment to
  `benchmark-boundary` (disable with `--no-fingerprints`).
- Docs: `docs/failure-fingerprints.md`.

### Active-learning recommender calibration

- When `beats_global_best_baseline=False`, `MethodRecommender.recommend()` attaches an
  **evidence-acquisition todo**: dataset×method pairs ranked by expected information
  gain (similarity × importance × novelty × decision relevance).
- CLI: `histoweave calibrate-recommender`; also printed under `histoweave recommend`.
- Docs: `docs/active-calibration.md`.

### Digital-twin synthetic validation

- Added `histoweave.datasets.make_digital_twin` — builds a synthetic twin that matches
  a real sample on **13 target-free dimensions** (sparsity, library-size stats, Moran's I,
  Hopkins tendency, effective rank, …) while planting known domain labels.
- Added `histoweave.benchmark.run_digital_twin_validation` — benchmarks methods on the
  twin and returns the ranking as a **predicted ranking** for unlabelled real data.
- CLI: `histoweave digital-twin --in data.ttab --out-dir DIR` writes JSON artifacts and
  `digital_twin_report.html`.
- Docs: `docs/digital-twin.md`.

### Spatial AutoML compiler

- Added `histoweave.automl.run_spatial_automl` — combines the NL compiler (`histoweave ask`)
  with the landscape recommender: feature extract → k-NN retrieval → auto-run top-3
  methods → multi-objective **Pareto** ranking → full HTML report.
- CLI: `histoweave automl "Find spatial domains …" --in data.ttab --knowledge-base KB.json`.
- Docs: `docs/spatial-automl.md`.

### External validation landscape (5 datasets x 15 methods)

- Added `benchmark_external_validation/`, a cross-study benchmark spanning Visium HD,
  Xenium, Xenium Prime, Visium v2, and MERFISH datasets from human colorectal, lung,
  ovarian cancer and mouse brain.
- Added preparers for all five datasets with strict anatomical/pathology/manual
  `domain_truth`, plus a shared Xenium pathology-polygon assignment helper.
- Added a 5 x 15 x 3-seed experiment driver, bootstrap confidence intervals,
  leave-one-dataset-out recommender validation, and four publication-ready figures.
- Registered `visium_hd_crc`, `xenium_lung_cancer`, `xenium_ovarian_cancer`,
  `visium_mouse_brain`, and `allen_merfish_brain_section` in the real-data registry.
- Archived results: 0.80 top-1/top-3 accuracy and 0.0059 mean regret, tying rather
  than beating the global-best baseline.
- Updated the method-selection guide and leaderboard generator.

### Discovery track: multi-method uncertainty niches

- Added `research/discovery_uncertainty_niches/` — end-to-end biological discovery
  pipeline using HistoWeave tools (non-oracle *K*, multi-method domains, boundary
  uncertainty, Moran SVG + FDR, spatial-shift nulls, cryptic-component geometry)
  on three DLPFC donor slices. Artifacts and honest GO/NO-GO report under
  `DISCOVERY_REPORT.md`.
- Component deep-dives (L6 n=154, L3 n=138 on 151508; L3 n=137 on 151669) with
  layer adjacency + DE; pre-registered **ENC1/HOPX/MBP** panel validation,
  cross-donor direction gates, IF ROI export, and `IF_PROTOCOL.md` hand-off.
- Full **12-section DLPFC cohort** (`run_cohort_panel.py`): pure L3/L6 cryptic
  niches + panel meta-analysis — L3 direction **14/15**, hard same-layer **0/15**;
  frozen status in `PROJECT_STATUS.md` / `COHORT_META_REPORT.md`.
- **Donor-stratified bootstrap CIs** for direction_ok L3 components
  (`histoweave.benchmark.donor_bootstrap` / `histoweave discovery bootstrap-ci`):
  L3 Δrest 0.29 [0.22, 0.34], myelin −0.35 [−0.38, −0.32] (both exclude 0).
- CLI: `histoweave stats-review` (landscape rank/FDR review),
  `histoweave discovery {run,cohort,bootstrap-ci,panel,if-package,if-analyze}`,
  `python -m histoweave` entry via `__main__.py` (sota already present).
- **IF validation path** for top niches (151508 L3+L6, optional 151669 L3):
  lab package (`prepare_if_lab_package.py`), claim ladder, and
  `analyze_if_return.py` to upgrade narrative when protein gates pass.
- **Second tissue context — Xenium lymph node:**
  `research/discovery_xenium_lymph/` applies the same cryptic-niche pipeline
  (uncertainty → components → B/T/GC lymphoid panels) outside DLPFC;
  CLI: `histoweave discovery xenium-lymph`.
- **Official Xenium matrix swap + reassess:** download CDN v3.0.0
  `cell_feature_matrix.h5` + `cells.csv.gz`; GeoJSON auto-calibrate;
  rare-domain stratified floor; full pathology names; `swap_and_rerun.py`;
  synthetic→official AUROC/panel Δ table (`OFFICIAL_SWAP_COMPARISON.md`).
- **GC-enriched deep-dive (DLPFC-parity):** `analyze_gc_components.py` —
  adjacency + DE vs rest/same-domain/abutting; CLI
  `histoweave discovery xenium-lymph --swap-official|--gc-deep-dive`.
- **Multi-dataset validation expansion (5 methods):** formal reports under
  `docs/methods/validation/` for `agglomerative`, `birch`, `minibatch_kmeans`,
  `banksy`, `cell2location`; protocol
  `histoweave.method_validation.multidataset.v1`; compiler
  `research/method_validation/compile_validation_evidence.py`; promoted to
  `VALIDATION_EVIDENCE` / `MethodMaturity.VALIDATED` (8 validated total).
- **SOTA validation batch (5 methods):** `spagcn` (real DLPFC multi-slice ARI),
  `graphst` / `stagate` (structural multi-dataset), `rctd` (fail-closed),
  `spatialde` (SVG multi-dataset); runner
  `research/method_validation/run_sota_batch_multidataset.py`; protocol
  `histoweave.method_validation.sota_batch.v1` → **13 validated** total.
- **Real GraphST ARI:** fixed adapter import (`GraphST.GraphST.GraphST`);
  official multi-slice DLPFC grid 5×3 seeds (max_obs=1000, epochs=120),
  mean ARI≈0.121 (15/15 success); runner
  `research/method_validation/run_real_graphst_stagate_ari.py`.
- **Onboarding pack:** Chinese quickstart (`docs/zh/quickstart.md`), README
  acquisition block (Try it now / paths / validation wall), and one-click
  30-minute workshop notebook (`examples/workshop_30min.ipynb`).

### Method docs coverage, community guides, unclassified-method gate

- Method guides now cover **all registered methods** (not just 5 deep pages):
  generated catalog (`docs/methods/catalog.md`), 11 category guides, and one
  generated page per method under `docs/methods/generated/` via
  `scripts/generate_method_docs.py`. Deep hand-written guides retained for
  banksy_py / spectral / gaussian_mixture / spagcn / cell2location.
- Expanded [CONTRIBUTING.md](CONTRIBUTING.md) (plugin checklist, maturity
  classification, docs regen, PR checklist) and full Contributor Covenant 2.1
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md); docs-site mirrors under `docs/`.
- `method_coverage_report()` now lists **tracks** (production / beta / validated /
  research / baseline), surfaces `unclassified_names`, and fails the release
  gate when any method is unclassified (`unclassified_is_zero`).

### Statistical review, non-oracle K, FDR control

- Added an independent **statistical review layer**
  (`benchmark/stats_review.py`, `benchmark/multiple_testing.py`):
  cell-bootstrap ARI CIs, dataset-bootstrap rank stability, Dirichlet–multinomial
  Bayesian rank posteriors, paired permutation tests, and BH/BY/Holm/Bonferroni
  FDR adjustment via `review_landscape()`.
- **Oracle-K is no longer the default.** `run_landscape` / `run_benchmark` default
  to `k_policy="estimate"` (silhouette / BIC-GMM / gap in
  `benchmark/k_selection.py`). Oracle injection requires
  `allow_oracle_k=True` (and `TaskContract` notes documenting the ablation).
- Moran's I SVG now emits `morans_i_pval` / `morans_i_padj` with Benjamini–Hochberg
  FDR and reports `n_significant_fdr` in `uns['svg']`.
- CLI: `histoweave benchmark --stats --k-policy estimate|oracle|fixed
  --allow-oracle-k --n-boot N`.
- Docs: `docs/statistical-review.md`. CI splits oracle capacity gate from
  non-oracle + stats smoke.

### Quality gates: mypy, Hypothesis, performance CI, test inventory

- Strengthened mypy: `follow_imports=silent`, global `ignore_missing_imports=false`,
  `check_untyped_defs`, `no_implicit_optional`, explicit third-party overrides
  (global silence removed; science deps overridden per-module), plus
  `pandas-stubs` / `types-psutil` in the dev extra.
- Added Hypothesis property tests for `_math`, task contracts, and sparse
  `SpatialTable` construction (`pytest -m property`).
- Added performance regression baselines (`tests/perf_baselines.json`) and CI job
  steps for micro-benchmark ceilings (kNN / z-score / PCA / feature extraction),
  independent of ARI scientific gates.
- Raised the test/source inventory floor to ≥0.80 with meta-tests in
  `tests/test_quality_inventory.py` and expanded unit/smoke modules.

### P2 SOTA reproduction, large-imaging scale, leaderboard filters

- Added `benchmark/sota_pipeline.py` + `histoweave sota` / `scripts/run_sota_dlpfc.py`
  for probe/dry-run/run of SpaGCN, GraphST, STAGATE, BayesSpace, and `banksy_py`,
  emitting `sota_benchmark_long.csv`, `sota_throughput.json`, and `sota_probe.json`.
- Added environment contract YAML, Nextflow `workflows/nextflow/sota.nf`, and
  operator docs (`docs/sota-reproduction.md`).
- Added scale contracts for 10⁵-cell imaging tables
  (`datasets/scale_contract.py`) and tutorial
  `docs/tutorials/05_large_imaging_scale.md`.
- Leaderboard UI: filter by **family=sota** and **task** (domain vs cell_type),
  with self-supervised labels excluded from the domain board.

### P1 landscape merge, registry contracts, uncertainty report

- Added `benchmark/landscape_io.py` to import/merge long CSVs (baseline + SOTA),
  attach `dataset_meta` task contracts, validate contracts, and write schema-v3
  knowledge bases (`scripts/build_merged_landscape.py`).
- Extended the real-data registry with analysis_task / ground_truth_kind / study
  fields, multi-platform filters, Slide-seqV2 entry, and `registry_summary()`.
- Default HTML reports now emit multi-method **boundary uncertainty** maps when
  ≥2 partitions are present (`uns['method_predictions']` or multiple `domain*` columns).
- Leaderboard feed v2: task-contract fields, SOTA family, config keys as methods,
  external submission protocol (`leaderboard/SUBMISSION.md`).
- Method guide pages under `docs/methods/` (when to use / when not / failure modes).

### P0 platform hardening (method selection under uncertainty)

- Repositioned the product claim from “universal best-method recommender” to
  **method × spatial-context selection under uncertainty**, with global-best
  baselines and negative-result diagnostics.
- Registered first-class SOTA plugins: `spagcn`, `graphst`, `stagate`,
  `bayesspace`, `rctd` (fail closed when optional backends are missing).
- Added task contracts that reject Leiden / self-supervised labels as
  spatial-domain ground truth (`AnalysisTask` / `GroundTruthKind`).
- Recommendation engine v2: task + platform priors, `method@policy` ranking,
  regret vs global-best, applicability warnings, knowledge-base schema v3.
- Maturity de-inflation with three **validated** methods
  (`banksy_py`, `spectral`, `gaussian_mixture`) and experimental demotion of
  teaching baselines / in-house autoencoders.
- Replaced brute-force spatial kNN with `scipy.spatial.cKDTree` (`O(n log n)`)
  and made landscape feature extraction sparse-safe.
- Default synthetic benchmarks skip SOTA/research tracks (opt-in via `methods=`).

### Added - cross-tissue real-data validation

- Added a 7x19 validation protocol spanning five DLPFC slices, pathology-labelled
  Xenium Prime Human Lymph Node, and Allen Brain Atlas MERFISH mouse brain.
- Added reproducible bundle preparation, truth-source manifests, deterministic
  stratified subsampling, and explicit unsupported status for BayesSpace on
  non-Visium coordinate systems.

### Added - SOTA spatial-domain benchmark

- Extended the DLPFC landscape from 15 to 19 methods with official SpaGCN,
  GraphST, BayesSpace, and STAGATE backends.
- Added method-specific isolated interpreter support, a native BayesSpace R
  bridge, explicit per-cell failure fields, dynamic 5x19 reports/figures, and
  adapter contract tests.

### Added - production compiler v1

- Added content-addressed `hwc1_...` plan identities and SHA-256 catalog digests so an
  executable plan is tied deterministically to its question, registry-backed steps,
  capability gaps, execution contract, model, and catalog snapshot.
- Added a bounded strict-JSON schema for plans and parameters, including finite-number,
  nesting, size, step-count, and unknown-field validation before registry validation.
- Added atomic `save_plan`/`load_plan` persistence with schema and integrity verification,
  live-registry revalidation, and optional strict catalog-drift detection.
- Pinned every compiled step to its exact `method_version`, materialized registry defaults,
  and persisted the real `catalog_assay` filter separately from model assay assumptions.
- Required a sealed, untampered plan plus explicit non-persistent execution confirmation;
  CLI `--yes` and SDK `confirmed=True` no longer mutate plan identity.
- Added provider timeouts, configurable repair-attempt limits, and CLI plan-artifact output
  for auditable, reproducible natural-language compilation.

### Added - platform topography and boundary uncertainty

- Added a target-free cross-platform terrain experiment over the archived 8-dataset x
  15-configuration landscape. The deterministic z-score PCA, winner/runner-up margins,
  spatial-weight response panel, machine-readable CSV/JSON, and SHA-256 manifest can be
  rebuilt without raw assay files.
- Added a label-permutation-invariant boundary-uncertainty map that combines method-specific
  neighbourhood boundary votes without using ground truth during discovery.
- Added a DLPFC case study showing that uncertainty >= P80 enriches boundaries uniquely
  missed by one method 1.83x and recovers 82/129 (63.6%) of those blind spots, with
  per-spot outputs, validation metrics, publication figures, and a checksum manifest.

### Added - real method adapters and method lifecycle

- Registered strict adapters for cell2location, Bioconductor BANKSY, SpatialDE,
  Cellpose 2, and scANVI. Missing optional backends now fail with installation
  diagnostics; none of these adapters falls back to a heuristic substitute.
- Added structured external-backend requirements and release gates covering the
  critical scientific methods and the broader external-wrapper catalogue.
- Versioned registry identities as `category:name@version`, with latest-active
  resolution, exact version pinning in the SDK, pipeline, CLI, manifests, and
  provenance, plus all-version discovery.
- Added visible deprecation warnings, exact replacement targets, scheduled removal
  metadata, declarative parameter renames/removals, cycle-safe chained migration,
  and final-schema validation through `migrate_method_params`.

### Added - method coverage and multimodal maturity

- Expanded the built-in registry from 33 to 51 runnable method contracts.
- Added 10 dependency-light production methods for QC, normalization, and SVG analysis.
- Added eight bounded PyTorch representation models, including four registered-image
  plus expression fusion models, and a dedicated `deep-learning` install extra.
- Added machine-readable modality/model-family metadata and release-gate reporting.
- Reached 100% BETA+, 80.39% PRODUCTION+, 0% experimental, and 11 deep models.

### Added - evidence-based method recommendation

- Added "histoweave recommend": bundle feature extraction, reference-fit scaling,
  k-nearest dataset retrieval, similarity-weighted method ranking, and strict
  JSON/file output.
- Added a 16-feature target-free recommendation schema. Domain truth, predicted
  domains, and cell-type labels are excluded to prevent target leakage.
- Recommendation evidence now reports per-method uncertainty, support, coverage,
  neighbour scores, and a clearly labelled, non-executed ensemble suggestion.
- Added versioned, atomic knowledge-base persistence with finite JSON encoding and
  task/metric/feature-order metadata.
- Corrected expression entropy and Hopkins self-neighbour calculations.
- Added scientific-property, SDK round-trip, missing-score, and CLI integration tests.
- Added "histoweave benchmark --suite figure3": a fixed 10-method x 3-dataset
  performance landscape, dataset-level leave-one-out recommendation validation,
  plotting-ready CSV/JSON outputs, SHA-256 manifest, and explicit caveat report.
- Added BIRCH, MiniBatchKMeans, and BisectingKMeans domain methods to reach ten
  locally runnable Python methods in the Figure 3 protocol.
- Replaced process-randomized synthetic suite seeds with stable CRC32 seeds.
- Forwarded random_state to GaussianMixture and fixed n_init, after independent
  process reruns exposed nondeterministic Figure 3 results.

### Added — Phase-1 ingestion & pipeline execution
- **Native Visium & Xenium readers** (`histoweave.io`): parse the real Space Ranger /
  Xenium on-disk layout (10x feature-barcode `.h5` matrix + `spatial/`·`cells.parquet`
  tables) directly via `h5py`/`pyarrow`, with no dependency on the heavy `spatialdata`
  stack. `spatialdata-io` remains available as `engine="spatialdata"`. Xenium ingestion
  filters negative-control probes by default; legacy/modern Visium position files are
  both handled.
- **Format-faithful vendor fixtures** (`datasets.write_visium_fixture` /
  `write_xenium_fixture`): tiny datasets written in the exact vendor layout so the
  readers are tested end-to-end in CI without a multi-GB download.
- **Portable persistence bundle** (`io.write_bundle` / `read_bundle`): a dependency-light
  `*.ttab` directory (`.npy` + Parquet + JSON) that round-trips the full `SpatialTable`
  (X, obs, var, obsm, layers, images, shapes, uns) between pipeline stages.
- **CLI stage commands** `histoweave ingest` / `step` / `report`, threading bundles between
  isolated steps — the interface the Nextflow processes call.
- New optional extra `io = [h5py, pyarrow]` for the native ingestion path.

### Changed — Nextflow pipeline is now runnable
- `workflows/nextflow/main.nf` shells out to the real `histoweave ingest/step/report`
  commands over `*.ttab` bundles (was a placeholder skeleton). Added a container-free
  `local` profile so it runs on a laptop/CI with the `histoweave` on PATH, an `annotate`
  toggle, and `--param` passthrough for per-step method parameters.
- Updated `main.nf` to the strict Nextflow grammar (>=25): parameters live in
  `nextflow.config`'s `params {}` block and the run banner moved inside `workflow {}`
  (no top-level statements). Fixed the `annotate` gate to coerce the CLI value (a bare
  `--annotate false` arrives as the truthy string `"false"`), and set `overwrite = true`
  on the report/timeline/dag/trace so `-resume` re-runs refresh them. Verified end-to-end
  on Nextflow 26.04.6: the demo DAG (`-profile local,test`) runs all six processes to
  completion, and the **real 10x Visium mouse-brain sample** runs the full
  ingest→QC→normalize→domains→report path (annotation skipped) — every process exit 0.
- `basic_qc` mitochondrial detection now matches the prefix against a gene-symbol column
  (`feature_name`/`gene_name`/…) when present, so it works on reader output where `var`
  is indexed by a stable feature id rather than by symbol.

### Added — Phase-0 scaffold
- Six-layer package skeleton (`ingestion → data → workflow → plugins → benchmark → report`).
- Canonical `SpatialTable` data model with structured `Provenance` and an AnnData bridge.
- Plugin layer: typed `Method` interface, `MethodSpec` metadata, and a queryable registry
  with `histoweave.plugins` entry-point discovery.
- Built-in reference methods: `basic_qc`, `log1p_cp10k`, `kmeans` (spatial domains),
  `marker_score` (annotation).
- In-process workflow runner with a captured `RunManifest`; default Phase-1 pipeline.
- Benchmarking harness with a working domain-detection task scored by Adjusted Rand Index.
- Self-contained HTML reporting with inline-SVG spatial maps (no plotting dependency).
- CLI: `version`, `list-methods`, `run`, `benchmark`.
- Deterministic synthetic dataset generator for tests, tutorials, and benchmarking.
- Test suite (67 tests) on tiny canonical datasets; GitHub Actions CI.
- Nextflow DSL2 pipeline stub mirroring the in-process runner.
- Docs (MkDocs), plugin template, and project governance files.

### Changed
- `SpatialTable` gains a `layers` mapping (AnnData-style, shape-aligned to `X`) as the
  proper home for alternative matrices; it is validated on construction and carried
  through `copy`, `subset_obs`, and the AnnData bridge.
- `SpatialTable` gains `images` and `shapes` spatial-layer slots (SpatialData-style,
  coordinate-system aligned rather than obs-aligned) — the prerequisite for a lossless
  SpatialData bridge. They are deep-copied and, being non-obs-aligned, carried through
  `subset_obs` unchanged; the AnnData bridge documents that it drops them.
- Split the domain-detection tests by responsibility: the method test now asserts the
  run/output contract (labels, categorical dtype, `X_pca` shape, determinism) while the
  benchmark test is the sole owner of recovery-accuracy (ARI) regression protection.
- `log1p_cp10k` now stashes the pre-normalization counts in `layers['counts']` instead of
  only setting an unbacked `uns['counts_preserved']` flag, so count-based methods can
  actually recover the raw counts — and, being shape-aligned, they stay aligned through
  later subsetting.
- Raised the domain-detection benchmark regression floor from ARI `> 0.4` to `> 0.90`
  (kmeans lands at ~0.99 on the reference sample), so a broken normalizer/PCA/clustering
  is caught instead of sliding under a loose bar.
- Added a dedicated `tests/test_math.py` covering `_math` invariants directly (ARI edge
  cases, PCA energy/distance reconstruction, k-means recovery, kNN self-neighbour).

### Fixed
- `spatial_scatter_svg` validates that `coords` and `labels` are the same length and zips
  them strictly, instead of silently truncating to the shorter of the two.
- `Task.prep` is typed `list[PipelineStep]` (was bare `list`).

[Unreleased]: https://github.com/histoweave-spatial/histoweave/commits/main

### Added - computational scalability proof

- Added sparse `make_scalable_synthetic`, isolated scaling harness, empirical complexity fits,
  and the `histoweave scale` CLI for reproducible pyramid scans with timeout/OOM ceilings.
- Added the 16-vCPU/64-GB 1,000,000-cell x 30-method proof artifacts: 150 measured cells,
  source CSV/JSON, report, and five editable SVG/PNG figures under `scalability_proof/`.
