# Changelog

All notable changes to this project will be documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
