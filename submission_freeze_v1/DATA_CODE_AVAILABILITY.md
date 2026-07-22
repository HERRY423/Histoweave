# Data and code availability checklist

## Code

- Repository: `https://github.com/HERRY423/Histoweave`
- License: BSD-3-Clause (`LICENSE`)
- Python package name: `histoweave-spatial`
- Submission freeze entry point:
  `submission_freeze_v1/reproduce_submission_freeze.py`
- Required pre-submission action: create a stable release tag and archive it in
  Zenodo, Figshare, Software Heritage, or an equivalent repository. Add the DOI
  here before journal submission.

## Reproducing the frozen figures and table

Run from the repository root:

```bash
python submission_freeze_v1/reproduce_submission_freeze.py
```

This rebuilds the freeze metadata and supplement table, and hashes these existing frozen figure files:

- `benchmark_external_validation/figures/fig1_performance_heatmap.{svg,png}`
- `benchmark_external_validation/figures/fig2_method_boxplot.{svg,png}`
- `benchmark_external_validation/figures/fig3_landscape_embedding.{svg,png}`
- `benchmark_external_validation/figures/fig4_recommender_regret.{svg,png}`
- `benchmark_external_validation/figures/selective_regret_coverage.{svg,png}`
- `submission_freeze_v1/main_figures.lock.json`
- `submission_freeze_v1/supplement_benchmark_table.csv`
- `submission_freeze_v1/submission_freeze_manifest.json`

To redraw the five figures before hashing, use:

```bash
python submission_freeze_v1/reproduce_submission_freeze.py --regenerate-figures
```

This requires the prepared dataset cache used by
`benchmark_external_validation/make_figures.py`.

To reproduce the second TLS dataset, the aligned lymph-node BANKSY cell, and
strict panel v2, use:

```bash
python submission_freeze_v1/reproduce_submission_freeze.py --regenerate-strict-panel
```

This requires the local official 10x Xenium Prime reactive lymph-node bundle.

To reproduce the frozen Wu 2021 independent test, use:

```bash
python submission_freeze_v1/reproduce_submission_freeze.py --regenerate-independent-test
```

This requires the official Wu et al. filtered matrices, spatial files, and
metadata from Zenodo DOI `10.5281/zenodo.4739739`. The six-patient test
cohort remains excluded from training and model selection.

## Derived benchmark data included in the repository

- External validation benchmark:
  `benchmark_external_validation/benchmark_long.csv`,
  `benchmark_external_validation/performance_matrix_mean.csv`,
  `benchmark_external_validation/recommendation_loocv.json`,
  `benchmark_external_validation/decision_validation.json`.
- Strict n=8 independent-unit validation:
  `benchmark_external_validation/n8_strict_region/loocv_summary.json`,
  `benchmark_external_validation/n8_strict_region/loocv_rows.csv`,
  `benchmark_external_validation/n8_strict_region/tissue_condition_flip.csv`.
- Strict task-stratified external panel v2:
  `benchmark_external_validation/strict_external_panel_v2/loocv_summary.json`,
  `benchmark_external_validation/strict_external_panel_v2/strict_external_units.csv`,
  `benchmark_external_validation/strict_external_panel_v2/sota_coverage.csv`,
  `benchmark_external_validation/strict_external_panel_v2/tls_two_dataset_summary.json`.
- TLS second-dataset negative transport result:
  `research/phaseB_tls_consensus/second_dataset_xenium_lymph/tls_second_dataset_summary.json`,
  `research/phaseB_tls_consensus/second_dataset_xenium_lymph/REPORT_tls_second_dataset.md`.
- Preregistered independent study test (negative result retained):
  `benchmark_external_validation/independent_test_wu2021/preregistered_protocol.json`,
  `benchmark_external_validation/independent_test_wu2021/independent_test_summary.json`,
  `benchmark_external_validation/independent_test_wu2021/sample_regret.csv`,
  `benchmark_external_validation/independent_test_wu2021/independence_audit.json`,
  `benchmark_external_validation/independent_test_wu2021/REPORT_independent_test_wu2021.md`.
- Selective regret endpoint:
  `protocol_endpoints_results/selective_regret_coverage.json`,
  `protocol_endpoints_results/study_grouped_20_recommendation.json`.
- SOTA real-data benchmark:
  `5x15_spatial_aware/sota_benchmark_long.csv`,
  `5x15_spatial_aware/sota_method_means.csv`,
  `5x15_spatial_aware/performance_matrix_mean_full.csv`,
  `5x15_spatial_aware/sota_merge_manifest.json`.
- Federated evidence-network reference implementation:
  `federation/PROTOCOL.md`, `federation/CONTRIBUTING_EVIDENCE.md`,
  `src/histoweave/federation/`, and `tests/test_federation_*.py`.

## Raw and third-party data

Raw spatial transcriptomics datasets are not redistributed in this repository.
The sealed independent test uses Wu et al. 2021 (`10.5281/zenodo.4739739`),
downloaded from the official Zenodo record; local archives are checksum-verified
and ignored by version control.
The preparation scripts under `benchmark_external_validation/` and
`5x15_spatial_aware/` document how each public dataset is transformed into the
task-checked benchmark tables. Before submission, add dataset repository
accessions or DOIs for every public source used in the manuscript reference
list and cite them as datasets.

## Method-specific environments

The submission reproduction script uses already committed SOTA outputs. Full
reruns of STAGATE, GraphST, and BayesSpace require their method-specific
environments. The locked environment notes are:

- `5x15_spatial_aware/env_locks/stagate_env.txt`
- `5x15_spatial_aware/env_locks/graphst_env.txt`

Full rerun entry points:

```bash
python 5x15_spatial_aware/run_one_method.py stagate checkpoints
python 5x15_spatial_aware/run_one_method.py graphst checkpoints
python 5x15_spatial_aware/build_sota_and_merge.py checkpoints
```

## Manuscript disclosure items

- Add a data availability statement describing the public raw datasets and the
  derived benchmark artifacts above.
- Add a software availability statement with the repository URL, release tag,
  package version, license, and archive DOI.
- Disclose AI assistance in the Methods or Acknowledgements if retained in the
  submitted work; authors remain responsible for all code, analyses, text, and
  figures.
