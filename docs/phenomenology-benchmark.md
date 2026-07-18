# Phenomenology capability benchmark

HistoWeave evaluates methods against **spatial biological phenomena**, not against a single
platform leaderboard. The benchmark asks which phenomena a method can recover, how recovery
changes under observation noise, and whether the method ran within its declared resource budget.

## Factorial design

The frozen release benchmark contains:

- six phenomena: compartment, gradient, hotspot, boundary, mixture, and branching;
- five observation conditions: clean, low-depth/dropout, low-signal/noise, irregular sampling,
  and batch/platform confounding;
- 54 audited methods: every validated, production, or beta release plus the
  dependency-light `marker_deconv` baseline;
- five paired biological replicates with fixed seeds;
- locked defaults and, after calibration is frozen, a truth-blind tuned track.

This gives `6 × 5 × 54 = 1,620` method design units and `8,100` run attempts per track.
The primary key is
`phenomenon × condition × method × replicate × track`. Conditions share the biological
replicate; only the observation process changes.

## Scientific roles and applicability

A common plugin interface does not make every output comparable. Each method has a frozen
contract with one role:

- `direct_inference`: predicts domains, labels, mixtures, SVGs, graphs, communication, or
  segmentation directly;
- `preprocessing_preservation`: changes or filters data and is scored by downstream signal
  preservation;
- `representation_integration`: produces an integrated representation scored for biological
  conservation, batch mixing, recoverability, and oversmoothing;
- `ingestion_fidelity`: requires a vendor-specific round-trip fixture.

Deconvolution is applicable only to mixture scenarios. Cell-cell communication is applicable
to hotspot and mixture scenarios. An inapplicable pairing is recorded as `not_applicable`; it
is not treated as a failed scientific attempt or included in the score denominator.

Comparisons and summaries remain conditional on role and category. HistoWeave does **not**
produce a global rank across unlike tasks.

## Reproducible planning

Generate the full locked-track manifest without running methods:

```bash
histoweave benchmark --suite phenomenology --dry-run \
  --out-dir phenomenology_plan
```

The command writes:

- `experiment_manifest.json`: frozen scenarios, seeds, tracks, and run counts;
- `method_manifest.json`: exact method versions, defaults, and backend requirements;
- `capability_matrix.csv`: method × phenomenon applicability and metric contracts.

The method and scenario manifests are content-addressed with SHA-256. A changed method
version, parameter configuration, seed, or scenario creates a different run identifier.

## Running a subset

Start with a narrow locked-default subset:

```bash
histoweave benchmark --suite phenomenology \
  --phenomena compartment,gradient \
  --conditions clean,low_depth_dropout \
  --methods kmeans,banksy_py,morans_i \
  --track locked \
  --seeds 1729,2718 \
  --workers 2 \
  --out-dir phenomenology_subset
```

Use `--tiny` for a dependency-light smoke run. It uses 60 observations, 64 genes, one seed,
a 30-second ceiling, and a 4 GB memory ceiling. It is a software regression check, not
publishable evidence.

```bash
histoweave benchmark --suite phenomenology --tiny \
  --phenomena compartment --conditions clean \
  --methods basic_qc,log1p_cp10k,kmeans \
  --out-dir phenomenology_smoke
```

Runs execute in isolated child processes. Checkpoints are enabled by default; use
`--no-resume` to ignore them. Standard and heavy methods default to 600 and 1,800 seconds,
respectively, with a 16 GB memory limit.

## Tuned track boundary

`--track tuned` and `--track both` may be used for dry-run planning. Execution is intentionally
blocked until a separate calibration manifest freezes the truth-blind parameter choice. This
prevents evaluation truth from leaking into tuning. The preregistered search contains no more
than four candidates per tunable method.

## Status semantics

Every attempt has exactly one status:

| Status class | Values | Scientific denominator |
|---|---|---|
| Success | `ok` | included |
| Not applicable | `not_applicable`, `not_tunable` | excluded |
| Environment gap | `backend_unavailable`, `fixture_unavailable` | reported, not scored as zero |
| Scientific/execution failure | `invalid_input`, `method_error`, `timeout`, `oom`, `budget_exceeded` | scored as failure after applicability and backend availability are established |

Missing backends never trigger a silent substitute implementation.

## Output tables

An executed suite additionally writes:

- `runs.csv`: one status/resource row per attempt;
- `metrics.csv`: normalized and raw metrics in long form;
- `coverage_summary.csv`: applicability, success, environment-gap, and failure coverage;
- `capability_index.csv`: role/category-conditioned recovery, robustness, reliability, and
  efficiency summaries.

The capability index weights recovery 0.50, robustness 0.25, reliability 0.15, and efficiency
0.10. It is a compact summary, not permission to compare unrelated categories.

For confirmatory comparisons, use paired biological replicates, 95% hierarchical bootstrap
intervals, two-sided paired tests, and Benjamini-Hochberg control at 0.05 within
`role × track × primary metric` families. Large per-run artifacts should remain workflow
artifacts; only frozen manifests and compact summaries belong in version control.

## Interpretation limits

- A dry run validates design coverage, not method performance.
- A tiny run validates execution plumbing, not biological conclusions.
- Environment gaps quantify installation or fixture coverage and must not be conflated with
  method failure.
- The current tuned execution path remains blocked until calibration manifests are implemented.
- Empirical false-discovery control is evaluated at the truth-sized discovery cutoff; dedicated
  all-null simulations should be added before making strong calibration claims.
