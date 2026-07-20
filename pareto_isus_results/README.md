# Pareto and ISUS reference results

**Role:** frozen **reference artefact** for multi-objective Pareto reports and
post-hoc ISUS calibration (not a pre-execution predictor).

**Registered in:** [`docs/methods/validation/index.md`](../docs/methods/validation/index.md)
under *Reference artefacts*.

This directory archives the reference outputs supplied with the Pareto/ISUS
change set and merged into HistoWeave on 2026-07-18.

## Contents

- `pareto_report.json`: four-objective report for five DLPFC slices.
- `pareto_frontier_151669.*` and `pareto_frontier_151673.*`: representative
  Pareto-frontier plots in SVG and PNG formats.
- `isus_calibration.json` and `isus_calibration_table.csv`: five-slice ISUS
  calibration against observed spatial ARI gain.
- `isus_calibration.*`: calibration figure in SVG and PNG formats.

## Reproduce

From the repository root:

```bash
histoweave pareto \
  --benchmark-long 5x15_spatial_aware/benchmark_long.csv \
  --scaling-dir scalability_proof \
  --out pareto_report.json
```

The archived Pareto frontiers and knee configurations exactly match a fresh
run against the bundled benchmark and scaling inputs. Recomputing the ISUS
calibration additionally requires the five source H5AD slices under the
benchmark directory's `data/` folder; those large input files are not bundled.

The legacy absolute thresholds 0.1/0.3 are heuristic only. Prefer
`--n-null` for permutation p-values/Z-scores and null-derived bands. In this
five-slice calibration the Spearman correlation is -0.30 and is not significant
(`predictor_status=underpowered`; gain-map reliability is low because n=5 and
the slope is essentially flat / non-positive). Re-run:

```bash
histoweave isus --calibrate 5x15_spatial_aware --n-null 99 --out isus_calibration.json
```

to attach coordinate-shuffle type-I controls and an explicit
`gain_calibration` block binding ISUS to `benchmark_long.csv` spatial ARI gain
(best `sw>0` − `sw0.0`). That map is exploratory when reliability is low.