# Pareto and ISUS reference results

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

The ISUS thresholds are provisional descriptors, not validated predictors of a
specific method's ARI gain. In this five-slice calibration the Spearman
correlation is -0.30 and is not significant.