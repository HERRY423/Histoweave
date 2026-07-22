# HistoWeave submission freeze v1

This directory is the submission-facing evidence lock for the Bioinformatics
paper track. It fixes the five main figures, one supplement benchmark summary
table, one reproduction script, and one data/code availability checklist.

## Frozen manuscript spine

1. Figure 1: external spatial-domain performance heatmap.
2. Figure 2: method ARI distribution across external datasets and seeds.
3. Figure 3: dataset-feature landscape embedding.
4. Figure 4: recommender regret versus global and random baselines.
5. Figure 5: selective regret-coverage, showing that abstention preserves the
   global default when personalisation is higher regret.

The figure lock is written to `main_figures.lock.json`. It records the PNG/SVG
paths, source data, generator, short caption, and SHA-256 hashes for every
frozen figure file.

## Frozen supplement table

`supplement_benchmark_table.csv` is a compact, manuscript-ready benchmark table
covering:

- n=5 external LOOCV;
- strict task-stratified panel v2: 10-unit registry, n=9 domain LOOCV,
  two-dataset TLS evidence, and aligned SOTA coverage;
- a one-shot, preregistered test on six previously unseen Wu 2021 breast-cancer
  patients/sections (retained as a negative external result);
- n=20 selective regret-coverage;
- DLPFC SOTA real-data benchmark;
- federated evidence-network reference implementation.

## Reproduction

From the repository root:

```bash
python submission_freeze_v1/reproduce_submission_freeze.py
```

The script rebuilds the supplement benchmark table, recomputes figure hashes
from the frozen figure files, and rewrites `submission_freeze_manifest.json`.
If the official Xenium lymph-node bundle is available, add
`--regenerate-strict-panel` to rerun the second TLS dataset, aligned BANKSY
cell, and panel v2. Use `--regenerate-independent-test` to rerun the frozen
Wu 2021 test when its official Zenodo raw bundle is available. If prepared
dataset caches are available, add `--regenerate-figures` to redraw
the external-validation figures before hashing. It does not rerun
STAGATE/GraphST/BayesSpace because those require external method-specific
environments; their locked outputs and environment notes are included in the
manifest and availability checklist.

## Submission boundary

This freeze supports the paper claim that HistoWeave is an evidence-governed
decision protocol for spatial transcriptomics method choice. The preregistered independent study test failed its 0.02-ARI regret margin
(observed 0.1313; 95% bootstrap CI 0.0340-0.2363). Accordingly, the
submission may claim a reproducible evidence-governance and abstention
framework, but must not claim that the frozen spectral policy transports or
that personalised recommendation is superior. The test cohort remains sealed
from training and threshold selection.
