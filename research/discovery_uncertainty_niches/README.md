# Discovery track: multi-method uncertainty niches (DLPFC)

Use HistoWeave as a **discovery instrument**, not only a benchmark harness.

## Scientific goal

Identify candidate **tissue niches** (contiguous spatial regions) and **cell-state
programs** (SVG gene sets) that:

1. sit where multiple domain methods disagree (high boundary uncertainty);
2. are **not** explained solely by known cortical layer boundaries;
3. replicate across independent donor slices under pre-registered gates.

## Run

```bash
# from repo root
python research/discovery_uncertainty_niches/run_discovery.py
```

Requires local DLPFC bundles under `datasets_cache/dlpfc/` (already present in this
checkout).

## Pipeline (HistoWeave tools)

1. Load 3 donor slices (`151508`, `151669`, `151673`) via `get_dataset`.
2. HVG selection after log1p library-size norm (exclude MT/Ribo/IG/HB prefixes).
3. `estimate_n_domains` (silhouette) → non-oracle *K*.
4. Domain ensemble at coarse *K* and fine *K* (`kmeans`, `spectral`, `banksy_py`,
   `gaussian_mixture`).
5. `boundary_uncertainty` → target-free uncertainty map.
6. Cryptic mask = high-U ∧ ¬ known layer boundary.
7. Contiguous cryptic components (kNN graph BFS).
8. Moran's I SVG + BH-FDR; spatial-shift null + BH for cryptic enrichment.
9. Cross-slice gene replication gates → GO / NO-GO.

## Interpreting status labels

| Status | Meaning |
|--------|---------|
| `CANDIDATE` | Cryptic gene programs pass FDR + geometry gates on this slice |
| `GEOMETRIC_CANDIDATE` | Contiguous cryptic niches exist; gene program gate fails |
| `WEAK_OR_NOGO` | Fails geometry and/or gene gates |
| `FAILED` | Runtime error |

## Upgrade path to a real discovery claim

### Done in-repo (computational)

```bash
# Preferred: package CLI (from repo root / editable install)
histoweave discovery cohort
histoweave discovery bootstrap-ci
histoweave discovery panel
histoweave discovery run
histoweave stats-review --landscape figure3_results/landscape.json --out stats.json
histoweave sota --dry-run

# Equivalent research scripts
python research/discovery_uncertainty_niches/run_cohort_panel.py
python research/discovery_uncertainty_niches/run_donor_bootstrap.py
python research/discovery_uncertainty_niches/run_discovery.py
python research/discovery_uncertainty_niches/validate_panel_and_rois.py
```

| Doc | Content |
|-----|---------|
| `COHORT_META_REPORT.md` | 12-slice meta-analysis (L3 **14/15** direction) |
| `DONOR_BOOTSTRAP_L3.md` | Donor-stratified 95% CIs (L3/myelin exclude 0) |
| `PROJECT_STATUS.md` | Frozen claim + next steps |
| `PANEL_VALIDATION_REPORT.md` | 3-donor pilot panel gates |
| `IF_PROTOCOL.md` | Protein validation hand-off |
| `results/cohort/IF_priority_L3_ROIs.csv` | IF-first L3 ROIs (direction + shift p≤0.05) |

### IF path to validated biology (151508 L3 + L6)

```bash
# 1) Build lab package (ROIs, GeoJSON, briefing figures, claim ladder)
histoweave discovery if-package
# or: python research/discovery_uncertainty_niches/prepare_if_lab_package.py

# 2) Wet lab: ENC1 + HOPX + MBP on section 151508 (see IF_LAB_BRIEF.md)

# 3) Drop intensity CSV into results/if_return/ then:
histoweave discovery if-analyze
# Dry-run analyzer only (NOT protein):  ... if-analyze --simulate-from-rna
```

| File | Role |
|------|------|
| `IF_LAB_BRIEF.md` | Core-facility hand-off (EN+ZH) |
| `CLAIM_LADDER.md` | Allowed language at each evidence level |
| `results/if_lab_package/` | ROIs, backgrounds, briefing maps |
| `results/if_return/VALIDATED_BIOLOGY_REPORT.md` | Auto-upgraded narrative after IF |

### Still required for a named biological claim

- **Real protein IF** (wet lab) — computational package is ready; Level 3 is PENDING_WET_LAB.
- Same-layer protein statistics (RNA same-layer hard still fails for L3).
- Do **not** rename L3+L6 as one cell state.

## Related prior work in this repo

- `research/cross_tissue_niches/` — cross-tissue GEI / scale hierarchy (prior NO-GO).
- `case_study_dlpfc_consistency/` — boundary-miss case study on 151673.
