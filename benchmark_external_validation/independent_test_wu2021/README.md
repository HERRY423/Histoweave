# Frozen independent test - Wu et al. 2021 breast-cancer Visium cohort

This directory is a one-shot external test, not another development fold. The
protocol in `preregistered_protocol.json` was written before downloading or
inspecting the deposited outcomes.

Independence is defined at the study level: all six primary breast cancers from
Wu et al. are absent from the HistoWeave training landscapes, the n=9 strict
panel, threshold selection, and the TLS discovery sample. The frozen policy is
the existing global `spectral` default. The other six common-panel methods are
run only to calculate test regret; their test performance cannot change the
selected policy or the 0.02 ARI margin.

Data source: Zenodo DOI `10.5281/zenodo.4739739` (CC BY 4.0). Raw third-party
files remain under `datasets_cache/raw_sources/wu2021_breast/` and are not
intended for source-control inclusion.

The frozen result is negative: mean spectral-policy regret is 0.1313 ARI
(patient/section bootstrap 95% CI 0.0340-0.2363), versus the preregistered
0.02 margin. Spectral is top-ranked in 2/6 sections. This cohort remains
excluded from training and threshold selection.

Reproduce with `run_independent_test.py`; see
`REPORT_independent_test_wu2021.md` for the full result.
