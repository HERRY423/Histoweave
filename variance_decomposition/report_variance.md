# Method-variance decomposition: what actually drives spatial-domain accuracy?

**Question:** when an analyst gets a different answer on the same tissue, how much of that
difference comes from the **method** they picked, versus the **preprocessing**, the
**hyper-parameters**, or just **sampling noise**?

**Design:** a fully-crossed factorial experiment on DLPFC slice 151673 (spatialLIBD manual
layers), 180 runs total:

```
4 preprocessing  ×  5 methods  ×  3 parameter settings  ×  3 subsamples  =  180 runs
```

- **preprocessing (4):** `log1p_cp10k`, `sqrt`, `scaled` (log1p→z-score), `arcsinh`
- **method (5):** `kmeans`, `gaussian_mixture`, `agglomerative`, `spectral`, `banksy_py`
- **param (3):** amount of **spatial context** each method uses — the single directly
  comparable knob across the family (`spatial_weight` 0.0/0.3/0.6 for the sklearn methods;
  BANKSY `lambda_param` 0.2/0.5/0.8). Levels labelled low / default / high.
- **subsample (3):** three random 80% spot subsamples (seeds 0/1/2)

**Metric:** Adjusted Rand Index (ARI) vs the manual layer annotation. All 180 runs
completed (0 failures).

## Variance decomposition (Type-II ANOVA on ARI)

| Factor | Sum of squares | **% of total variance** | F | p |
|---|---|---|---|---|
| **param (spatial context)** | 0.2577 | **41.3%** | 112.8 | 8.7e-32 |
| **method** | 0.1509 | **24.2%** | 33.0 | 2.6e-20 |
| preprocessing | 0.0231 | 3.7% | 6.7 | 2.6e-04 |
| subsample | 0.0011 | 0.2% | 0.5 | 0.63 (n.s.) |
| Residual (interactions + noise) | 0.1919 | 30.7% | — | — |

![variance decomposition](figures/fig_variance_decomposition.svg)

## What this means

1. **How much spatial context you use is the biggest lever (41%).** Averaged over
   everything else, ARI climbs from **0.166 (low)** → 0.220 (default) → **0.258 (high)**.
   Ignoring the neighbourhood term is the most common way to leave accuracy on the table.

2. **Method choice is the second-biggest lever (24%), and it is not independent of the
   parameter.** Mean ARI by method:

   | Method | mean ARI | std | Sensitivity to spatial-context param |
   |---|---|---|---|
   | **banksy_py** | **0.262** | 0.041 | **flat** (0.257→0.264) — spatial by construction |
   | kmeans | 0.218 | 0.047 | strong (0.170→0.259) |
   | spectral | 0.216 | 0.048 | strong (0.158→0.258) |
   | gaussian_mixture | 0.208 | 0.056 | strong (0.148→0.274) |
   | agglomerative | 0.171 | 0.065 | strong (0.098→0.237) |

   BANKSY is the most robust choice: highest mean and lowest variance because it always
   incorporates spatial context, so it is insensitive to the setting that makes or breaks
   the others. The generic clusterers can *match or beat* it — but only when tuned to high
   spatial weight. **This is exactly the "method × parameter" interaction that shows up in
   the 30.7% residual.**

3. **Preprocessing is nearly irrelevant here (3.7%).** For DLPFC domain detection on HVGs,
   swapping log1p / sqrt / scaled / arcsinh barely moves ARI — analyst effort is better
   spent on the method + spatial-context choice than on normalization.

4. **Results are stable to subsampling (0.2%, not significant).** Conclusions are not an
   artefact of which 80% of spots were used.

**Best single configuration:** `sqrt` + `spectral` + high spatial weight → ARI 0.312.

## Practical takeaways

- **Turn on spatial context.** The largest, cheapest accuracy gain is using a high
  neighbourhood weight — it dominates every other choice.
- **If you will not tune, pick a spatially-native method (BANKSY).** It removes the single
  most dangerous free parameter.
- **Do not over-invest in normalization** for this task; it explains <4% of the spread.
- **Report method + parameter together.** Because they interact, quoting a method's
  accuracy without its spatial-context setting is misleading.

## Limitations

- Single slice, one tissue (within-study). The *relative* factor ranking is the finding;
  absolute ARI values are slice-specific.
- The "param" factor is deliberately harmonised to spatial context so it is comparable
  across methods; a per-method multi-parameter grid would attribute more variance to
  method-specific tuning (and shrink the residual).
- `banksy_py` is HistoWeave's native BANKSY scaffold, not the Bioconductor implementation.

## Reproduce

```bash
python variance_decomposition/run_variance_experiment.py
# -> variance_runs.csv (180 rows), variance_components.csv, figures/
```
