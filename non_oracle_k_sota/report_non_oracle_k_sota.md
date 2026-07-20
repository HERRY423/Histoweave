# Non-oracle K SOTA benchmark (DLPFC × SpaGCN / STAGATE)

**Protocol:** `histoweave.non_oracle_k_sota.v1`  
**Slices:** 151673, 151674, 151507, 151669, 151670 (seed 42)  
**Methods:** SpaGCN (official 1.2.7), STAGATE_pyG  
**K policies:** `oracle` (ablation) vs `estimate` with `silhouette` / `spatial_silhouette` / `ensemble`  
**Dual-track:** every slice records oracle *K* vs estimated *K* (`DualTrackKReport`)

Figure: [`figures/non_oracle_k_ari_recovery.svg`](figures/non_oracle_k_ari_recovery.svg)

---

## Headline result (critical, not cosmetic)

| Method | Oracle-K mean ARI | Silhouette-estimate mean ARI | Δ (drop) |
|--------|------------------:|-----------------------------:|---------:|
| **SpaGCN** | **0.299** | **0.237** | **−0.062** |
| **STAGATE** | **0.232** | **0.219** | **−0.013** |

On the hardest drop (SpaGCN · 151673):

| K policy | K used | ARI |
|----------|-------:|----:|
| Oracle-K | 7 | **0.418** |
| estimate · silhouette | 2 | **0.186** |
| estimate · spatial_silhouette | 2 | 0.186 |
| estimate · ensemble | 2 | 0.186 |

**Removing Oracle-K protection costs SpaGCN 0.232 ARI on 151673** (≈55% relative loss). That is the primary engineering/scientific claim: published SOTA numbers that inject expert layer counts are not transferable to blind analyses.

---

## Dual-track K match rates

| Estimator | Exact match vs true *K* | Mean estimated *K* (range of truth ≈5–8) |
|-----------|------------------------:|------------------------------------------|
| silhouette | 0/5 | ~2.2 |
| spatial_silhouette | 0/5 | 2.0 |
| ensemble | 0/5 | 2.0 |

All blind estimators **collapse toward *K*=2** on these layered Visium sections. Chance-corrected spatial coherence and weighted ensemble reduce that bias on synthetic data (unit tests), but on real DLPFC layers they do **not** recover the true domain count under the current score geometry.

---

## Mean ARI by method × K mode

| Method | Mode | Mean ARI | Std | Mean *K* used |
|--------|------|---------:|----:|--------------:|
| spagcn | oracle | 0.299 | 0.098 | 6.8 |
| spagcn | estimate:silhouette | 0.237 | 0.107 | 2.2 |
| spagcn | estimate:spatial_silhouette | 0.231 | 0.107 | 2.0 |
| spagcn | estimate:ensemble | 0.231 | 0.107 | 2.0 |
| stagate | oracle | 0.232 | 0.089 | 6.8 |
| stagate | estimate:silhouette | 0.219 | 0.109 | 2.2 |
| stagate | estimate:spatial_silhouette | 0.214 | 0.108 | 2.0 |
| stagate | estimate:ensemble | 0.214 | 0.108 | 2.0 |

### Recovery vs silhouette baseline

| Method | Recovered by spatial_silhouette | Recovered by ensemble | Fraction of drop recovered |
|--------|--------------------------------:|----------------------:|---------------------------:|
| SpaGCN | −0.006 | −0.006 | **0%** |
| STAGATE | −0.005 | −0.005 | **0%** |

**Honest interpretation:** on this five-slice DLPFC panel, spatial_silhouette / ensemble **do not yet reclaim** the ARI lost when silhouette underestimates *K*. They track the same low-*K* mode. The contribution of this experiment is therefore:

1. **Quantify Oracle-K inflation** for two official SOTA backends under a fixed protocol.
2. **Show dual-track failure** of classical and spatial-aware *K* estimators on layered Visium.
3. **Motivate** further non-oracle *K* work (spatial BIC, resolution search without labels, multi-scale criteria) — not claim a solved recovery.

---

## Per-slice SpaGCN (oracle vs estimate)

| Slice | Oracle *K* | Est. silhouette *K* | Oracle ARI | Silhouette ARI | Δ |
|-------|-----------:|--------------------:|-----------:|---------------:|--:|
| 151673 | 7 | 2 | 0.418 | 0.186 | −0.232 |
| 151674 | 7 | 2 | 0.273 | 0.199 | −0.074 |
| 151507 | 7 | 3 | 0.385 | 0.248 | −0.137 |
| 151669 | 8 | 2 | 0.207 | 0.137 | −0.070 |
| 151670 | 5 | 2 | 0.213 | 0.415* | +0.202* |

\*On 151670, SpaGCN with *K*=2 **exceeds** oracle *K*=5 ARI under this seed — a reminder that ARI is not monotone in *K*, and that dual-track reporting must separate “match truth” from “maximise ARI”.

---

## Reproducibility

```bash
# Full dual-track + SpaGCN/STAGATE (seed 42)
set KMP_DUPLICATE_LIB_OK=TRUE
set HISTOWEAVE_STAGATE_EPOCHS=150
set HISTOWEAVE_SPAGCN_MAX_EPOCHS=120
set HISTOWEAVE_NON_ORACLE_SEEDS=42
set HISTOWEAVE_K_MAX_OBS=2500
python non_oracle_k_sota/run_non_oracle_k_sota.py
```

Checkpoints are written under `checkpoints/` and reused unless `HISTOWEAVE_NON_ORACLE_FORCE=1`.

Artefacts:

- `benchmark_long.csv` / `.json` — 40 cells (5×2×4)
- `dual_track_k.csv` / `.json` — DualTrackKReport per slice × estimator
- `summary.json` — mean ARI, drops, recovery fractions
- `figures/non_oracle_k_ari_recovery.{svg,png}`

---

## Take-home for papers / reviews

1. **Default scientific path must be `k_policy=estimate`.** Oracle *K* is an opt-in ablation.
2. **SpaGCN (and to a lesser extent STAGATE) ARI is highly sensitive to *K* misspecification** on DLPFC.
3. **Expression-only silhouette is insufficient** as a blind *K* oracle substitute; spatial-aware estimators are necessary infrastructure, but **this panel shows they are not yet sufficient** for layered cortex.
4. **Dual-track reporting** (oracle vs estimated *K* + ARI under both) is the minimal audit trail for any SOTA claim that still reports an oracle-K number.
