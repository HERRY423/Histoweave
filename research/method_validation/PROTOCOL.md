# Multi-dataset method validation protocol

**Protocol ID:** `histoweave.method_validation.multidataset.v1`
**Maturity target:** `MethodMaturity.VALIDATED`

## Purpose

Promote a registered method from production/beta to **validated** only when
multi-dataset evidence is written, reproducible, and linked from
`VALIDATION_EVIDENCE` in `src/histoweave/plugins/builtin/release_manifest.py`.

## Dataset tiers

| Tier | Datasets | Metric |
|------|----------|--------|
| **A — synthetic domain** | Figure 3 suite: `clean_easy`, `noisy_hard`, `sparse_scattered` | ARI vs planted domains |
| **B — real domain (DLPFC)** | Maynard 2021 Visium: ≥3 slices among 151507/669/670/673/674 | ARI vs manual cortical layers |
| **C — spatial-aware domain** | Same DLPFC slices × spatial_weight ∈ {0.0, 0.3, 0.8} | ARI (best-sw and mean) |
| **D — deconvolution structural** | ≥3 SpatialTables (real or synthetic) with reference signatures | shared-gene coverage, simplex proportions, contract success |

A domain method needs **A + (B or C)**.
A deconvolution method needs **D** on ≥3 datasets; full scRNA proportion
concordance is optional and marked separately when unavailable.

### SOTA batch (protocol `histoweave.method_validation.sota_batch.v1`)

| Method class | Gate |
|--------------|------|
| Official multi-slice ARI available (e.g. SpaGCN) | mean ARI ≥ 0.12 on ≥3 DLPFC slices, ≥9 runs |
| Isolated DL backends (GraphST / STAGATE) | structural multi-dataset 100% + mock disclosed + no silent fallback |
| R deconvolution (RCTD) | fail-closed without driver on ≥3 datasets; no marker fallback |
| SVG (SpatialDE) | multi-dataset ranking export on ≥3 datasets |

Paper-grade ARI for GraphST/STAGATE/BayesSpace requires isolated envs
(`HISTOWEAVE_*_PYTHON`) via `histoweave.benchmark.sota_pipeline`.

## Thresholds (honest, not SOTA-chasing)

| Task | Gate | Notes |
|------|------|-------|
| Domain ARI (synthetic mean) | ≥ 0.40 | Algorithms that collapse (DBSCAN noise-only) fail |
| Domain ARI (DLPFC mean, oracle *k*) | ≥ 0.12 | Layer recovery is hard; baselines often 0.15–0.30 |
| Domain multi-slice coverage | ≥ 3 slices / synthetic sets | Single-slice case studies do not validate |
| Deconvolution contract | 100% success on declared suite | No silent marker-score fallback |
| Independent review | Written limitations section | Required in each report |

## Report package (per method)

`docs/methods/validation/<method>.md` must contain:

1. Identity (category, wrap, maturity before/after)
2. Protocol + dataset table
3. Primary metrics (table)
4. Failure modes / limitations
5. Decision (validated / hold)
6. Artifact paths (CSV/JSON)

## Reproducibility

```bash
python research/method_validation/compile_validation_evidence.py
python research/method_validation/run_cell2location_multidataset.py  # optional extra
```

Artifacts land in `research/method_validation/results/`.
