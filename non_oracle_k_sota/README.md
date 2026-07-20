# Non-oracle K × SOTA (DLPFC) — **reference artefact**

**Role:** frozen dual-track SOTA archive for protocol endpoint
`histoweave.oracle_k_leakage.v1` (Oracle-K leakage / non-oracle K ARI drop).

**Registered in:** [`docs/methods/validation/index.md`](../docs/methods/validation/index.md)
under *Reference artefacts*.

Re-runs **SpaGCN** and **STAGATE** on five DLPFC slices under:

| Mode | Meaning |
|------|---------|
| `oracle` | `n_domains = domain_truth.nunique()` (opt-in ablation) |
| `estimate:silhouette` | expression-only silhouette |
| `estimate:spatial_silhouette` | neighbourhood-smoothed silhouette |
| `estimate:ensemble` | weighted spatial+expression ensemble |

Every slice also emits a **DualTrackKReport** (oracle *K* vs estimated *K*).

## Quick start

```bash
python non_oracle_k_sota/run_non_oracle_k_sota.py
# Consume as protocol endpoint 5:
python scripts/run_protocol_endpoints.py --skip-dlpfc-expand
# → protocol_endpoints_results/oracle_k_leakage.json
```

## Key figure

`figures/non_oracle_k_ari_recovery.svg` — mean ARI, per-slice SpaGCN curves, ΔARI vs oracle, dual-track *K*.

See `report_non_oracle_k_sota.md` for numbers and interpretation. Do **not**
quote oracle-track ARI as blind performance.
