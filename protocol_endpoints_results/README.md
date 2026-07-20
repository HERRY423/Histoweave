# Protocol endpoints 1–5 — reference artefacts

**Protocol family:** `histoweave.protocol_endpoints.*` (bundle summary  
`histoweave` endpoints in `protocol_endpoints_summary.json`)  
**Role:** frozen **in-repo** falsifiable evaluation bundle aligned with
[`docs/decision-protocol.md`](../docs/decision-protocol.md).

These files are **tracked in git** (small JSON/MD). Rebuild with:

```bash
python scripts/run_protocol_endpoints.py
```

## Tracked files

| File | Endpoint / role |
|------|-----------------|
| `protocol_endpoints_summary.json` | **Primary** machine-readable bundle |
| `protocol_endpoints_report.md` | Human-readable summary |
| `study_grouped_20_recommendation.json` | Study-grouped personalisation (n=20) |
| `study_grouped_20_report.md` | Narrative for study-grouped run |
| `selective_regret_coverage.json` | Selective regret–coverage policy |
| `pareto_stability.json` | Bootstrap frontier stability |
| `sota_unified_resource.json` | Resource-matched SOTA cells |
| `oracle_k_leakage.json` | Oracle vs estimate ARI drop |
| `multisource_landscape.json` | Multi-source landscape used by endpoints |

## Citation rule

1. Name the endpoint key inside `protocol_endpoints_summary.json`.
2. Do not mix oracle-K and estimate tracks when quoting SOTA ARI.
3. Negative personalisation results (`beats_global_best: false`) are first-class.

See [Reference artefacts](../docs/reference-artefacts.md).
