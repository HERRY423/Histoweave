# External method submission protocol

HistoWeave accepts offline benchmark submissions that can be merged into the
public leaderboard and into a recommender knowledge base.  Submissions are
**append-only evidence**, not a claim that the method is production-ready.

> **Two ways to contribute evidence.** This page covers the **one-shot central
> submission** (a maintainer merges your CSV once). For a **living, continuously
> growing evidence landscape** where multiple labs contribute *signed* results
> that gain trust as other labs independently reproduce them — without sharing
> raw data — see the **[Federated Evidence Network](../federation/PROTOCOL.md)**
> (`histoweave fed`) and its [contributor guide](../federation/CONTRIBUTING_EVIDENCE.md).
> The federation reuses the same task contracts and metrics as this protocol.

## Schema: `histoweave.external_submission.v1`

Provide a CSV (or JSON list) with one row per dataset × method × seed:

| column | required | description |
|--------|:--------:|-------------|
| `dataset` | yes | Registry name or bare DLPFC slice id (`151673`) |
| `method` | yes | Method slug (`spagcn`, `my_lab_model`) |
| `seed` | yes | Integer random seed |
| `ari` | yes* | Primary metric (ARI for spatial-domain recovery) |
| `seconds` | no | Wall time |
| `status` | no | `success` / `failed` / `timeout` |
| `error` | no | Short error string when failed |
| `n_domains` | no | Domain count supplied to the method |
| `config` | no | Optional `method@policy` key (preferred for sweeps) |

\* For non-ARI tasks use `score` + declare `metric` in the sidecar JSON.

### Sidecar JSON (optional but recommended)

```json
{
  "schema_version": "histoweave.external_submission.v1",
  "task": "spatial_domain",
  "metric": "ARI",
  "higher_is_better": true,
  "method": {
    "name": "spagcn",
    "version": "1.2.7",
    "wraps": "SpaGCN",
    "language": "python",
    "container_digest": null,
    "url": "https://github.com/jianhuupenn/SpaGCN"
  },
  "datasets": [
    {
      "id": "151673",
      "registry_name": "dlpfc_151673",
      "task": "spatial_domain",
      "ground_truth_kind": "spatial_domain",
      "label_key": "domain_truth",
      "platform": "visium"
    }
  ],
  "notes": "Official SpaGCN defaults; CPU; fixed seed."
}
```

## Hard rules

1. **Task contracts.** `spatial_domain` submissions may only score against expert
   spatial partitions.  Leiden / Louvain / self-clustering labels are rejected.
2. **No silent fallbacks.** Failed backends must set `status=failed` and leave
   `ari` empty — never substitute a toy method under the same name.
3. **Seeds.** Report at least 3 seeds when the method is stochastic.
4. **Oracle k.** Document whether `n_domains` was the true domain count.
5. **License.** Upstream method license must allow redistribution of scores
   (scores only; not the trained weights unless permitted).

## How to merge

```bash
# 1. Drop CSV next to the other artefacts
cp my_sota_results.csv 5x15_spatial_aware/sota_benchmark_long.csv

# 2. Rebuild the public leaderboard feed
python leaderboard/generate.py

# 3. Build a recommender knowledge base with dataset_meta
python scripts/build_merged_landscape.py --out figure3_results/landscape_sota.json
```

Or from Python:

```python
from histoweave.benchmark import (
    landscape_from_long_csv,
    merge_landscapes,
    write_landscape_json,
    validate_landscape_contracts,
)

sota = landscape_from_long_csv("my_sota_results.csv", prefer_config_as_method=False)
problems = validate_landscape_contracts(sota)
assert not problems, problems
write_landscape_json(sota, "my_landscape.json")
```

## Review checklist (maintainers)

- [ ] Task / ground_truth_kind consistent with registry
- [ ] Method name does not collide with a different algorithm
- [ ] Seeds and runtime present
- [ ] Failures explicit
- [ ] PR links to upstream commit / container digest
