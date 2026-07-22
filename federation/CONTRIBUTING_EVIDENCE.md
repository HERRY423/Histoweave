# Contributing evidence to the HistoWeave federation

This guide walks a lab through contributing benchmark results to the shared, living
evidence landscape **without sharing raw data**. You transmit only scalar scores plus
public dataset descriptors; a privacy gate rejects anything raw-data-shaped.

Read `PROTOCOL.md` first for the data model and trust rules. The whole flow uses the
`histoweave fed` CLI and needs no servers.

## Prerequisites

```bash
pip install "histoweave-spatial[federation]"   # adds the Ed25519 backend (cryptography)
histoweave fed --help
```

## Step 1 — Create your node identity

```bash
histoweave fed init-node \
  --node-id lab-cambridge \
  --display-name "Cambridge Spatial Lab" \
  --contact team@cam.example \
  --evidence-feed https://cam.example/histoweave/bundle.json \
  --out-dir federation
```

This writes two files:

- `federation/nodes/lab-cambridge.json` — your **public** registry entry (goes into the PR).
- `federation/lab-cambridge.key.json` — your **PRIVATE** Ed25519 key. **Never commit it.**
  It is already covered by `.gitignore` (`federation/*.key.json`), but treat it like any
  other secret: back it up securely and keep it out of shared drives.

> Losing the private key just means you rotate to a new key (add a new `active` public key,
> mark the old one `revoked` in your registry entry). Leaking it means anyone can sign as
> your node — rotate immediately.

## Step 2 — Produce benchmark results locally

Run your method through the normal HistoWeave benchmark so you get a
`benchmark_long.csv` with one row per `dataset × method × seed`:

| column | required | notes |
|--------|:--------:|-------|
| `dataset` | yes | registry name or bare DLPFC slice id (`151673`) |
| `method` | yes | method slug |
| `seed` | yes | integer; report ≥3 seeds if stochastic |
| `ari` | yes | primary metric (use `--score-col` for a different column) |
| `seconds` | no | wall time |
| `status` | no | `success` / `failed` / `timeout` |
| `n_domains_truth` | no | mapped to `n_domains`; document oracle-K use |

## Step 3 — Write a sidecar

The sidecar supplies bundle-level metadata that isn't per-row. **`node_id` must match your
registry entry** and `dataset_meta` must include `ground_truth_kind`:

```json
{
  "node_id": "lab-cambridge",
  "task": "spatial_domain",
  "metric": "ARI",
  "higher_is_better": true,
  "method": {
    "name": "my_lab_model",
    "version": "0.3.1",
    "wraps": "MyLabModel",
    "url": "https://github.com/cam-lab/my-lab-model"
  },
  "dataset_meta": {
    "151673": {
      "platform": "Visium",
      "tissue": "DLPFC",
      "task": "spatial_domain",
      "ground_truth_kind": "spatial_domain",
      "label_key": "domain_truth",
      "dataset_visibility": "public",
      "n_domains": 7
    }
  },
  "environment": {"os": "linux", "python": "3.11"}
}
```

## Step 4 — Sign an evidence bundle

```bash
histoweave fed sign \
  --in benchmark_long.csv \
  --sidecar sidecar.json \
  --key federation/lab-cambridge.key.json \
  --out bundle.json
```

The bundle is content-hashed and signed. Any later edit to any record invalidates both the
hash and the signature. If the schema, task contract, or privacy gate rejects your input,
`sign` prints the reason and exits non-zero — fix the data, don't work around the gate.

## Step 5 — Verify locally (optional but recommended)

```bash
histoweave fed verify bundle.json --nodes-dir federation/nodes
# VALID: bundle <id> verified for node lab-cambridge (scheme=ed25519)   -> exit 0
```

## Step 6 — Publish + register

You have two transport options (both are "you own and sign your own evidence"):

- **Host it yourself (preferred).** Put `bundle.json` at the `evidence_feed` URL you
  declared. The feed may be a single bundle, a JSON list of bundles, or an index object
  with a `"bundles"` array.
- **Contribute by PR.** If you can't host, include the bundle in your PR and point the feed
  at its path.

Then open a PR that adds **only** `federation/nodes/lab-cambridge.json` (the public entry).
A maintainer reviews it into the allowlist. **Do not** include your `.key.json`.

## Step 7 — What happens after merge

The aggregator (repo CI / a maintainer) runs:

```bash
histoweave fed pull      --nodes-dir federation/nodes --store federation/evidence_store.jsonl
histoweave fed consensus --store federation/evidence_store.jsonl --out federation/consensus.json
```

Your scores enter the append-only store and the consensus view is rebuilt. Your new cells
start as **`unverified`**. When another node independently reports the same
(dataset, method) cell within tolerance (`|Δ ARI| ≤ 0.05` by default), the cell is upgraded
to **`verified`**; irreconcilable disagreement marks it **`disputed`** (resolvable later by
a further agreeing report). The public leaderboard then shows the federation status.

## Hard rules (same spirit as the external submission protocol)

1. **Task contracts.** `spatial_domain` scores must come from expert spatial partitions.
   Leiden / Louvain / self-clustering labels are rejected.
2. **No silent fallbacks.** A failed backend sets `status=failed` and leaves `ari` empty —
   never substitute a toy method under the same name.
3. **Seeds.** Report at least 3 seeds for stochastic methods.
4. **Oracle-K.** Document whether `n_domains` was the true domain count.
5. **Raw data never leaves.** Only scores + public dataset descriptors. `dataset_visibility`
   must be `public` in v1; the privacy gate rejects coordinate/expression/matrix-like
   payloads.
6. **Keys.** Never commit `federation/*.key.json`. Rotate by adding a new active key and
   revoking the old one.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `sign` exits 2, "sidecar must include 'node_id'" | Add `node_id` to the sidecar. |
| `pull` rejects a bundle: `node_id '…' != registry '…'` | The bundle's `node_id` must match the registry filename/entry it's pulled under. |
| `verify` exits 1 (INVALID) | Tampered bundle, wrong key, or node not in the registry. |
| `verify` exits 3 (SKIP) | Bundle is sigstore-signed but the `sigstore` extra isn't installed. |
| Privacy gate rejection | A field name or a long numeric array looks like raw data — remove it; contribute scores only. |
