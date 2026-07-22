# HistoWeave Federated Evidence Network — Protocol v1

**Status:** reference implementation (v1). **Schema versions:** `histoweave.evidence.v1`,
`histoweave.node_registry.v1`, `histoweave.consensus.v1`.

## 1. Problem

The HistoWeave benchmark is centrally stored, and external validation today is
**one-shot and local**: a lab runs a method on its own machine, and the result never
re-enters the shared evidence base in a comparable, trustable form. This protocol turns
that one-shot validation into a **living, continuously growing evidence landscape** that
multiple labs extend **without sharing raw data**.

## 2. Design goals

1. **No raw data leaves a lab.** Only scalar benchmark scores plus registry-style dataset
   descriptors are ever transmitted. A privacy gate actively rejects raw-data-shaped
   payloads (coordinate/expression/matrix-like keys, long numeric arrays).
2. **Trust is earned, not asserted.** A self-reported score is accepted immediately but
   marked `unverified`. It is upgraded to `verified` only when an *independent* node
   reproduces the same cell within tolerance, and flagged `disputed` when reproductions
   disagree irreconcilably.
3. **Append-only truth + derived consensus.** The raw evidence store is append-only and
   never mutated (corrections are new lines); the consensus view is a pure function of the
   store and can always be rebuilt.
4. **Git/GitHub-native transport (v1).** Labs host and sign their own evidence; the node
   registry is a maintainer-curated allowlist reviewed by PR. No new servers.
5. **Additive and versioned.** Existing tests, the leaderboard, and the recommender
   knowledge base all keep working with **no** federation files present.

## 3. Roles

| Role | Responsibility |
|------|----------------|
| **Contributing node (lab)** | Runs benchmarks locally, signs an evidence bundle with its private key, hosts the bundle at a URL it controls (or contributes it by PR), and registers a public key + feed URL in the node registry. |
| **Aggregator (maintainer)** | Reviews node-registry PRs (the allowlist), pulls registered feeds into the append-only store, rebuilds the consensus view, and publishes the leaderboard. In v1 this is the HistoWeave repo's CI. |
| **Consumer** | Reads the consensus view / leaderboard, or builds a recommender knowledge base from it. |

> **"Decentralized" in v1** means labs *own, host, and sign* their own evidence; the index
> and consensus are aggregated at a maintainer-governed merge point. Fully serverless / P2P
> transport is a v2 swap enabled by the abstracted transport layer in `registry.py`
> (`fetch_feed` already accepts `https://`, `file://`, and glob paths).

## 4. Data model

### 4.1 Evidence bundle (`histoweave.evidence.v1`)

A signed, content-addressed collection of benchmark records from one node for one task.

```
EvidenceBundle
├── schema_version   "histoweave.evidence.v1"
├── bundle_id        uuid4 (transport id; NOT part of the content hash)
├── node_id          must match the signing node's registry entry
├── task             e.g. "spatial_domain"
├── metric           "ARI" (default); higher_is_better: bool
├── records[]        one per dataset × method × config × seed
│   ├── dataset, method, config, seed
│   ├── score / ari  the scalar metric only
│   ├── seconds, status, error, n_domains, oracle_k
├── dataset_meta{}   registry-style descriptors per dataset
│   └── REQUIRES ground_truth_kind; dataset_visibility must be "public" (v1)
├── method           optional MethodInfo (name, version, wraps, commit, digest, url)
├── environment{}    os / python / package versions (free-form, non-raw)
├── content_hash     sha256 over the canonical payload (excludes hash + signature)
└── signature        {scheme, value, public_key_id, certificate?}
```

**Content addressing.** `content_hash` is a SHA-256 over the canonical-JSON payload with the
`content_hash` and `signature` fields removed. Canonical JSON uses sorted keys and compact
separators, so the hash is stable under key reordering. The signature signs the
`content_hash` string, so any mutation to any record invalidates both the hash and the
signature.

**Contract enforcement (delegates to the task contract).** A `spatial_domain` bundle may
only score against expert spatial partitions. Bundles are rejected when they:
- carry a `label_key` in `{leiden, louvain, proxy_leiden, self_cluster_labels}` for a
  ground-truth task,
- report `oracle_k` without an accompanying note,
- set `status=failed` while still carrying a score,
- contain unknown record keys, or
- declare `dataset_visibility` other than `public` (v1).

**Privacy gate.** Independently of the contract, the bundle is rejected if any field name
matches the raw-data denylist (`counts`, `coords`, `coordinate`, `spatial`, `expression`,
`matrix`, `adata`, `obsm`, `layers`, `barcode`, `cell_id`, `spot_id`, `raw_x`, …) or embeds
a numeric array longer than 8 elements. This is a structural guard, **not** a formal
(differential-privacy) guarantee.

### 4.2 Node registry entry (`histoweave.node_registry.v1`)

One JSON file per node under `federation/nodes/<node_id>.json`, added by PR (the allowlist).

```
NodeRegistryEntry
├── schema_version    "histoweave.node_registry.v1"
├── node_id           stable slug (e.g. "lab-cambridge")
├── display_name      human-readable
├── evidence_feed[]   one or more URLs/paths where this node's bundles live
├── public_keys[]     {scheme, id, value, status}  (ed25519 and/or sigstore)
├── sigstore_identity optional OIDC identity for keyless verification
├── contact           optional
├── added_at          ISO-8601
└── status            "active" | "revoked"
```

Keys are **rotated by adding a new active key and marking the old one revoked** — never by
deleting history. Verification uses only `active` keys.

### 4.3 Consensus view (`histoweave.consensus.v1`)

A derived, rebuildable summary. One `ConsensusCell` per (task, dataset, method):

```
ConsensusCell
├── task, dataset, method, metric, higher_is_better
├── n_labs, n_records, n_seeds
├── consensus_score      median of per-lab means (robust to a single outlier lab)
├── mean, mad, spread    dispersion across lab means
├── cross_lab_ci         bootstrap CI over lab means
├── reproducibility      fraction of labs within tolerance of the median
├── verification_status  unverified | verified | disputed
├── node_ids / labs      contributors
├── outlier_node_ids     labs outside tolerance
└── dataset_meta
```

## 5. Trust state machine

```
             1 lab reports                 2nd independent lab
   (none) ─────────────────▶ unverified ─────────────────────▶ verified
                                  │        agrees within tol
                                  │
                                  │        2nd+ lab disagrees
                                  └─────────────────────────▶ disputed
                                           (irreconcilable)         │
                                                                    │ a later lab
                                                                    │ agrees within tol
                                                                    ▼
                                                                 verified
```

- **Tolerance** is `|Δ metric| ≤ 0.05` by default (ARI), configurable per task via
  `per_task_tolerance`.
- A cell with **1 lab** is always `unverified` (its CI collapses to a point).
- A cell with **≥2 labs** is `verified` when ≥2 lab means fall within tolerance of the
  median, otherwise `disputed`.
- Disputes are **resolvable**: because the store is append-only, a later reproducing report
  can move a cell from `disputed` back to `verified`.

## 6. Reference workflow (`histoweave fed`)

```bash
# 1. A lab creates its identity + registry entry (private key stays local, never committed)
histoweave fed init-node --node-id lab-cambridge --display-name "Cambridge Lab" \
    --contact team@cam.example --evidence-feed https://cam.example/histoweave/bundle.json

# 2. Sign local benchmark results into an evidence bundle
histoweave fed sign --in 5x10_dlpfc_benchmark/benchmark_long.csv \
    --sidecar sidecar.json --key federation/lab-cambridge.key.json \
    --out bundle.json

# 3. Anyone can offline-verify a bundle against the registry
histoweave fed verify bundle.json --nodes-dir federation/nodes

# 4. The aggregator pulls all registered feeds into the append-only store
histoweave fed pull --nodes-dir federation/nodes --store federation/evidence_store.jsonl

# 5. Rebuild the derived consensus view (and check it still drives the landscape)
histoweave fed consensus --store federation/evidence_store.jsonl \
    --out federation/consensus.json --check-landscape

# 6. Inspect the living landscape
histoweave fed status --store federation/evidence_store.jsonl --nodes-dir federation/nodes
```

**Exit codes.** `verify` → `0` VALID / `1` INVALID / `3` SKIP (sigstore unavailable);
`pull` → `1` if any bundle was rejected; `consensus` / `status` → `2` if no store exists.

## 7. Identity backends

- **Ed25519 (default).** Offline, deterministic, no network. Requires the `cryptography`
  library (shipped in the `federation` extra).
- **Sigstore / OIDC (opt-in).** Keyless verification via a guarded import; if the extra is
  absent the verifier **degrades gracefully** (skips, never crashes). Pluggable through the
  verifier interface in `signing.py`.

## 8. Leaderboard integration (additive, v3)

When `federation/consensus.json` is present, `leaderboard/generate.py` enriches matching
records with `verification_status`, `n_labs`, `cross_lab_ci`, `reproducibility`, and
`contributor_node_ids`, bumps the feed protocol to `histoweave.leaderboard.v3`, and adds a
top-level `federation` block. With **no** federation files the feed stays at v2 and records
are byte-for-byte identical to today's — federation is purely additive.

## 9. Threat model & limitations (v1)

- **Sybil / fake labs.** Mitigated by the maintainer-curated allowlist (registry PRs), not
  by proof-of-work or stake. A node with no track record contributes only `unverified`
  cells until independently reproduced.
- **Score fabrication.** A lab can sign a false score; it stays `unverified` until another
  node reproduces it. Auto-verification by re-execution is a v2 item.
- **Privacy.** The gate blocks raw-data-shaped payloads but is **not** a formal DP
  guarantee; only scores + public dataset descriptors are intended to cross the boundary.
- **Not enabled in v1:** private-tissue aggregate contribution (schema-forward-compatible
  only), reputation weighting / slashing, and content-addressed P2P transport.

## 10. Versioning & governance

All schemas are explicitly versioned; consumers must ignore unknown additive fields.
Breaking changes bump the schema version. The node allowlist, the reproduction tolerance,
and the merge point are maintainer-governed. Ed25519 **private** keys must never enter the
repository (`.gitignore` blocks `federation/*.key.json`).

See `CONTRIBUTING_EVIDENCE.md` for the step-by-step lab onboarding guide.
