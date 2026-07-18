# Method validation reports (multi-dataset)

**Protocol family:** `histoweave.method_validation.*`

HistoWeave separates two multi-dataset evidence kinds (do **not** conflate them):

| Kind | Maturity | Count | Meaning |
|------|----------|------:|---------|
| **Scientific** | `validated` | **10** | Concordance vs independent ground truth on real multi-dataset grids (e.g. ARI) |
| **Contract** | `contract_validated` | **3** | Interface / mock / fail-closed structural gates across datasets (CI-safe) |
| **Total evidence packages** | — | **13** | `10 + 3` — single ledger in `release_manifest.VALIDATION_EVIDENCE` |

Canonical sets: `SCIENTIFIC_VALIDATED_METHODS`, `CONTRACT_VALIDATED_METHODS`,
`MULTI_DATASET_EVIDENCE_METHODS` in
[`release_manifest.py`](https://github.com/histoweave-spatial/histoweave/blob/main/src/histoweave/plugins/builtin/release_manifest.py).

## Scientific validated (10)

| Method | Category | Decision | Report |
|--------|----------|----------|--------|
| `agglomerative` | domain_detection | **validated** | [report](agglomerative.md) |
| `banksy` | domain_detection | **validated** | [report](banksy.md) |
| `banksy_py` | domain_detection | **validated** | [report](banksy_py.md) |
| `birch` | domain_detection | **validated** | [report](birch.md) |
| `gaussian_mixture` | domain_detection | **validated** | [report](gaussian_mixture.md) |
| `graphst` | domain_detection | **validated** | [report](graphst.md) |
| `minibatch_kmeans` | domain_detection | **validated** | [report](minibatch_kmeans.md) |
| `spagcn` | domain_detection | **validated** | [report](spagcn.md) |
| `spectral` | domain_detection | **validated** | [report](spectral.md) |
| `stagate` | domain_detection | **validated** | [report](stagate.md) |

## Contract validated (3)

| Method | Category | Decision | Report |
|--------|----------|----------|--------|
| `cell2location` | deconvolution | **contract_validated** | [report](cell2location.md) |
| `rctd` | deconvolution | **contract_validated** | [report](rctd.md) |
| `spatialde` | svg | **contract_validated** | [report](spatialde.md) |

Contract-validated methods wrap real upstream libraries and pass multi-dataset
**I/O / fail-closed / structural** gates (often with mock backends in CI). They are
**not** claimed as scientifically concordant until real multi-dataset label metrics
pass the scientific gate.

## Batch narrative

See `research/method_validation/results/VALIDATION_BATCH_REPORT.md` and the
[validation protocol](https://github.com/histoweave-spatial/histoweave/blob/main/research/method_validation/PROTOCOL.md).

## Related

- [Method guide index](../index.md)
- [Method lifecycle](../../method-lifecycle.md)
- [Release manifest](https://github.com/histoweave-spatial/histoweave/blob/main/src/histoweave/plugins/builtin/release_manifest.py)
