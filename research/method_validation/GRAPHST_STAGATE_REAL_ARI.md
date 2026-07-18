# Real GraphST / STAGATE ARI (official backends)

**Protocol:** histoweave.sota_dlpfc.v1
**Settings:** max_obs=1000, GraphST epochs=120, STAGATE epochs=150, seeds={42,1,2}, CPU

## Headline

| Method | n success | mean ARI | std | best slice | worst slice |
|--------|----------:|---------:|----:|------------|-------------|
| GraphST | 15/15 | **0.121** | 0.038 | 151673 (0.161) | 151669 (0.065) |
| STAGATE | 15/15 | **0.285** | 0.094 | 151507 (0.432) | 151669 (0.164) |
| SpaGCN (prior SOTA grid) | 15/15 | **0.317** | 0.099 | 151674 (0.396) | 151669 (0.199) |

## Per-slice mean ARI

| Slice | GraphST | STAGATE |
|-------|--------:|--------:|
| 151507 | 0.127 | **0.432** |
| 151669 | 0.065 | 0.164 |
| 151670 | 0.097 | 0.253 |
| 151673 | 0.161 | 0.294 |
| 151674 | 0.152 | 0.282 |

## Engineering notes

- GraphST: fixed import path GraphST.GraphST.GraphST; requires scikit-misc for seurat_v3 HVG.
- STAGATE: Windows torch-sparse binary often fails to load against mismatched torch; site-packages gat_conv.py uses a soft-import shim so edge_index training works without SparseTensor.
- Artifacts:
esults/graphst_stagate_real_ari.json, checkpoints under 5x15_spatial_aware/checkpoints/sota_{graphst,stagate}__*.

## Reproduce

`ash
python research/method_validation/run_real_graphst_stagate_ari.py --methods graphst,stagate --max-obs 1000
python research/method_validation/merge_real_ari_checkpoints.py
python research/method_validation/compile_validation_evidence.py
`
