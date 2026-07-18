# spagcn

**Category:** domain detection · **Maturity:** validated · **Implementation:** external (SpaGCN)

Official SpaGCN graph-convolutional spatial domains.

## When to use

- Visium-scale domain detection where you want a **published SOTA** comparator.
- Benchmark tracks that must include field-standard spatial methods.

## When not to use

- Backend not installed �?HistoWeave **fails closed** (no toy substitute).
- Non-Visium geometries without validating the adjacency recipe.

## Failure modes

- Missing SpaGCN / torch / scanpy �?`ModuleNotFoundError`.
- Histology-free mode only in the default wrapper (image-aware SpaGCN needs extra setup).
- Mutually incompatible dependency pins vs GraphST/STAGATE �?use isolated envs.

## Evidence

Multi-dataset validated (`histoweave.sota_dlpfc.v1` + `sota_batch.v1`):
official SpaGCN on 5 DLPFC slices × 3 seeds, mean ARI ≈ **0.32**.
Formal report: [validation/spagcn.md](validation/spagcn.md).
