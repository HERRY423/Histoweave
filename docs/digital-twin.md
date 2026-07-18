# Digital-twin synthetic validation

When a user uploads a **real spatial sample without ground truth**, HistoWeave cannot
score methods with ARI against biology.  *Digital-twin validation* builds a synthetic
**statistical twin** of that sample — matching target-free structure, planting known
labels — benchmarks methods on the twin, and returns the ranking as a **predicted
ranking** for the real sample.

## What is matched?

By default the twin matches **13 target-free dimensions**
(`histoweave.datasets.TWIN_MATCH_FEATURES`):

| Dimension | Feature key |
|-----------|-------------|
| Sparsity | `sparsity` |
| Mean non-zero expression | `mean_nonzero` |
| Library size mean | `library_mean` |
| Library size CV | `library_cv` |
| Expression entropy | `expression_entropy` |
| Moran's I (spatial autocorrelation) | `spatial_autocorrelation` |
| Hopkins cluster tendency | `cluster_tendency` |
| Mean NN distance | `mean_nn_distance` |
| Spatial density CV | `spatial_density_cv` |
| Spatial entropy | `spatial_entropy` |
| Effective rank 90% | `effective_rank_90` |
| Effective rank 95% | `effective_rank_95` |
| Singular-value entropy | `sv_entropy` |

Size-like quantities (`n_obs`, `n_vars`) are matched by construction (with optional
caps for large slides).  Domain labels are **planted**, not copied from biology.

## Pipeline

```
real sample (no GT)
       │
       ▼
 extract 13-D target-free features
       │
       ▼
 random-search synthetic knobs + expression calibration
 (reuse real coordinates → preserve geometry / Moran's I)
       │
       ▼
 twin with obs['domain_truth']
       │
       ▼
 run_benchmark(domain_detection) on twin
       │
       ▼
 predicted method ranking  +  HTML report
```

## CLI

```bash
# Ingest a real sample (or demo)
histoweave ingest --demo --out data.ttab

# Twin validation
histoweave digital-twin \
  --in data.ttab \
  --out-dir digital_twin_out \
  --methods kmeans,spectral,gaussian_mixture \
  --n-trials 16
```

Artifacts in `--out-dir`:

- `digital_twin_validation.json` — full machine-readable result
- `twin_match.json` — per-feature match errors
- `leaderboard.json` — predicted ranking
- `digital_twin_report.html` — human report

## Python API

```python
import logging

import histoweave as ts
from histoweave.benchmark import run_digital_twin_validation
from histoweave.datasets import make_digital_twin

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

data = ts.datasets.make_synthetic(seed=0)  # stand-in for a real upload
# data.obs has no domain_truth in production uploads

twin_pack = make_digital_twin(data, seed=0, n_trials=12)
_LOGGER.info("%s", twin_pack.match.summary())

result = run_digital_twin_validation(
    data,
    methods=["kmeans", "spectral"],
    out_dir="digital_twin_out",
)
_LOGGER.info("%s", result.summary())
_LOGGER.info("Predicted best: %s", result.best_method())
```

## Scientific caveats

Twin ARI is a **proxy ranking under statistical matching**.  It is *not* a claim that
twin performance equals biological accuracy.  Prefer:

1. Twin ranking as a shortlist for the real sample.
2. Spatial AutoML multi-method consensus on the real sample.
3. Expert review / orthogonal assays when available.

Low match cosine (`< 0.85`) or high match L2 (`> 2`) triggers explicit warnings.
