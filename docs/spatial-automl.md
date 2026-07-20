# Spatial AutoML execution adapter

This is a product-facing execution adapter for the
[evidence-governed decision protocol](decision-protocol.md), not a separate
method-selection claim. It joins:

1. **`histoweave ask`** — natural-language → validated pipeline plan (LLM compiler).
2. **Decision protocol** — compatible-evidence retrieval, fallback/abstention,
   and canonical set-valued comparison.

into a loop that **audits, runs an allowed comparison panel, and reports** methods.

## User story

> “为我我的 Visium 肝癌数据找到空间域。”  
> “Find spatial domains for my Visium liver cancer data.”

```bash
histoweave automl \
  "Find spatial domains for my Visium liver cancer data." \
  --in sample.ttab \
  --knowledge-base figure3_results/landscape.json \
  --out-dir automl_out \
  --top 3 \
  --platform visium
```

## Pipeline

```
1. Extract target-free features from the user sample
2. Hard-filter incompatible task and ground-truth evidence
3. Build a DecisionCard: fallback, comparison panel, or abstention
4. Execute only the allowed panel (after log1p_cp10k prep)
5. Compare with proxy quality metrics (no GT required):
     • spatial coherence (kNN label agreement)
     • expression silhouette
     • cross-method consensus ARI
     • runtime
6. Use the canonical Pareto implementation (quality × seconds × reference proxy)
7. Emit decision_card.json and automl_report.html
```

An advisory compiler plan is attached by default (`--model mock` offline).
Pass `--no-compiler` to skip the NL plan.

Spatial coherence, silhouette, and consensus are post-run diagnostics; none is
ground truth. An `evidence_required` action runs a panel to acquire evidence and
must not be reported as an automatically validated winner.

## CLI options

| Flag | Meaning |
|------|---------|
| `question` | Natural-language request (positional) |
| `--in` | Input `.ttab` bundle |
| `--knowledge-base` | Landscape JSON from a prior benchmark |
| `--top` | How many recommended methods to run (default 3) |
| `--methods` | Optional explicit method list (overrides recommender) |
| `--platform` | Platform prior (`visium`, `xenium`, …) |
| `--out-dir` | Artifacts directory |
| `--no-compiler` | Skip `histoweave ask` plan |
| `--json` | Print full result JSON |

## Python API

```python
import logging

from histoweave.automl import run_spatial_automl
from histoweave.datasets import make_synthetic

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

data = make_synthetic(seed=0)
result = run_spatial_automl(
    data,
    "Find spatial domains for my Visium data.",
    knowledge_base="figure3_results/landscape.json",
    top_k=3,
    platform="visium",
    out_dir="automl_out",
)
_LOGGER.info("%s", result.summary())
_LOGGER.info("Pareto-preferred: %s", result.best_method())
```

## Pareto ranking

Without ground truth, AutoML maximises:

| Objective | Definition |
|-----------|------------|
| **quality** | Mean of spatial coherence, scaled silhouette, consensus ARI |
| **speed** | `1 / (1 + seconds)` |
| **recommendation** | Landscape neighbour-weighted score |

Non-dominated methods form the **Pareto front** (badge in the HTML report).
A scalarised z-score breaks ties for a single “preferred” method.

## How this relates to digital-twin validation

| | Digital twin | Spatial AutoML |
|--|--------------|----------------|
| Runs on | Synthetic twin | Real sample |
| Ground truth | Planted labels | None |
| Ranking signal | Twin ARI | Proxy quality + consensus + runtime |
| Best for | Method shortlist with a score | Production multi-method report |

They compose well: twin ranking → AutoML method list → real-sample consensus report.

## Knowledge base

Use any landscape JSON produced by HistoWeave landscape runs
(e.g. `figure3_results/landscape.json`, `5x15_spatial_aware/landscape.json`)
or `MethodRecommender(...).save_knowledge_base(path)`.
