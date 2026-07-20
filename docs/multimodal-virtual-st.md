# Multimodal tasks and virtual spatial transcriptomics

HistoWeave treats **analysis tasks as hard scientific contracts**. Domain recovery
on RNA, protein, and chromatin are *related but non-transferable* questions.
Predicting spatial expression from H&E (**virtual ST**) is a separate question
again, with its own ground-truth kind and metric.

This page documents:

1. the multimodal `AnalysisTask` surface,
2. executable **cross-modal evidence** rules used by `decide()` / `recommend()`,
3. the **virtual_st** method category and how to run it on **real H&E**.

## Multimodal analysis tasks

| `AnalysisTask` | Scientific question | Primary ground truth | Default metric |
|---|---|---|---|
| `spatial_domain` | Recover spatial partitions from RNA/ST | expert spatial domains | ARI |
| `spatial_protein_domain` | Recover spatial partitions from protein imaging | protein-domain labels | ARI |
| `spatial_chromatin_domain` | Recover spatial partitions from chromatin assays | chromatin-domain labels | ARI |
| `cell_type` | Recover cell types / states | cell-type (or proxy) labels | ARI |
| `svg` | Rank spatially variable features | marker sets / none | precision@k |
| `deconvolution` | Estimate cell-type proportions | proportions | 1 − RMSD |
| `virtual_st` | Predict expression from histology (H&E→ST) | **measured expression** | mean gene Pearson |

```python
from histoweave.benchmark import AnalysisTask, GroundTruthKind, TaskContract

TaskContract(
    task=AnalysisTask.SPATIAL_PROTEIN_DOMAIN,
    ground_truth_kind=GroundTruthKind.SPATIAL_PROTEIN_DOMAIN,
    label_key="protein_domain_truth",
).validate()

TaskContract(
    task=AnalysisTask.VIRTUAL_ST,
    ground_truth_kind=GroundTruthKind.MEASURED_EXPRESSION,
    label_key="X",                 # measured matrix slot or layer name
    metric="mean_gene_pearson",
).validate()
```

## Cross-modal evidence compatibility

Recommendation and decision engines **never soft-weight** incompatible evidence.

| Relation | Example | Admitted into ranking? |
|---|---|---|
| `SAME` | `spatial_domain` ↔ `spatial_domain` | **Yes** |
| `SAME_FAMILY` | `spatial_domain` ↔ `spatial_protein_domain` | **No** (audit only) |
| `INCOMPATIBLE` | `virtual_st` ↔ `spatial_domain` | **No** |

```python
from histoweave.benchmark import (
    cross_modal_relation,
    tasks_admissible,
    evidence_compatibility_report,
    CrossModalRelation,
)

assert tasks_admissible("spatial_domain", "spatial_domain")
assert not tasks_admissible("spatial_domain", "spatial_protein_domain")
assert cross_modal_relation("spatial_domain", "spatial_protein_domain") is (
    CrossModalRelation.SAME_FAMILY
)

report = evidence_compatibility_report(
    "spatial_domain",
    "spatial_protein_domain",
    reference_ground_truth_kind="spatial_protein_domain",
)
assert report["admissible"] is False
```

Rationale: methods operate on different molecular matrices. A ranking learned on
CODEX protein domains does not justify a Visium RNA domain method, even though
both recover spatial partitions. Virtual ST (image → expression) is isolated
from all domain-partition rankings.

These gates are enforced in:

* `MethodRecommender._find_neighbours` (zero boost for non-admissible refs)
* `DecisionEngine` / `_task_evidence_status` (FAIL when no task-matched evidence)

## Virtual ST methods

Category: `MethodCategory.VIRTUAL_ST`.

| Method | Design lineage | Notes |
|---|---|---|
| `virtual_st_morphology` | Patch-statistic baseline | Fast, no deep learning |
| `virtual_st_scellst` | sCellST-style weak supervision (Chadoutaud et al., *Nat Commun* 2026) | Paired mode fits log1p targets |
| `virtual_st_storm` | STORM-inspired hierarchical morphology + spatial fusion (2026) | Neighbourhood-aware features |

These are **contract-stable reference implementations** for evaluation and
orchestration. They do not ship multi-million-parameter foundation checkpoints;
external backends can replace the native path without changing the task contract.

### Modes

| `mode` | Behaviour |
|---|---|
| `auto` | `paired` if measured expression is present, else `inference` |
| `paired` | Fit morphology → expression on measured ST (evaluation path) |
| `inference` | Morphology-only pseudo-expression (no measured targets) |

### Outputs

* `layers['virtual_st']` — predicted expression
* `obsm['X_virtual_st']` — morphology embedding
* `uns['virtual_st'][<method>]` — `mean_gene_pearson`, gene indices, supervision flag

## Real H&E data path

### 1. Public Visium mouse brain (recommended)

10x Visium V1 Adult Mouse Brain with registered H&E, via squidpy:

```python
from histoweave.datasets import load_visium_hne_paired
from histoweave.plugins import MethodCategory, create_method
from histoweave.plugins.builtin import register_all

register_all()

# Downloads once (or uses data/anndata/visium_hne_adata.h5ad / squidpy cache).
data = load_visium_hne_paired(prefer="lowres", n_hvg=2000)
assert "image" in data.images
assert data.spatial is not None

result = create_method(
    MethodCategory.VIRTUAL_ST,
    "virtual_st_storm",
    mode="paired",
    image_key="image",
    n_genes=64,
    seed=0,
).run(data)

meta = result.uns["virtual_st"]["virtual_st_storm"]
print(meta["mean_gene_pearson"], result.layers["virtual_st"].shape)
```

### 2. Registry entry (after prepare)

```bash
python scripts/prepare_visium_hne_virtual_st.py \
  --source data/anndata/visium_hne_adata.h5ad
```

```python
from histoweave.datasets import get_dataset, ensure_histology, prepare_virtual_st_table

entry = get_dataset("visium_mouse_brain_hne")
data = entry.load()                 # attaches images from uns['spatial'] when present
data = prepare_virtual_st_table(data)
print(entry.task_contract())        # virtual_st + measured_expression + mean_gene_pearson
```

### 3. Your own Space Ranger Visium folder

```python
from histoweave.io import read
from histoweave.datasets import prepare_virtual_st_table
from histoweave.plugins import MethodCategory, create_method
from histoweave.plugins.builtin import register_all

register_all()

# Native Visium reader attaches tissue_lowres/hires PNGs when present.
data = read("visium", "/path/to/outs")
data = prepare_virtual_st_table(data, spatial_dir="/path/to/outs/spatial")

result = create_method(
    MethodCategory.VIRTUAL_ST,
    "virtual_st_scellst",
    mode="paired",
    image_key="image",
).run(data)
```

### 4. Benchmark harness

```python
from histoweave.benchmark import get_task, run_benchmark
from histoweave.datasets import load_visium_hne_paired

data = load_visium_hne_paired(n_hvg=500, prefer="lowres")
task = get_task("virtual_st", dataset=data)
board = run_benchmark(
    task,
    methods=["virtual_st_morphology", "virtual_st_scellst", "virtual_st_storm"],
    method_params={
        name: {"mode": "paired", "n_genes": 32, "seed": 0}
        for name in [
            "virtual_st_morphology",
            "virtual_st_scellst",
            "virtual_st_storm",
        ]
    },
)
print(board.leaderboard)
```

## Claim boundary

* Virtual ST scores **expression agreement** (Pearson / Spearman), not biological
  domain correctness.
* Positive synthetic morphology–expression coupling does **not** transfer to a
  claim of clinical-grade H&E→ST foundation performance.
* Cross-modal domain evidence remains rejected even when histology is available.

See also: [decision protocol](decision-protocol.md), [tutorial 06](tutorials/06_virtual_st_he.md),
[method selection](method-selection.md).
