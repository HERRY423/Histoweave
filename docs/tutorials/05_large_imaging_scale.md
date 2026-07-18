# Tutorial 5 — Large imaging tables (10⁵ cells)

Sequencing Visium slides fit in a few GB. **Xenium / MERFISH full slides** do
not. This tutorial shows the HistoWeave **scale contract** path: sparse I/O,
planned subsampling, and domain methods that stay inside a workstation envelope.

!!! note "What you need"
    ```bash
    pip install "histoweave-spatial[io,spatial,scanpy]"
    ```
    Optional: a prepared registry bundle such as `xenium_breast_cancer` or
    `merfish_mouse_brain` under `datasets_cache/`.

## 1. Inspect the scale plan

```python
import logging
from histoweave.datasets import get_dataset
from histoweave.datasets.scale_contract import scale_contract_for_assay, registry_scale_table

log = logging.getLogger("histoweave.tutorial")

# Whole registry with recommended subsample ceilings
for row in registry_scale_table():
    if row["assay"] in {"xenium", "merfish"}:
        log.info(
            "%s n_obs=%s subsample=%s RAM~%sGB",
            row["dataset"], row["n_obs"], row["subsample_to"], row["peak_ram_gb_estimate"],
        )

entry = get_dataset("merfish_mouse_brain")
contract = scale_contract_for_assay(entry.assay, entry.n_obs)
log.info("plan=%s", contract.plan_for(entry.n_obs))
```

Typical output for atlas-scale MERFISH:

* `sparse_required: True`
* `subsample_to: 40000`
* `knn_backend: cKDTree` (no O(n²) densify in the main path)

## 2. Load sparsely and subsample for domain sweeps

```python
from histoweave.datasets import get_dataset
from histoweave.plugins import create_method
import numpy as np

entry = get_dataset("xenium_breast_cancer")  # or merfish_mouse_brain
# load() may raise if the local bundle is not prepared — that is intentional.
try:
    data = entry.load()
except FileNotFoundError as exc:
    raise SystemExit(f"Prepare the bundle first: {exc}") from exc

plan = scale_contract_for_assay(entry.assay, data.n_obs).plan_for(data.n_obs)
if plan["subsample_to"] and data.n_obs > plan["subsample_to"]:
    rng = np.random.default_rng(0)
    keep = rng.choice(data.n_obs, size=plan["subsample_to"], replace=False)
    mask = np.zeros(data.n_obs, dtype=bool)
    mask[keep] = True
    data = data.subset_obs(mask)

# Imaging-friendly transform
data = create_method("normalization", "arcsinh_transform").run(data)
data.uns["n_domains"] = int(data.uns.get("n_domains") or 12)
result = create_method(
    "domain_detection", "banksy_py", n_domains=data.uns["n_domains"], lambda_param=0.5
).run(data)
```

## 3. Multi-method review without loading three full copies

Run two light methods and push both partitions into the report uncertainty map:

```python
from histoweave.report import build_report

gmm = create_method(
    "domain_detection", "gaussian_mixture",
    n_domains=data.uns["n_domains"], spatial_weight=0.3, random_state=0,
).run(result)

result.uns["method_predictions"] = {
    "banksy_py": result.obs["domain"].astype(str).to_numpy(),
    "gaussian_mixture": gmm.obs["domain"].astype(str).to_numpy(),
}
result.obs["domain_alt"] = gmm.obs["domain"].values
build_report(result, "large_imaging_report.html")
```

Open the HTML report: the **Boundary uncertainty** section highlights cells
where the two methods disagree — review priority for pathologists.

## 4. Task contract reminder

| Dataset | Task | Do not do |
|---------|------|-----------|
| `merfish_mouse_brain` | `spatial_domain` (CCF anatomy) | Compare ARI to Xenium cell-type labels |
| `xenium_breast_cancer` | `cell_type` | Put it in a spatial-domain leaderboard |
| `xenium_human_lymph_node` | `spatial_domain` (pathology polygons) | Use Leiden as GT |

## 5. Throughput expectations

Use `histoweave scale` for synthetic ceilings, and
`python scripts/run_sota_dlpfc.py --dry-run` for SOTA backend availability.
Domain sklearn methods typically OOM near 1e5 densified cells; keep the
expression matrix sparse until the method boundary, and subsample for sweeps.
