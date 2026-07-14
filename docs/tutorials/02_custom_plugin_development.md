# Tutorial 2 — Developing a custom method plugin

HistoWeave's core idea: a method is *just* a class that declares a `MethodSpec` and
implements `run(self, data) -> SpatialTable`. Once registered, it gets parameter
validation, provenance, the CLI, reporting, and benchmarking **for free** — and it can
be benchmarked head-to-head against the 51 built-in methods with no extra glue.

This tutorial builds a real, useful plugin — a **graph-Laplacian spatial smoother** that
denoises expression using the spatial neighbourhood — and then a second one that wraps
an external library. You will learn the plugin contract, how to ship a plugin as its own
pip-installable package, and how to benchmark it.

## The plugin contract

Every method subclasses `histoweave.plugins.Method`:

```python
from histoweave.plugins import Method, MethodCategory, MethodSpec, ParamSpec, register

@register                        # adds the class to the global registry
class MyMethod(Method):
    spec = MethodSpec(
        name="my_method",        # unique within its category
        category=MethodCategory.NORMALIZATION,
        version="0.1.0",
        summary="One-line description.",
        params=(
            ParamSpec("strength", "float", 0.5, "How much to smooth.",
                      minimum=0.0, maximum=1.0),
        ),
        wraps="my library or algorithm name",
        language="python",
    )

    def run(self, data):
        data = data.copy()             # never mutate the input
        # ... transform data.X / data.obs / data.obsm ...
        return self.finalize(data, step=self.spec.category.value)
```

Three rules:

1. **Copy before you mutate** — `data = data.copy()`.
2. **Validate declaratively** — express bounds/choices in `ParamSpec`; the base class
   enforces them before `run` is called, so `self.params["strength"]` is always valid.
3. **Finish with `finalize`** — it appends the provenance entry that makes results
   reproducible.

## A real plugin: spatial neighbourhood smoothing

```python
import numpy as np
from scipy.spatial import cKDTree
from histoweave.plugins import Method, MethodCategory, MethodSpec, ParamSpec, register

@register
class SpatialSmooth(Method):
    """Denoise expression by mixing each spot with its spatial neighbours."""

    spec = MethodSpec(
        name="spatial_smooth",
        category=MethodCategory.NORMALIZATION,
        version="0.1.0",
        summary="k-NN spatial-neighbourhood expression smoothing.",
        params=(
            ParamSpec("k", "int", 6, "Spatial neighbours per spot.", minimum=1),
            ParamSpec("alpha", "float", 0.5,
                      "Self weight; 1.0 keeps the spot unchanged.",
                      minimum=0.0, maximum=1.0),
        ),
        assumptions=("obsm['spatial'] present.",),
        wraps="scipy.spatial.cKDTree",
        language="python",
    )

    def run(self, data):
        data = data.copy()
        coords = data.spatial
        if coords is None:
            raise ValueError("obsm['spatial'] is required for spatial smoothing")

        k = int(min(self.params["k"], data.n_obs - 1))
        alpha = float(self.params["alpha"])

        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1)          # +1: first neighbour is self
        neighbour_mean = np.asarray(data.X)[idx[:, 1:]].mean(axis=1)

        data.layers["pre_smooth"] = np.asarray(data.X).copy()
        data.X = alpha * np.asarray(data.X) + (1.0 - alpha) * neighbour_mean
        data.uns["spatial_smooth"] = {"k": k, "alpha": alpha}
        return self.finalize(data, step="normalization")
```

Use it exactly like a built-in method:

```python
import histoweave as ts
from histoweave.plugins import create_method

data = ts.datasets.make_synthetic(n_cells=400, n_genes=40, n_domains=3, seed=0)
smoothed = create_method("normalization", "spatial_smooth", k=8, alpha=0.6).run(data)
print(smoothed.provenance[-1])          # {'step': 'normalization', 'method': 'spatial_smooth', ...}
```

## Working through AnnData instead

If your library speaks scanpy/scvi-tools/squidpy, override `run_on_anndata` and delegate
from `run` — the base class bridges `SpatialTable ↔ AnnData` and re-attaches spatial
layers automatically:

```python
class ScanpyHVG(Method):
    spec = MethodSpec(name="scanpy_hvg", category=MethodCategory.NORMALIZATION,
                      version="0.1.0", summary="scanpy highly-variable genes",
                      language="python")

    def run(self, data):
        return self._run_via_anndata(data)

    def run_on_anndata(self, adata):
        import scanpy as sc
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        return adata
```

## Shipping it as an installable package

Methods do not have to live in HistoWeave. Copy the [`plugin-template/`](https://github.com/histoweave-spatial/histoweave/tree/main/plugin-template)
directory, which is a minimal `pyproject.toml` package that advertises an entry point:

```toml
# pyproject.toml
[project.entry-points."histoweave.plugins"]
myplugin = "histoweave_myplugin:register"
```

Once `pip install`-ed alongside HistoWeave, the entry point is discovered automatically
and your method appears in `list_methods()` and the CLI — no fork, no PR required. This
is how HistoWeave stays a *distribution layer* rather than a monolith.

## Benchmark your plugin against the built-ins

```python
from histoweave.benchmark import domain_detection_task, run_benchmark

# Register a domain-detection plugin, then benchmark every registered method at once.
bench = run_benchmark(domain_detection_task(data))
for row in bench.leaderboard:
    print(row["rank"], row["method"], row["score"])
```

Your method is ranked next to BANKSY, spectral clustering, GMM, and the rest on the same
task and metric — the whole point of the platform.

## Runnable script

[`examples/tutorial_custom_plugin.py`](https://github.com/histoweave-spatial/histoweave/blob/main/examples/tutorial_custom_plugin.py)
defines and exercises the `spatial_smooth` plugin end-to-end:

```bash
python examples/tutorial_custom_plugin.py
```
