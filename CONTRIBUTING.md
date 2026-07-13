# Contributing to HistoWeave

HistoWeave is built **with the ecosystem, not around it**. Contributions — bug reports,
method plugins, benchmark tasks, docs — are structural to the project, not incidental.

## Ground rules

- Be kind and constructive. See the [Code of Conduct](CODE_OF_CONDUCT.md).
- Prefer upstreaming fixes to scverse / Bioconductor / spatialdata-io where they belong.
- Every change ships with tests and docs. Reproducibility is the product.

## Development setup

```bash
git clone https://github.com/histoweave-spatial/histoweave
cd histoweave
python -m venv .venv && source .venv/bin/activate   # or your env manager of choice
pip install -e ".[dev]"
pytest
```

## Repository strategy

- **This repo is the lean core**: data model, workflow runner, plugin API, benchmarking
  harness, reporting.
- **Methods live in separate, independently versioned plugin repos** so the fast-moving
  frontier can move without destabilizing the core. Use [`plugin-template/`](plugin-template/)
  as a starting point.

## Adding a method (plugin)

1. Subclass `histoweave.plugins.Method` and declare a `MethodSpec` (category, version,
   parameters, assumptions, what it `wraps`).
2. Implement `run(self, data) -> SpatialTable`; treat inputs as immutable (`data.copy()`),
   write results into `data.obs` / `data.obsm`, and end with `return self.finalize(data)`
   so provenance is captured.
3. Register it: `@register` in-tree, or advertise it on the `histoweave.plugins`
   entry-point group in your plugin package's `pyproject.toml`.
4. Add a benchmark entry: a method is only "done" when it can be evaluated. Add or extend
   a `Task` in `histoweave/benchmark/` with reference data and a metric.

## Testing & benchmarking

- Unit/integration tests run on **tiny canonical datasets** so CI is fast and
  deterministic (`histoweave.datasets.make_synthetic`).
- The benchmarking harness gates releases against performance regressions.

## Style

- `ruff check .` and `ruff format .` for lint/format.
- `mypy src` for typing.
- Conventional, descriptive commit messages; small, reviewable PRs.

## Good first issues

Look for the `good first issue` label. Ingestion readers (`io/readers.py`) and additional
benchmark metrics are well-scoped starting points.

## Licensing / DCO

Contributions are accepted under the project's [BSD-3-Clause](LICENSE) license. Sign your
commits (`git commit -s`) to certify the Developer Certificate of Origin.
