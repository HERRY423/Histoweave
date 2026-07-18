# Contributing to HistoWeave

Thank you for helping build the connective tissue of spatial transcriptomics.
Contributions — bug reports, method plugins, benchmark tasks, docs, and review —
are structural to the project, not incidental.

## Ground rules

1. **Be kind.** Read and follow the [Code of Conduct](CODE_OF_CONDUCT.md).
2. **Prefer upstreaming.** Fixes that belong in scverse / Bioconductor /
   spatialdata-io should go there; HistoWeave wraps, it does not fork.
3. **Ship tests and docs with every change.** Reproducibility is the product.
4. **Do not inflate maturity.** New methods start as `experimental` or `beta`
   until evidence in `release_manifest.py` justifies promotion.
5. **Do not leak oracle *K*** in default benchmark paths. Use
   `k_policy="estimate"` unless the PR is an explicit ablation.

## Development setup

```bash
git clone https://github.com/histoweave-spatial/histoweave
cd histoweave
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
mypy src
ruff check .
```

Optional method extras (only when you touch those backends):

```bash
pip install -e ".[scanpy,cell2location,spatialde,deep-learning]"
```

## Repository layout (where to put work)

| Area | Path | Notes |
|------|------|-------|
| Plugin API / registry | `src/histoweave/plugins/` | Contracts, maturity, coverage |
| Built-in methods | `src/histoweave/plugins/builtin/` | Prefer thin wrappers |
| Release maturity lists | `.../builtin/release_manifest.py` | **Required** for new methods |
| Benchmarks / stats | `src/histoweave/benchmark/` | Task contracts, stats review, K |
| Method guides | `docs/methods/` | Catalog is generated |
| Tests | `tests/` | Mirror modules; keep ratio ≥ 0.8 |

## Adding a method (checklist)

1. **Implement** a `Method` subclass with `MethodSpec` (category, version,
   params, assumptions, `wraps`, backends).
2. **Register** with `@register` (in-tree) or the `histoweave.plugins`
   entry-point group (external package).
3. **Classify** the method in `release_manifest.py`:
   - `PRODUCTION_METHODS` / `BETA_METHODS` / `VALIDATED_METHODS` /
     `RESEARCH_METHODS` / `EXPERIMENTAL_BASELINES`
   - Leave **no unclassified names** — `method_coverage_report()` gates this.
4. **Tests:** smoke run on `make_synthetic`, plus any backend skip markers.
5. **Docs:** regenerate the catalog:
   ```bash
   python scripts/generate_method_docs.py
   ```
   Add a hand-written deep guide under `docs/methods/<name>.md` only when the
   method is field-facing and needs nuanced when-to-use guidance.
6. **Benchmark (when applicable):** task contract, non-oracle *K*, and optional
   `review_landscape` stats for multi-dataset claims.

### Method maturity (do not skip)

| Tier | Meaning |
|------|---------|
| experimental | Contract-stable only; research / baseline |
| beta | Real upstream wrap + structural tests |
| production | Pinned path, real-data smoke, ops diagnostics |
| validated | Production + multi-dataset evidence in `VALIDATION_EVIDENCE` |

## Documentation

- User docs: MkDocs Material (`mkdocs serve` / `mkdocs build --strict`).
- Method inventory must cover **100%** of registered methods via
  `docs/methods/catalog.md` + `docs/methods/generated/` (CI checks this).
- Prefer short, accurate prose over marketing claims.

## Testing & quality gates

```bash
pytest -q
pytest -m property -q          # Hypothesis invariants
pytest -m perf -q              # micro-benchmark ceilings
mypy src
ruff check .
python scripts/generate_method_docs.py
```

Coverage / maturity:

```python
from histoweave.plugins import method_coverage_report
report = method_coverage_report()
assert report["counts"]["unclassified_methods"] == 0
assert report["passes_all_targets"]
```

## Pull request checklist

- [ ] Tests added/updated; suite green locally
- [ ] `release_manifest.py` updated if methods were added/renamed
- [ ] Method docs regenerated (`generate_method_docs.py`)
- [ ] No new oracle-*K* defaults without `allow_oracle_k` + notes
- [ ] CHANGELOG `[Unreleased]` entry for user-visible changes
- [ ] Signed-off commits (`git commit -s`) for DCO

## Style

- `ruff check .` and `ruff format .`
- `mypy src` (global `ignore_missing_imports = false`)
- Small, reviewable PRs; conventional commit messages welcome

## Good first issues

Labels: `good first issue`, `docs`, `plugin`. Useful entry points:

- Ingestion readers (`io/readers.py`)
- Method guide deep pages for production domain methods
- Benchmark tasks / metrics under `benchmark/`
- Research incubator dual-reporting against production baselines

## Security / conduct reports

- Security: open a private advisory on GitHub or email maintainers.
- Conduct: **conduct@histoweave-spatial.org** (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)).

## Licensing / DCO

Contributions are accepted under the project [BSD-3-Clause](LICENSE) license.
Sign commits (`git commit -s`) to certify the
[Developer Certificate of Origin](https://developercertificate.org/).
