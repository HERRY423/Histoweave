# Contributing

HistoWeave welcomes plugins, benchmarks, docs, and review. The full maintainer-facing
guide lives at the repository root:

**[CONTRIBUTING.md](https://github.com/HERRY423/Histoweave/blob/main/CONTRIBUTING.md)**

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
mypy src
ruff check .
python scripts/generate_method_docs.py   # keep method guides complete
```

## Adding a method (short form)

1. Implement `Method` + `MethodSpec`, register with `@register`.
2. **Classify** it in `src/histoweave/plugins/builtin/release_manifest.py`
   (production / beta / validated / research / baseline). Unclassified methods
   fail `method_coverage_report()`.
3. Add tests (synthetic smoke at minimum).
4. Regenerate docs: `python scripts/generate_method_docs.py`.
5. Do not introduce oracle-*K* defaults; use `k_policy="estimate"`.

## Community standards

Participation is governed by the
[Code of Conduct](code-of-conduct.md)
(Contributor Covenant 2.1). Report incidents to **conduct@histoweave-spatial.org**.

## Method documentation policy

Every registered method must appear in:

* [Method catalog](methods/catalog.md)
* A [category page](methods/categories/domain_detection.md)
* A generated page under `methods/generated/`

Hand-written deep guides (e.g. [banksy_py](methods/banksy_py.md)) are optional
extras for field-facing methods.
