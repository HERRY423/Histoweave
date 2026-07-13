# HistoWeave plugin template

A minimal, copy-paste starting point for an **external HistoWeave method plugin**. Methods
live in separate, independently versioned repos so the fast-moving frontier can move
without destabilizing the core.

## Layout

```
plugin-template/
├── pyproject.toml            # advertises the method on the `histoweave.plugins` group
└── src/
    └── histoweave_myplugin/
        ├── __init__.py       # register_all() entry point
        └── method.py         # your Method subclass(es)
```

## How registration works

Your package advertises a hook on the `histoweave.plugins` entry-point group:

```toml
[project.entry-points."histoweave.plugins"]
myplugin = "histoweave_myplugin:register_all"
```

When HistoWeave enumerates methods (e.g. `histoweave list-methods`), it loads every hook on that
group, which imports your `Method` subclasses and runs their `@register` decorators. No
change to the HistoWeave core is needed.

## Try it

```bash
cd plugin-template
pip install -e .            # also needs `pip install histoweave-spatial`
histoweave list-methods        # your method now appears in the registry
```

## Checklist for a "done" method

- [ ] `MethodSpec` declares category, version, params, assumptions, and what it `wraps`.
- [ ] `run()` treats input as immutable (`data.copy()`) and ends with `self.finalize(data)`.
- [ ] Unit test on `histoweave.datasets.make_synthetic`.
- [ ] A benchmark `Task` (or an entry in an existing one) so the method can be evaluated.
- [ ] Container recipe if it wraps R/Bioconductor or heavy native deps.
