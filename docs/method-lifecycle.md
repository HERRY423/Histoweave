# Method versions and migration

HistoWeave versions the wrapper contract independently from the upstream scientific
package. A registry identity is the tuple `category:name@version`. This allows an
incompatible wrapper upgrade to coexist with the old release during a migration
window instead of changing existing workflows in place.

## Selecting a release

The SDK selects the newest non-deprecated release when no version is supplied:

```python
from histoweave.plugins import create_method

method = create_method("deconvolution", "cell2location")
pinned = create_method(
    "deconvolution",
    "cell2location",
    version="0.1.0",
)
```

Declarative pipelines use `PipelineStep.method_version`; the single-step CLI uses
`--method-version`. Every run manifest and provenance entry records the resolved
wrapper version.

```bash
histoweave step deconvolution --method cell2location --method-version 0.1.0 \
  --in normalized.ttab --out deconvolved.ttab
```

Use `histoweave list-methods --all-versions` to inspect active and deprecated
releases. JSON output includes the replacement target, removal release, parameter
renames, removed parameters, and migration notes.

## Deprecating an incompatible release

Keep the old class registered and attach a `MethodDeprecation` to its `MethodSpec`.
The replacement is an exact `MethodReference`; approximate or floating replacements
are not accepted.

```python
from histoweave.plugins import MethodDeprecation, MethodReference, MethodSpec

spec = MethodSpec(
    name="example",
    category="qc",
    version="1.0.0",
    deprecation=MethodDeprecation(
        since="0.2.0",
        remove_in="0.4.0",
        replacement=MethodReference("qc", "example", "2.0.0"),
        parameter_renames=(("old_cutoff", "threshold"),),
        removed_parameters=("legacy_mode",),
        notes="Review the new upstream normalization default.",
    ),
)
```

Selecting an explicitly deprecated release raises `MethodDeprecationWarning`, a
visible `FutureWarning` subclass. Default resolution skips deprecated releases.

## Migrating parameters

`migrate_method_params` follows replacement chains, applies renames one release at a
time, validates the final parameter schema, and returns an auditable migration path.
It never silently discards a removed parameter.

```python
from histoweave.plugins import migrate_method_params

migration = migrate_method_params(
    "qc", "example", "1.0.0", {"old_cutoff": 5}
)
# migration["params"] == {"threshold": 5}
# migration["version"] == "2.0.0"
```

If a removed parameter is present, migration stops with a `ValueError` containing
the release notes. Conflicting old and new parameter names also stop migration.

## Real external methods

External adapters declare `implementation="external"`, the exact upstream method in
`wraps`, and one or more `BackendRequirement` records. Missing optional backends
raise an installation diagnostic. Scientific adapters must not fall back to a native
heuristic or toy substitute because that would make provenance scientifically false.
