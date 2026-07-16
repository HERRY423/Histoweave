# Spatial Pipeline Compiler

The compiler translates a natural-language spatial transcriptomics question into the same
`PipelineStep` objects used by HistoWeave's normal executor. It schedules methods from the
live plugin registry; it does not generate Python or shell code. Compiler v1 emits a sealed,
strict-JSON plan whose method releases, resolved parameters, catalog provenance, and content
identity can be reviewed before anything runs.

## Safe first run

```bash
histoweave ask "Which genes are spatially variable?" \
  --in sample.ttab --model mock \
  --plan-only --plan-out spatial-genes.plan.json \
  --timeout 60 --max-repair-attempts 1 --json
```

`--plan-only` compiles and validates without execution. `--plan-out` atomically persists the
same sealed v1 plan printed by `--json`. The offline `mock` provider requires no API key and
provides seven registry-backed scenarios: `tumor`, `brain`, `developmental`, `immune`,
`drug`, `cross_section`, and a conservative `generic` fallback. The examples are also used
as few-shot messages for online providers, so mock and LLM plans share one contract.

## The v1 plan contract

A compiled artifact contains the following audit fields in addition to its question,
rationale, steps, capability gaps, executor, and model:

| Field | Contract |
|---|---|
| `schema_version` | Exactly `1`; unsupported versions are rejected on load. |
| `plan_id` | `hwc1_` plus 24 hexadecimal SHA-256 characters derived from canonical plan content. |
| `catalog_digest` | SHA-256 digest of the exact ordered registry catalog shown to the model. |
| `catalog_assay` | The actual input-assay filter used to build that catalog, or `null` for the full catalog. |
| `attempt_count` | The successful request number, including the initial request. |
| `steps[].method_version` | Exact registered wrapper release selected during compilation. |
| `steps[].params` | Full materialized parameters, including registry defaults resolved before sealing. |

The plan fingerprint covers the question, rationale, fully materialized steps, gaps, assay
assumption, executor, dry-run state, model, attempt count, and catalog provenance. Identical
content compiled against the same catalog therefore receives the same ID. This is content
identity, not a promise that an online model will produce identical content on separate
requests.

`assay_assumed` and `catalog_assay` intentionally mean different things. The former records
the model's analytical assumption; the latter records the real registry filter derived from
the input bundle. Keeping both prevents a model assumption from creating a false catalog-
drift failure later.

## Validation and repair

Compiler output passes three gates before it can be sealed:

1. The provider must return one JSON object. Online requests use JSON-object response mode
   and temperature zero; empty envelopes and non-finite JSON numbers are rejected.
2. The schema rejects unknown fields and non-JSON values. It limits a question to 4,000
   characters, a plan to 32 steps and 16 gaps, each parameter object to 32 KiB and depth 8,
   and arrays/objects to 256 entries. Numeric values must be finite.
3. Registry validation rejects invented or non-executable categories, backward stage order,
   duplicate steps, unavailable method releases, unknown parameters, and invalid values.
   HistoWeave then fixes the exact `method_version`, resolves defaults, and validates again.

`--max-repair-attempts N` allows `N` schema/registry repair responses after the initial
response (`0 <= N <= 3`; default `1`). Provider or credential failures surface directly.
`--timeout` sets the model-request timeout in seconds (`1` to `600`). The corresponding
environment variables are `HISTOWEAVE_COMPILER_MODEL` and
`HISTOWEAVE_COMPILER_TIMEOUT`.

## Save, load, and verify

The SDK exposes the same artifact workflow as the CLI:

```python
from histoweave.compiler import compile, load_plan, run_compiled, save_plan
from histoweave.io import read_bundle

spatial = read_bundle("sample.ttab")
plan = compile(
    "Which genes are spatially variable?",
    data=spatial,
    provider="mock",
    timeout=60,
    max_repair_attempts=1,
)
save_plan(plan, "spatial-genes.plan.json")

reviewed = load_plan("spatial-genes.plan.json", require_catalog_match=True)
result = run_compiled(
    reviewed,
    data=spatial,
    out="spatial-genes-report.html",
    confirmed=True,
)
```

`save_plan` first verifies the ID, writes finite JSON to a temporary file, and atomically
replaces the destination. `load_plan` always performs strict schema parsing, fingerprint
verification, and live-registry validation of every pinned method and parameter.

`require_catalog_match=True` additionally requires the current assay-filtered catalog digest
to equal the saved digest. The default is `False` because adding an unrelated method changes
the full catalog without necessarily invalidating the referenced steps; exact referenced
methods are still validated in either mode. A hand-edited or partially corrupted artifact
fails its `plan_id` check and should be recompiled rather than executed.

## Model providers

Install the optional provider dependency and pass any supported LiteLLM model identifier:

```bash
pip install "histoweave-spatial[compiler]"
histoweave ask "Map tumour-immune communication" \
  --in xenium.ttab --model openai/YOUR_MODEL \
  --timeout 90 --max-repair-attempts 2 \
  --plan-only --plan-out tumour-immune.plan.json
```

Provider credentials use LiteLLM's normal environment variables. HistoWeave wraps provider
failures as compiler errors and does not silently substitute the mock provider.

## Confirmation, executors, and capability gaps

Compilation never grants execution authority. In the CLI, omitting `--plan-only` displays a
confirmation prompt; non-interactive sessions remain inert unless `--yes` is present.
`--yes` passes a non-persistent confirmation to the executor without mutating the sealed
plan. In the SDK, `run_compiled` likewise defaults to refusal and requires the explicit
`confirmed=True` argument. Before writing any output, it revalidates the registry contract
and verifies that the plan is sealed and untampered.

Requests such as an explicit invasive-margin ROI currently exceed the registry's
capabilities. The compiler may approximate them with spatial domains and a neighbourhood
graph, but it must list that degradation in the plan, emit a runtime warning, and append the
missing capability to [the gap log](COMPILER_GAPS.md).

For Nextflow, `--executor nextflow` preflights supported stages, then writes a validated
`*.params.json` hand-off and input bundle. It returns the command to run and deliberately
does not spawn Nextflow from inside the compiler.

## CLI options at a glance

| Option | Effect |
|---|---|
| `--plan-only` | Compile and validate; never execute. |
| `--plan-out PATH` | Atomically save the sealed v1 JSON plan. |
| `--timeout SECONDS` | Set each model request timeout (`1`-`600`). |
| `--max-repair-attempts N` | Allow `0`-`3` repair responses after the first response. |
| `--executor in-process|nextflow` | Execute in process or emit a Nextflow hand-off. |
| `--yes` | Supply explicit non-interactive execution confirmation. |
| `--json` | Emit the compiled plan as JSON; use with `--plan-only` for one plan document. |
| `--gaps-file PATH` | Select the Markdown capability-gap audit log. |
