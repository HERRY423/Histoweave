# Spatial Pipeline Compiler

The compiler translates a natural-language spatial transcriptomics question into the
same `PipelineStep` objects used by HistoWeave's normal executor. It is a scheduler over
the live plugin registry—not a code-generating agent—so a model cannot invent Python,
shell commands, or unregistered analysis methods.

## Safe first run

```bash
histoweave ask "Which genes are spatially variable?" \
  --in sample.ttab --model mock --plan-only --json
```

Compilation is a dry run by default. Omit `--plan-only` to see the confirmation prompt,
or use `--yes` in a reproducible script. The offline `mock` provider supplies seven
registry-backed scenarios without an API key: `tumor`, `brain`, `developmental`,
`immune`, `drug`, `cross_section`, and a conservative `generic` fallback. These same
examples are injected as few-shot messages for online model providers, so mock and LLM
plans follow one executable contract.

## Model providers

Install the optional dependencies and pass any LiteLLM model identifier:

```bash
pip install "histoweave-spatial[compiler]"
histoweave ask "Map tumour–immune communication" \
  --in xenium.ttab --model openai/gpt-4o --plan-only
```

`HISTOWEAVE_COMPILER_MODEL` sets the default model. Provider credentials use LiteLLM's
normal environment variables. Model output must be one strict JSON object and is checked
twice: first against the compiler schema, then against the live method and parameter
registry. A rejected plan gets one repair attempt and never executes silently.

## Approximations and gaps

Requests such as an explicit invasive-margin ROI currently exceed the registry's
capabilities. The compiler may approximate them with spatial domains and a neighbourhood
graph, but it must list that degradation in the plan, emit a runtime warning, and append
the missing capability to [the gap log](COMPILER_GAPS.md).

For Nextflow, `--executor nextflow` writes a validated `*.params.json` hand-off and the
command to run. It deliberately does not spawn Nextflow from inside the compiler.
