# API Reference

The API reference is generated from HistoWeave's source and docstrings with
[mkdocstrings](https://mkdocstrings.github.io/).  The modules below are the
supported entry points for data containers, workflows, plugins, ingestion,
benchmarking, and reporting.

## Top-level API

::: histoweave
    options:
      members: true
      show_root_heading: true

## Data model ? `histoweave.data`

::: histoweave.data.model
    options:
      members: true
      show_root_heading: true

## Datasets ? `histoweave.datasets`

::: histoweave.datasets
    options:
      members: true
      show_root_heading: true

## Workflow engine ? `histoweave.workflow`

::: histoweave.workflow
    options:
      members: true
      show_root_heading: true

## Plugin system ? `histoweave.plugins`

::: histoweave.plugins.interfaces
    options:
      members: true
      show_root_heading: true

::: histoweave.plugins.registry
    options:
      members: true
      show_root_heading: true

## I/O and portable bundles ? `histoweave.io`

### Base reader

::: histoweave.io.base

### Per-assay readers

::: histoweave.io.readers
    options:
      members: true

### Bundle persistence

::: histoweave.io.bundle
    options:
      members: true

## Benchmarking ? `histoweave.benchmark`

::: histoweave.benchmark
    options:
      members: true
      show_root_heading: true

## Spatial AutoML ? `histoweave.automl`

::: histoweave.automl
    options:
      members: true
      show_root_heading: true

## Reporting ? `histoweave.report`

::: histoweave.report
    options:
      members: true
      show_root_heading: true
