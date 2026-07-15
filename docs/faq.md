# FAQ

## General

**What is HistoWeave, in one sentence?**
An orchestration and evaluation layer on top of scverse and Bioconductor that wraps many
spatial-transcriptomics methods behind one plugin interface, benchmarks them, and
recommends the best one for your data — it is *not* another method.

**How is it different from squidpy / Seurat / Giotto?**
Those are analysis toolkits (they *provide* methods). HistoWeave *distributes, orchestrates,
and evaluates* methods — including ones from those very toolkits — behind a stable API with
provenance, containerized pipelines, and a data-driven recommender. It is complementary,
not competing.

**Which platforms are supported?**
Ingestion for Visium, Xenium, CosMx, MERSCOPE, MERFISH, Slide-seq, and Stereo-seq. See the
[Xenium/MERFISH tutorial](tutorials/04_xenium_merfish.md) for imaging-based data.

**Do I need a GPU?**
No for the core, classical methods (QC, normalization, sklearn-family domain detection,
BANKSY). Deep-learning methods (`scanvi`, `cell2location`, autoencoders) run on CPU but are
much faster on GPU.

## Installation & extras

**`pip install histoweave-spatial` gave me a tiny install — where are the methods?**
The core is deliberately light (NumPy + pandas + Jinja2). Method backends live in extras:

```bash
pip install "histoweave-spatial[scanpy]"       # sklearn domain detection
pip install "histoweave-spatial[scanvi]"       # scANVI annotation
pip install "histoweave-spatial[io,spatial]"   # real vendor data ingestion
pip install "histoweave-spatial[all]"          # everything
```

**How do I run R-backed methods (BANKSY, nnSVG)?**
They require the `histoweave-r` container image (ghcr.io). If you have no Docker, use the
native Python fallbacks where available (e.g. `banksy_py` instead of `banksy`).

## Data & formats

**What is a `.ttab` bundle?**
HistoWeave's portable analysis bundle — a directory holding the count matrix, obs/var
tables, spatial coordinates, and provenance. Create one with `histoweave ingest`.

**Can I bring my own AnnData / SpatialData?**
Yes. `SpatialTable` bridges to AnnData ↔ SpatialData; the data model wraps a
`spatialdata.SpatialData` object internally.

**My data has no ground-truth labels. Can I still use it?**
Yes — that is the point of the recommender. It extracts *target-free* features and ranks
methods without needing labels. You only need labels to *build* a benchmark knowledge base.

## Methods & results

**Which domain-detection method should I use?**
See the [decision tree](method-selection.md). Short version: turn on the spatial term for
whatever method you use (it is the biggest accuracy lever); `banksy`/`banksy_py` is the most
robust unsupervised default; `scanvi` wins if you have even a few labels.

**Why do I get a different answer than a colleague on the same tissue?**
Mostly the spatial-context setting and method choice. In the HistoWeave variance experiment,
the spatial-context parameter explained 41% of ARI variance and method choice 24%;
preprocessing and subsampling together <4%. Always report method **and** its spatial
parameter.

**What does the ARI number mean?**
Adjusted Rand Index: agreement between predicted domains and reference labels, corrected for
chance. 1.0 = perfect, 0 = random. On real DLPFC, unsupervised methods typically land in
0.2–0.35; supervision pushes higher.

**Is `banksy_py` the same as Bioconductor BANKSY?**
No — it is a faithful native re-implementation of the BANKSY feature construction (own +
neighbourhood-mean + azimuthal-gradient features, λ weighting) so BANKSY-style analysis runs
without Docker/R. Use the R `banksy` wrapper for the canonical implementation.

## Reports & Vitessce

**The interactive viewer shows "see static plots below" — what happened?**
The Vitessce viewer failed to load (offline CDN, blocked scripts, or an old browser). The
report always ships static SVG fallbacks, so your results are still there. See
[Troubleshooting](troubleshooting.md#vitessce-viewer-does-not-load).

**Can I explore clusters and marker genes interactively?**
Yes. As of the linked-views enhancement, selecting a cluster in the scatterplot or obsSets
panel highlights those cells everywhere and updates the heatmap to that cluster's top marker
genes.

**Are the reports self-contained?**
Yes — data is inlined and Vitessce loads from CDN. One HTML file, no server, works offline
for the static plots.

## See also
- [Method selection](method-selection.md) · [Troubleshooting](troubleshooting.md)
- [Quickstart](quickstart.md) · [Concepts](concepts.md)
