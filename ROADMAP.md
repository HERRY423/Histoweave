# HistoWeave Roadmap

Four phases over ~30 months, each with a clear objective, headline deliverables, and an
**exit gate** that must be met before proceeding. Discipline at these gates is the main
defense against scope creep.

| Phase | Window | Headline outcome |
|------:|--------|------------------|
| 0 · Foundations | Months 0–3 | Bets de-risked; project & governance stood up. |
| 1 · MVP core | Months 3–9 | Reproducible end-to-end pipeline, 3 assays, alpha. |
| 2 · Breadth + benchmarking | Months 9–18 | 20+ methods, live leaderboards, beta, first paper. |
| 3 · Scale + AI + multi-omics | Months 18–30 | Billion-cell scale, foundation-model hub, v1.0. |
| 4 · Ecosystem | Months 30+ | Governance, marketplace, sustainable funding. |

---

## Phase 0 — Foundations & validation · months 0–3
**Objective:** de-risk the core technical bets and establish the project.

- [ ] User research: ~10 target-user interviews to validate the five bottlenecks and prioritize modules.
- [ ] Technical spikes: SpatialData ingestion for Visium HD + Xenium; a minimal Nextflow QC→clustering pipeline; an out-of-core load test on a large public dataset.
- [ ] Governance & licensing decided; GitHub org, code of conduct, public RFC for the plugin API v0.
- [ ] Select 3 canonical public datasets and 2 driving biological use cases; recruit a small advisory group.

**Exit gate:** spikes prove SpatialData + Nextflow + out-of-core is viable; RFC published; advisory group and first datasets secured.

> **This scaffold covers the Phase-0 "walking skeleton":** the layered architecture, the
> plugin API v0, an in-process pipeline, the benchmarking harness pattern, and reporting
> — all exercised end-to-end on synthetic data. The remaining Phase-0 items are research
> and infrastructure decisions, tracked as issues.

## Phase 1 — MVP core · months 3–9
**Objective:** one reproducible end-to-end pipeline across three assay families.

- [ ] Ingestion for Visium/Visium HD, Xenium, and one sequencing-based assay (e.g. Stereo-seq) → canonical SpatialData. *(readers stubbed in `io/`)*
- [ ] End-to-end Nextflow pipeline: ingest → QC → normalization → spatial domain detection → annotation → automated HTML report. *(in-process runner done; Nextflow port in `workflows/nextflow/`)*
- [ ] Python SDK + CLI; plugin API v1 with ~5 wrapped methods; containerized, tested, documented.
- [ ] Alpha release on PyPI/conda with two executable tutorials.

**Exit gate:** an external user reproduces the full pipeline on their own data from the tutorial, unaided.

## Phase 2 — Method breadth & benchmarking · months 9–18
**Objective:** become genuinely useful and trustworthy — the benchmarking differentiator goes live.

- [ ] Plugin ecosystem across all core stages (target 20+ methods), including containerized R/Bioconductor methods.
- [ ] Benchmarking harness live with 3–4 tasks (domain detection, deconvolution, cell–cell communication), metrics, and public leaderboards; in-workflow method recommendations.
- [ ] Visualization integration (Vitessce + napari-spatialdata); richer reporting.
- [ ] Beta release; first preprint (platform + benchmarking); first community workshop / hackathon.

**Exit gate:** external contributors add ≥3 plugins; leaderboards cover the top methods per task; ≥1 partner lab adopts HistoWeave for a real project.

## Phase 3 — Scale, AI & multi-omics · months 18–30
**Objective:** reach the frontier the field is moving toward.

- [ ] Validated out-of-core / cloud execution on billion-cell-scale corpora.
- [ ] Foundation-model hub: apply / fine-tune / benchmark Nicheformer, Novae, scGPT-spatial as plugins.
- [ ] Spatial multi-omics (same-slide RNA+protein, spatial ATAC) and 3D reconstruction; histology-to-expression inference.
- [ ] Atlas interoperability with HuBMAP / HTAN / CELLxGENE; v1.0 release + application paper.

**Exit gate:** a billion-cell workflow runs end-to-end; ≥1 foundation model benchmarked against classical baselines in-platform; v1.0 shipped.

## Phase 4 — Ecosystem & sustainability · months 30+
**Objective:** durability beyond the founding team.

- [ ] Open governance foundation (steering council; consider a NumFOCUS-style fiscal host).
- [ ] Plugin "marketplace," certification, and teaching materials; optional clinical/reproducibility profiles.
- [ ] Diversified, multi-year funding secured; documented maintainer succession.

**Exit gate:** ≥3 independent institutions contributing; sustainable funding runway; healthy bus factor.
