# Benchmark capability comparison and reviewer response

This document positions HistoWeave against adjacent benchmarking and spatial-analysis projects without treating unlike systems as direct substitutes. The comparison is limited to capabilities documented in the cited papers, official documentation, and the current HistoWeave implementation.

## Reading the table

- **● Native**: the capability is an explicit, executable part of the framework.
- **? Adjacent**: a related capability exists, but not with the same inferential contract.
- **○ Not documented**: no dedicated implementation was found in the cited official source.
- **N/A**: the capability is outside the project's stated evaluation target.

“Not documented” is narrower than “impossible”: the table asks what each project itself makes explicit, reproducible, and difficult to misuse.

> **Name-resolution note.** We could not identify a unique public spatial/single-cell benchmarking project named **IBERO**. The column below provisionally interprets it as **IBRAP** (Integrated Benchmarking scRNA-seq Analytical Pipeline), the closest verifiable match. Replace this column if a different DOI or repository was intended.

## Functional comparison

| Functional dimension | **HistoWeave** | **SpatialBench** | **IBERO / IBRAP*** | **scIB** | **Squidpy** | **Giotto Suite** | **Open Problems** |
|---|---|---|---|---|---|---|---|
| Primary object evaluated | Spatial methods, configurations, and reproducible workflows | AI-agent answers from real spatial-data snapshots | scRNA-seq pipeline combinations | Single-cell integration methods | Spatial analysis functions and data structures | End-to-end spatial multi-omics ecosystem | Community-defined single-cell tasks and submitted methods |
| Native spatial scope | **●** Multiple technologies/tasks | **●** Five technologies, seven agent task categories | **○** scRNA-seq | **○** scRNA/scATAC integration | **●** | **●** | **?** Mainly single-cell |
| Unified executable method interface | **●** Typed plugins with version, maturity, backend, parameters, provenance | **?** Agent harness, not method-plugin contract | **●** Interchangeable stages | **●** Integration wrappers/metrics | **?** Modular API, not benchmark contract | **?** Modular ecosystem, not submission contract | **●** Task/component submission standards |
| Hard task and ground-truth contract | **●** Spatial-domain, cell-type, and self-supervised labels are typed; invalid combinations fail closed | **?** Deterministic graders, no method-level ground-truth type system | **?** Labelled tasks, no spatial guard | **?** Batch/label keys, no spatial task typing | **○** | **○** | **●/?** Strong task schemas, no spatial ground-truth-kind guard |
| Non-oracle K | **●** Estimate is default; silhouette, BIC-GMM, or gap without labels | **N/A** | **?** No oracle-versus-estimated K contract | **?** No non-oracle spatial K policy | **○** | **○** | **N/A/○** Task dependent |
| Oracle-K leakage guard | **●** Oracle needs explicit flag and documented ablation notes | **○** | **○** | **○** | **○** | **○** | **○** |
| Causal performance landscape | **●** Interventional DGP, ACE, bootstrap CI, feature displacement | **○** | **○** | **○** | **○** | **○** | **○** |
| Adversarial failure boundary | **●** Direction-aware sweep, threshold crossing, uncertainty, Safe Operating Cards | **?** Agent fragility, not method boundaries | **?** Combination comparison, no boundary estimator | **?** Task/scaling comparisons, no boundary estimator | **○** | **○** | **?** Broad evaluation, no standard boundary crossing |
| Failure states separated from scientific failure | **●** N/A, backend unavailable, timeout, OOM, invalid input, method error, scored failure are distinct | **?** Execution/format/answer failures | **?** No frozen cross-role taxonomy found | **?** Operational failures, no capability-matrix taxonomy | **○** | **○** | **●/?** Task-specific execution/result states |
| Role-conditioned capability matrix | **●** Inference, preprocessing, integration, ingestion are not globally ranked together | **○** | **?** Pipeline stages separated | **●/?** Bio-conservation and batch correction are separate | **○** | **○** | **●/?** Task-specific leaderboards |
| Donor-stratified bootstrap | **●** Donors resampled first; within-donor structure preserved | **○** | **○** | **○** No dedicated donor-stratified CI in core benchmark | **○** | **○** | **○** Not a general primitive |
| Rank uncertainty | **●** Bootstrap rank CI and probability of being best | **?** Repeated agent runs, not method-rank bootstrap | **○** Point matrices | **?** Aggregation, no general rank posterior | **○** | **○** | **?** Living leaderboard, no universal rank bootstrap |
| Method-level FDR | **●** Paired across-dataset permutation tests; BH, BY, Holm, Bonferroni | **○** | **○** | **○** Metric aggregation is not rank FDR | **?** Analysis-level adjusted p-values only | **?** Analysis-level correction only | **○/?** Task-specific, no framework-wide rank FDR |
| Dataset-specific recommendation | **●** Target-free features, priors, LOOCV, global-best baseline, regret/warnings | **○** | **●/?** Selects pipeline combinations | **○** Scores rather than deployable recommendation | **○** | **○** | **?** Leaderboards guide selection, not query-dataset recommendation |
| Synthetic stress tests plus real validation | **●** Explicit DGP; synthetic and real claims separated | **○/?** Primarily real snapshots | **●/?** Real and simulated data | **●** Simulated and real integration tasks | **○** Analysis framework | **○** Analysis framework | **●/?** Task dependent |
| External real-data evidence | **●** Study-grouped and cross-platform landscapes | **●** Real spatial snapshots | **●** Case studies | **●** Thirteen published tasks | **●** Multi-technology demos | **●** Multiscale/multimodal demos | **●** Community datasets |
| Resource/scalability evidence | **●** Runtime, memory, timeout, scale contracts, resource-aware execution | **●/?** Agent cost/harness effects | **?** Designed for large studies | **●** Atlas-scale evaluation | **●** Sparse/Dask | **●** Scalable multiscale framework | **●/?** Task-specific |
| Continuous public leaderboard/submissions | **?** Local/static leaderboard and CI; hosted service is future work | **●** | **○/?** Local benchmark/R Shiny | **?** Reproducible website/pipeline | **○** | **○** | **●** Core strength |

*Do not cite the provisional IBRAP interpretation as “IBERO” until the intended source is confirmed.*

## Defensible positioning

The table does not support a claim that HistoWeave is broader than every ecosystem. Squidpy and Giotto Suite are richer interactive analysis toolboxes; Open Problems has a stronger community benchmark ecosystem; scIB is deeper for single-cell integration; and SpatialBench evaluates AI agents.

The narrower claim is:

> HistoWeave combines spatial-method execution with inferential guardrails usually external to analysis toolboxes and leaderboards: typed task/ground-truth contracts, non-oracle K by default, explicit oracle ablations, causal and adversarial stress tests, donor-aware uncertainty, rank stability, and method-level FDR control.

This must be paired with the limitation that community adoption, external dataset breadth, and a hosted leaderboard remain behind mature ecosystems.

## Common reviewer objections and responses

| Likely objection | Recommended response | Evidence |
|---|---|---|
| **“Another wrapper.”** | HistoWeave does not claim a new clustering algorithm. The contribution is the evaluation contract: identical inputs, typed tasks, backend states, non-oracle defaults, uncertainty, and reproducible comparison. | plugins/interfaces.py, task_contract.py, provenance/manifests |
| **“Unrelated tasks are mixed.”** | No universal rank spans ingestion, preprocessing, integration, and inference. Summaries are role/category conditioned; incompatible ground truth fails validation. | phenomenology_contracts.py, TaskContract.validate |
| **“True K was leaked.”** | Estimated K is the default. Oracle K is an opt-in ablation requiring a flag and notes; both tracks are reported separately. | k_selection.py, statistical-review.md |
| **“Synthetic benchmarks are circular.”** | Synthetic data are restricted to known-DGP questions: interventions and boundaries. Transportability is assessed separately on real data. | causal.py, failure_boundary.py, external validation |
| **“Point ranks do not prove superiority.”** | Report rank intervals, probability of best, paired across-dataset tests, and adjusted q-values; no multi-method significance from one dataset. | stats_review.py, multiple_testing.py |
| **“Spots are pseudoreplicates.”** | Resample donors as the top-level unit; label spot bootstrap as conditional precision, not cohort generalization. | donor_bootstrap.py |
| **“Recommender overfits.”** | Use leave-one-dataset-out prediction against a global-best baseline; preserve negative regret and warnings. | recommend.py, LOOCV outputs |
| **“Failure threshold is arbitrary.”** | Make tau explicit and preregistered; report interpolation/uncertainty and add threshold sensitivity before publication. | failure_boundary.py, Safe Operating Cards |
| **“Missing dependencies count as bad science.”** | Backend unavailable, timeout, OOM, invalid input, method error, N/A, and scientific failure have separate states and denominators. | phenomenology_runner.py, coverage summaries |
| **“Why not Squidpy/Giotto?”** | Use them for exploration. HistoWeave is complementary, adding method-neutral execution, benchmarking, recommendation, and guardrails. | Plugin interface and AnnData bridge |
| **“Why not scIB/Open Problems?”** | scIB is the integration specialist and Open Problems the living-benchmark specialist. HistoWeave adds spatial semantics, K leakage controls, boundaries, and donor-aware review. | Comparison table |
| **“Cross-platform labels differ.”** | Platform never defines task equivalence. Records carry task, ground-truth kind, label key, platform, and spatial policy; invalid/proxy comparisons fail or warn. | task_contract.py, landscape_io.py |
| **“No independent adoption.”** | Concede it. Present technical/empirical validation, not ecosystem dominance; prioritize external submissions, donors, tissues, and hosting. | Limitations and roadmap |

## Paste-ready reviewer response

> We thank the reviewer for asking how HistoWeave differs from existing spatial-analysis and benchmarking ecosystems. We have revised the manuscript to avoid treating these projects as interchangeable. SpatialBench evaluates AI agents on verifiable spatial-analysis problems; scIB specializes in single-cell integration; Squidpy and Giotto Suite are rich analysis environments; and Open Problems provides mature living-benchmark infrastructure. HistoWeave is complementary. Its narrower contribution is an executable spatial-method evaluation contract combining typed task and ground-truth semantics, non-oracle domain-count selection by default, explicit oracle-K ablations, controlled causal and adversarial stress tests, donor-stratified uncertainty, bootstrap rank stability, and FDR-controlled across-dataset method comparisons.
>
> We do not claim that one method is universally optimal, that synthetic experiments establish biological validity, or that the repository has the community scale of Open Problems, Squidpy, or Giotto Suite. Synthetic experiments are restricted to identifiable DGP and boundary estimands; external real-data analyses are reported separately; missing backends and resource failures are not counted as scientific failures; and recommender performance is compared with a global-best baseline under leave-one-dataset-out evaluation.

## Primary external sources

1. **SpatialBench:** Workman et al., arXiv:2512.21907 (2025), <https://arxiv.org/abs/2512.21907>.
2. **IBRAP (provisional “IBERO” interpretation):** <https://pmc.ncbi.nlm.nih.gov/articles/PMC10025434/>.
3. **scIB:** Luecken et al., Nature Methods 19, 41–50 (2022), <https://www.nature.com/articles/s41592-021-01336-8>; <https://scib.readthedocs.io/en/latest/>.
4. **Squidpy:** Palla et al., Nature Methods 19, 171–178 (2022), <https://www.nature.com/articles/s41592-021-01358-2>.
5. **Giotto Suite:** <https://pmc.ncbi.nlm.nih.gov/articles/PMC10705291/>; <https://giottosuite.com/>.
6. **Open Problems:** <https://pmc.ncbi.nlm.nih.gov/articles/PMC11030530/>; <https://github.com/openproblems-bio>.

## Before submission

1. Confirm what “IBERO” refers to and replace the provisional IBRAP column if needed.
2. Add failure-boundary sensitivity instead of reporting only tau=0.7.
3. Report estimated-K and oracle-K panels with identical data and seeds.
4. State independent donor counts for every cohort confidence interval.
5. Publish raw outcomes and adjusted pairwise q-values behind every rank claim.
6. Retain the limitation that hosted submissions and community scale remain behind mature ecosystems.
