"""Benchmarking & evaluation layer."""

from __future__ import annotations

from .active_calibration import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationPlan,
    EvidenceTask,
    attach_calibration,
    propose_evidence_acquisition,
)
from .causal import (
    CausalEffect,
    CausalLandscapeResult,
    causal_graph_svg,
    run_causal_landscape,
)
from .complexity import ComplexityFit, fit_complexity
from .digital_twin import (
    VALIDATION_SCHEMA_VERSION,
    DigitalTwinValidationResult,
    run_digital_twin_validation,
    write_digital_twin_artifacts,
)
from .donor_bootstrap import (
    DLPFC_SECTION_TO_DONOR,
    DonorBootstrapResult,
    donor_for_slice,
    donor_stratified_bootstrap_l3,
)
from .failure_boundary import (
    DEFAULT_TAU,
    Boundary,
    BoundaryStudyResult,
    SweepAxis,
    SweepPoint,
    build_axes,
    detect_boundary,
    make_svg_task_fixed,
    probe_runnable,
    run_boundary_study,
    run_sweep,
    write_cards_md,
    write_study_outputs,
)
from .failure_fingerprint import (
    FINGERPRINT_ORDER,
    FINGERPRINT_SCHEMA_VERSION,
    FailureFingerprint,
    FailureFingerprintAtlas,
    FailureModeProfile,
    classify_domain_failure,
    run_failure_fingerprint_probe,
    structural_severity,
    write_fingerprint_atlas,
)
from .features import (
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_dataframe,
    feature_vector,
)
from .figure3 import (
    FIGURE3_DATASETS,
    FIGURE3_METHODS,
    Figure3Result,
    run_figure3_experiment,
)
from .harness import (
    BenchmarkResult,
    Task,
    deconvolution_task,
    domain_detection_task,
    get_task,
    run_benchmark,
    svg_task,
)
from .isus import (
    ISUS_HIGH,
    ISUS_LOW,
    ISUSResult,
    compute_isus,
    compute_isus_from_table,
    isus_band,
    mi_discrete_continuous,
)
from .k_selection import (
    DualTrackKReport,
    KSelectionResult,
    compare_k_policies,
    estimate_n_domains,
    make_domain_k_factory,
    oracle_n_domains,
)
from .landscape import (
    LandscapeResult,
    MultiLandscapeResult,
    landscape_svg,
    run_landscape,
    run_multi_landscape,
    run_task_landscape,
)
from .landscape_io import (
    attach_dataset_meta,
    attach_features_from_tables,
    build_dlpfc_merged_landscape,
    landscape_from_long_csv,
    merge_landscapes,
    meta_from_registry,
    validate_landscape_contracts,
    write_landscape_json,
)
from .multiple_testing import fdr_adjust, pairwise_fdr_table, reject_nulls
from .pareto import (
    OBJECTIVE_DIRECTIONS,
    OBJECTIVE_LABELS,
    ObjectiveTable,
    ParetoDatasetResult,
    ParetoReport,
    analyze_dataset,
    build_report,
    knee_point,
    nondominated_sort,
    pareto_frontier,
    pareto_svg,
)
from .pareto_io import (
    bootstrap_ci_width,
    load_memory_gb,
    objective_tables_from_landscape,
    objective_tables_from_long_csv,
)
from .phenomenology_contracts import (
    EvaluationRole,
    FrozenMethodManifest,
    MethodEvaluationContract,
    ResourceClass,
    build_evaluation_contracts,
    capability_matrix_rows,
    freeze_release_manifest,
)
from .phenomenology_runner import (
    BenchmarkExecutionConfig,
    ParameterTrack,
    PhenomenologyRunSpec,
    RunOutcome,
    RunStatus,
    execute_run,
    write_long_tables,
)
from .phenomenology_statistics import (
    capability_index,
    coverage_summary,
    paired_bootstrap_ci,
    paired_method_comparisons,
)
from .phenomenology_suite import (
    DEFAULT_EVALUATION_SEEDS,
    PhenomenologySuitePlan,
    build_suite_plan,
    execute_suite,
    write_suite_plan,
)
from .recommend import MethodRecommender, MethodScore, Recommendation
from .scaling import (
    DEFAULT_COMPUTE_METHODS,
    ScalingConfig,
    ScalingRecord,
    ScalingResult,
    run_scaling,
    write_scaling_artifacts,
)
from .sota_pipeline import (
    env_contract,
    probe_all,
    probe_backend,
    run_sota_benchmark,
    write_sota_artifacts,
)
from .stats_review import (
    BootstrapARIResult,
    MethodRankSummary,
    StatsReviewReport,
    bootstrap_ari,
    bootstrap_rank_stability,
    ranks_from_scores,
    review_landscape,
)
from .task_contract import (
    AnalysisTask,
    DatasetBenchmarkRecord,
    GroundTruthKind,
    TaskContract,
    assert_labels_usable,
    classify_platform,
    default_spatial_context_policy,
)
from .uncertainty import (
    BoundaryUncertaintyResult,
    boundary_mask_from_labels,
    boundary_uncertainty,
    uncertainty_enrichment,
)

__all__ = [
    # harness
    "Task",
    "BenchmarkResult",
    "run_benchmark",
    "domain_detection_task",
    "deconvolution_task",
    "svg_task",
    "get_task",
    # Figure 3 experiment
    "FIGURE3_DATASETS",
    "FIGURE3_METHODS",
    "Figure3Result",
    "run_figure3_experiment",
    # landscape
    "LandscapeResult",
    "MultiLandscapeResult",
    "run_landscape",
    "run_task_landscape",
    "run_multi_landscape",
    "landscape_svg",
    # causal
    "CausalEffect",
    "CausalLandscapeResult",
    "run_causal_landscape",
    "causal_graph_svg",
    # failure boundary mapping
    "DEFAULT_TAU",
    "Boundary",
    "BoundaryStudyResult",
    "SweepAxis",
    "SweepPoint",
    "build_axes",
    "detect_boundary",
    "make_svg_task_fixed",
    "probe_runnable",
    "run_boundary_study",
    "run_sweep",
    "write_cards_md",
    "write_study_outputs",
    # failure fingerprint atlas
    "FINGERPRINT_ORDER",
    "FINGERPRINT_SCHEMA_VERSION",
    "FailureFingerprint",
    "FailureFingerprintAtlas",
    "FailureModeProfile",
    "classify_domain_failure",
    "run_failure_fingerprint_probe",
    "structural_severity",
    "write_fingerprint_atlas",
    # active recommender calibration
    "CALIBRATION_SCHEMA_VERSION",
    "CalibrationPlan",
    "EvidenceTask",
    "attach_calibration",
    "propose_evidence_acquisition",
    # landscape IO / SOTA merge
    "attach_dataset_meta",
    "attach_features_from_tables",
    "build_dlpfc_merged_landscape",
    "landscape_from_long_csv",
    "merge_landscapes",
    "meta_from_registry",
    "validate_landscape_contracts",
    "write_landscape_json",
    # SOTA reproduction
    "env_contract",
    "probe_all",
    "probe_backend",
    "run_sota_benchmark",
    "write_sota_artifacts",
    # recommend
    "MethodRecommender",
    "MethodScore",
    "Recommendation",
    # digital twin validation
    "VALIDATION_SCHEMA_VERSION",
    "DigitalTwinValidationResult",
    "run_digital_twin_validation",
    "write_digital_twin_artifacts",
    # task contracts
    "AnalysisTask",
    "GroundTruthKind",
    "TaskContract",
    "DatasetBenchmarkRecord",
    "assert_labels_usable",
    "classify_platform",
    "default_spatial_context_policy",
    # features
    "RECOMMENDATION_FEATURE_ORDER",
    "extract_features",
    "feature_vector",
    "feature_dataframe",
    # boundary uncertainty
    "BoundaryUncertaintyResult",
    "boundary_uncertainty",
    "boundary_mask_from_labels",
    "uncertainty_enrichment",
    # phenomenology
    "EvaluationRole",
    "FrozenMethodManifest",
    "MethodEvaluationContract",
    "ResourceClass",
    "build_evaluation_contracts",
    "capability_matrix_rows",
    "freeze_release_manifest",
    "BenchmarkExecutionConfig",
    "ParameterTrack",
    "PhenomenologyRunSpec",
    "RunOutcome",
    "RunStatus",
    "execute_run",
    "write_long_tables",
    "capability_index",
    "coverage_summary",
    "paired_bootstrap_ci",
    "paired_method_comparisons",
    "DEFAULT_EVALUATION_SEEDS",
    "PhenomenologySuitePlan",
    "build_suite_plan",
    "execute_suite",
    "write_suite_plan",
    # scaling
    "ComplexityFit",
    "fit_complexity",
    "DEFAULT_COMPUTE_METHODS",
    "ScalingConfig",
    "ScalingRecord",
    "ScalingResult",
    "run_scaling",
    "write_scaling_artifacts",
    # statistical review layer
    "BootstrapARIResult",
    "MethodRankSummary",
    "StatsReviewReport",
    "bootstrap_ari",
    "bootstrap_rank_stability",
    "review_landscape",
    "ranks_from_scores",
    "fdr_adjust",
    "pairwise_fdr_table",
    "reject_nulls",
    # non-oracle K selection
    "DualTrackKReport",
    "KSelectionResult",
    "compare_k_policies",
    "estimate_n_domains",
    "make_domain_k_factory",
    "oracle_n_domains",
    # donor-stratified discovery CIs
    "DLPFC_SECTION_TO_DONOR",
    "DonorBootstrapResult",
    "donor_for_slice",
    "donor_stratified_bootstrap_l3",
    # multi-objective Pareto frontier
    "OBJECTIVE_DIRECTIONS",
    "OBJECTIVE_LABELS",
    "ObjectiveTable",
    "ParetoDatasetResult",
    "ParetoReport",
    "analyze_dataset",
    "build_report",
    "knee_point",
    "nondominated_sort",
    "pareto_frontier",
    "pareto_svg",
    "bootstrap_ci_width",
    "load_memory_gb",
    "objective_tables_from_landscape",
    "objective_tables_from_long_csv",
    # information-theoretic spatial utility score
    "ISUS_HIGH",
    "ISUS_LOW",
    "ISUSResult",
    "compute_isus",
    "compute_isus_from_table",
    "isus_band",
    "mi_discrete_continuous",
]
