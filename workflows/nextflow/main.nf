#!/usr/bin/env nextflow
/// HistoWeave — Nextflow DSL2 pipeline for spatial transcriptomics.

nextflow.enable.dsl = 2

params.input    = null
params.assay    = null
params.engine   = 'native'
params.n_domains = null
params.outdir   = 'results'
// Marker annotation is opt-in: real vendor inputs do not carry demo marker metadata.
params.steps    = 'qc,normalize,domain_detection,report'
params.demo     = false
params.seed     = 0
params.publish_mode = 'copy'

params.qc_method        = 'basic_qc'
params.normalize_method = 'log1p_cp10k'
params.domain_method    = 'kmeans'
params.annotation_method = 'marker_score'
params.deconvolution_method = 'marker_deconv'

params.qc_params        = ''
params.normalize_params = ''
params.domain_params    = ''
params.annotation_params = ''

// Override --container_version (or either full image parameter) for a released build.
params.container_version = '0.0.1'
params.python_image = "ghcr.io/histoweave-spatial/histoweave-python:${params.container_version}"
params.r_image      = "ghcr.io/histoweave-spatial/histoweave-r:${params.container_version}"

workflow {
    main:
    if (params.demo && params.input) {
        error("--demo and --input are mutually exclusive.")
    }
    if (!params.demo && !params.input) {
        error("One of --input or --demo is required.")
    }

    def selected_steps = _parse_steps(params.steps)
    def engine = _validate_engine(params.engine)
    def assay = _validate_assay(params.assay, params.demo)
    def n_domains = _validate_n_domains(params.n_domains)

    if (selected_steps.contains('domain_detection') && n_domains == null) {
        error("--n_domains must be a positive integer when domain_detection is selected.")
    }
    if (!params.demo && assay == 'stereo_seq' && engine != 'spatialdata') {
        error("Stereo-seq ingestion requires --engine spatialdata.")
    }

    if (params.demo) {
        INGEST_DEMO(params.seed)
        ch_bundle = INGEST_DEMO.out.bundle
    } else {
        ch_input = Channel.value(file(params.input, checkIfExists: true))
        INGEST_VENDOR(ch_input, assay, engine)
        ch_bundle = INGEST_VENDOR.out.bundle
    }

    if (selected_steps.contains('qc')) {
        QC(ch_bundle, params.qc_method, params.qc_params, params.seed)
        ch_bundle = QC.out.bundle
    }

    if (selected_steps.contains('normalize')) {
        NORMALIZE(ch_bundle, params.normalize_method, params.normalize_params, params.seed)
        ch_bundle = NORMALIZE.out.bundle
    }

    if (selected_steps.contains('domain_detection')) {
        DOMAIN_DETECTION(
            ch_bundle,
            params.domain_method,
            params.domain_params,
            n_domains,
            params.seed,
        )
        ch_bundle = DOMAIN_DETECTION.out.bundle
    }

    if (selected_steps.contains('annotation')) {
        ANNOTATION(ch_bundle, params.annotation_method, params.annotation_params, params.seed)
        ch_bundle = ANNOTATION.out.bundle
    }

    if (selected_steps.contains('deconvolution')) {
        DECONVOLUTION(ch_bundle, params.deconvolution_method, '', params.seed)
        ch_bundle = DECONVOLUTION.out.bundle
    }

    if (selected_steps.contains('report')) {
        REPORT(ch_bundle, params.seed)
        ch_report = REPORT.out.report
    }

    onComplete:
    log.info """
    ========================================================
      HistoWeave pipeline complete.
      Results  : ${params.outdir}
      Steps    : ${params.steps}
    ========================================================
    """.stripIndent()
}

process INGEST_DEMO {
    container   params.python_image
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode, pattern: "*.ttab"
    errorStrategy 'terminate'
    label 'low_mem'

    input:
    val seed

    output:
    path "${prefix}.ttab", emit: bundle

    script:
    prefix = "ingest_demo_seed${seed}"
    """
    histoweave ingest --demo --seed ${seed} --out ${prefix}.ttab
    """
}

process INGEST_VENDOR {
    tag         "${assay}:${vendor_input.simpleName}"
    container   params.python_image
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode, pattern: "*.ttab"
    errorStrategy 'terminate'
    label 'low_mem'

    input:
    path vendor_input
    val  assay
    val  engine

    output:
    path "${prefix}.ttab", emit: bundle

    script:
    prefix = "ingest_${assay}"
    """
    histoweave ingest --input "${vendor_input}" --assay ${assay} --engine ${engine} \
        --out "${prefix}.ttab"
    """
}

process QC {
    container   params.python_image
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode
    errorStrategy 'terminate'
    label 'low_mem'

    input:
    path bundle_in
    val  method
    val  params_str
    val  seed

    output:
    path "${bundle_name}_qc.ttab", emit: bundle

    script:
    bundle_name = bundle_in.simpleName - ~/\.ttab$/
    def param_args = params_str ? _param_args(params_str) : ''
    """
    histoweave step qc --method ${method} ${param_args} --in ${bundle_in} --out ${bundle_name}_qc.ttab
    """
}

process NORMALIZE {
    container   { method in ['sctransform', 'r_lognorm'] ? params.r_image : params.python_image }
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode
    errorStrategy 'terminate'
    label 'low_mem'

    input:
    path bundle_in
    val  method
    val  params_str
    val  seed

    output:
    path "${bundle_name}_normalized.ttab", emit: bundle

    script:
    bundle_name = bundle_in.simpleName - ~/\.ttab$/
    def param_args = params_str ? _param_args(params_str) : ''
    """
    histoweave step normalization --method ${method} ${param_args} --in ${bundle_in} --out ${bundle_name}_normalized.ttab
    """
}

process DOMAIN_DETECTION {
    container   { method == 'banksy' ? params.r_image : params.python_image }
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode
    errorStrategy 'terminate'
    label 'domain_detection'

    input:
    path bundle_in
    val  method
    val  params_str
    val  n_domains
    val  seed

    output:
    path "${bundle_name}_domains.ttab", emit: bundle

    script:
    bundle_name = bundle_in.simpleName - ~/\.ttab$/
    def param_args = params_str ? _param_args(params_str) : ''
    """
    histoweave step domain_detection --method ${method} ${param_args} \
        --param n_domains=${n_domains} --in ${bundle_in} --out ${bundle_name}_domains.ttab
    """
}

process ANNOTATION {
    container   params.python_image
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode
    errorStrategy 'terminate'
    label 'low_mem'

    input:
    path bundle_in
    val  method
    val  params_str
    val  seed

    output:
    path "${bundle_name}_annotated.ttab", emit: bundle

    script:
    bundle_name = bundle_in.simpleName - ~/\.ttab$/
    def param_args = params_str ? _param_args(params_str) : ''
    """
    histoweave step annotation --method ${method} ${param_args} --in ${bundle_in} --out ${bundle_name}_annotated.ttab
    """
}

process DECONVOLUTION {
    container   params.python_image
    publishDir  "${params.outdir}/bundles", mode: params.publish_mode
    errorStrategy 'terminate'
    label 'deconvolution'

    input:
    path bundle_in
    val  method
    val  params_str
    val  seed

    output:
    path "${bundle_name}_deconv.ttab", emit: bundle

    script:
    bundle_name = bundle_in.simpleName - ~/\.ttab$/
    def param_args = params_str ? _param_args(params_str) : ''
    """
    histoweave step deconvolution --method ${method} ${param_args} --in ${bundle_in} --out ${bundle_name}_deconv.ttab
    """
}

process REPORT {
    container   params.python_image
    publishDir  "${params.outdir}", mode: params.publish_mode
    errorStrategy 'terminate'
    label 'low_mem'

    input:
    path bundle_in
    val  seed

    output:
    path "*.html", emit: report

    script:
    bundle_name = bundle_in.simpleName - ~/\.ttab$/
    """
    histoweave report --in ${bundle_in} --out histoweave_report.html
    cp histoweave_report.html ${bundle_name}_report.html
    """
}

def _param_args(String param_str) {
    if (!param_str) return ''
    return param_str.split(',').collect { "--param ${it.trim()}" }.join(' ')
}

def _parse_steps(Object raw_steps) {
    def allowed = [
        'qc',
        'normalize',
        'domain_detection',
        'annotation',
        'deconvolution',
        'report',
    ] as Set
    def tokens = raw_steps instanceof Collection
        ? raw_steps.collect { it.toString().trim() }
        : (raw_steps == null ? [] : raw_steps.toString().split(',', -1).collect { it.trim() })

    if (!tokens || tokens.any { it.isEmpty() }) {
        error("--steps must be a comma-separated list of exact step names.")
    }
    def invalid = tokens.findAll { !allowed.contains(it) }.unique()
    if (invalid) {
        error("Unknown --steps token(s): ${invalid.join(', ')}. Allowed: ${allowed.join(', ')}")
    }
    if (tokens.size() != tokens.toSet().size()) {
        error("--steps contains duplicate tokens: ${tokens.join(', ')}")
    }
    return tokens as Set
}

def _validate_assay(Object raw_assay, boolean demo) {
    def allowed = ['visium', 'xenium', 'stereo_seq'] as Set
    def assay = raw_assay == null ? null : raw_assay.toString().trim().toLowerCase()
    if (assay && !allowed.contains(assay)) {
        error("Unknown --assay '${raw_assay}'. Allowed: ${allowed.join(', ')}")
    }
    if (!demo && !assay) {
        error("--assay is required with real --input data.")
    }
    return assay
}

def _validate_engine(Object raw_engine) {
    def allowed = ['native', 'spatialdata'] as Set
    def engine = raw_engine == null ? '' : raw_engine.toString().trim().toLowerCase()
    if (!allowed.contains(engine)) {
        error("Unknown --engine '${raw_engine}'. Allowed: ${allowed.join(', ')}")
    }
    return engine
}

def _validate_n_domains(Object raw_n_domains) {
    def value = raw_n_domains == null ? '' : raw_n_domains.toString().trim()
    if (!value) return null
    if (!(value ==~ /[1-9]\d*/)) {
        error("--n_domains must be a positive integer; received '${raw_n_domains}'.")
    }
    return value.toInteger()
}
