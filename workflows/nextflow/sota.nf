#!/usr/bin/env nextflow
/// HistoWeave P2 — SOTA DLPFC domain-detection grid (DSL2).
///
/// Runs one process per (method, slice, seed) so missing backends fail
/// independently and checkpoints can be resumed.
///
/// Usage:
///   nextflow run workflows/nextflow/sota.nf \
///     --repo /path/to/histoweave \
///     --methods banksy_py,spagcn \
///     --outdir results/sota

nextflow.enable.dsl = 2

params.repo     = "${projectDir}/../.."
params.outdir   = 'results/sota'
params.methods  = 'banksy_py,spagcn,graphst,stagate,bayesspace'
params.slices   = '151673,151674,151507,151669,151670'
params.seeds    = '42,1,2'
params.dry_run  = false
params.force    = false

process SOTA_PROBE {
    tag "probe"
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path repo

    output:
    path "sota_probe.json"

    script:
    """
    python "${repo}/scripts/run_sota_dlpfc.py" \
      --methods ${params.methods} \
      --dry-run \
      --out-dir "\$PWD"
    """
}

process SOTA_CELL {
    tag "${method}_${slice}_s${seed}"
    publishDir "${params.outdir}/checkpoints", mode: 'copy'
    errorStrategy 'ignore'
    maxForks 4

    input:
    path repo
    tuple val(method), val(slice), val(seed)

    output:
    path "sota_${method}__${slice}__seed${seed}.json", optional: true

    script:
    def force_flag = params.force ? '--force' : ''
    """
    python - <<'PY'
from pathlib import Path
import sys
repo = Path("${repo}")
sys.path.insert(0, str(repo / "src"))
from histoweave.benchmark.sota_pipeline import run_sota_cell
cell = run_sota_cell(
    "${method}",
    "${slice}",
    int("${seed}"),
    repo_root=repo,
    checkpoint_dir=Path("."),
    force=${params.force ? 'True' : 'False'},
)
print(cell.status, cell.ari, cell.seconds)
PY
    """
}

process SOTA_AGGREGATE {
    tag "aggregate"
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path repo
    path checkpoints

    output:
    path "sota_benchmark_long.csv"
    path "sota_throughput.json"

    script:
    """
    python - <<'PY'
from pathlib import Path
import sys, json, csv
repo = Path("${repo}")
sys.path.insert(0, str(repo / "src"))
from histoweave.benchmark.sota_pipeline import SotaCellResult, SotaBenchmarkReport, write_sota_artifacts, probe_all

cells = []
for path in Path(".").glob("sota_*.json"):
    if path.name in {"sota_probe.json", "sota_throughput.json"}:
        continue
    payload = json.loads(path.read_text())
    cells.append(SotaCellResult(
        dataset=str(payload.get("dataset", "")),
        method=str(payload.get("method", "")),
        seed=int(payload.get("seed", 0)),
        ari=payload.get("ari"),
        seconds=float(payload.get("seconds") or 0.0),
        status=str(payload.get("status", "failed")),
        error=payload.get("error") or None,
        n_domains_truth=payload.get("n_domains_truth"),
        n_obs=payload.get("n_obs"),
    ))
report = SotaBenchmarkReport(probes=probe_all(), cells=cells)
write_sota_artifacts(report, Path("."))
print("cells", len(cells))
PY
    """
}

workflow {
    main:
    ch_repo = Channel.value(file(params.repo, checkIfExists: true))
    SOTA_PROBE(ch_repo)

    methods = params.methods.tokenize(',')
    slices  = params.slices.tokenize(',')
    seeds   = params.seeds.tokenize(',').collect { it as Integer }

    if (params.dry_run) {
        // Probe-only path already wrote skipped grid via SOTA_PROBE
    } else {
        ch_cells = Channel.fromList(
            [methods, slices, seeds].combinations()
        )
        SOTA_CELL(ch_repo, ch_cells)
        SOTA_AGGREGATE(ch_repo, SOTA_CELL.out.collect().ifEmpty([]))
    }
}
