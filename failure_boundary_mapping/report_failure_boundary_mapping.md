# Adversarial Failure-Boundary Mapping for HistoWeave Methods

**Goal.** Not "which method scores best," but *where does each method break?* For every
analysis method registered in HistoWeave, we sweep a **single** synthetic-data parameter
across its axis and locate the **critical value** at which the method's performance crosses
from **acceptable** to **unacceptable**. The deliverable is a **Safe Operating Card** per
method: the conditions under which it can be trusted, and the exact point where it fails.

## Method

- **Engine.** `failure_boundary_mapping/failure_boundary.py` sits entirely on top of the
  public benchmark API (`histoweave.benchmark.harness.run_benchmark` / `Task`) and the
  synthetic generators in `histoweave.datasets`. No registered method was modified.
- **Acceptability threshold.** **τ = 0.7 (absolute)**, applied uniformly. Score is
  ARI for domain detection, `1 − RMSD` for deconvolution, and precision@k for SVG.
  A method is *acceptable* at a parameter value if its **mean score over 5 replicate
  seeds ≥ 0.7**.
- **Failure boundary `x*`.** For each (method, axis) we order the grid from easy → hard
  (direction-aware), then find the crossing where the mean score falls through τ and
  linearly interpolate `x*`. We also report the **safe range** (parameter interval with
  mean score ≥ τ), monotonicity, and Spearman ρ so non-monotone curves are flagged rather
  than reduced to a misleading single number.
- **Corrected SVG scoring.** The SVG task shipped in the harness has a mis-specified
  precision@k (it does not truncate a method's output to top-*k*, and reads only
  `uns['svg']` while `gearys_c` / `spatial_variance_ratio` write to a method-named key).
  This produced impossible scores (`morans_i` = 2.0) and false zeros. The mapping uses a
  corrected scorer (`make_svg_task_fixed`) that truncates to top-*k* and reads whichever
  `uns` key a method populates, so SVG boundaries reflect method robustness — not plumbing.

### Sweep axes (single knob varied, others pinned to a documented baseline)

| Task | Axis (knob) | Range | Harder direction |
|---|---|---|---|
| domain_detection | `marker_gene_lift` (signal) | 5.0 → 0.5 | smaller |
| domain_detection | `noise` (σ) | 0.1 → 1.9 | larger |
| domain_detection | `n_domains` | 2 → 20 | larger |
| domain_detection | `n_cells` (sample size) | 900 → 60 | smaller |
| deconvolution | `noise` (σ) | 0.05 → 1.15 | larger |
| deconvolution | `n_cell_types` | 2 → 12 | larger |
| svg | `marker_gene_lift` (signal) | 5.0 → 0.5 | smaller |
| svg | `noise` (σ) | 0.1 → 1.9 | larger |

**Scale of the study:** 25 runnable methods × 8 axes × grid × 5 seeds = **4,225 scored
runs → 80 method×axis Safe Operating Cards**. Backend-gated methods that cannot run in this
environment are recorded explicitly, not silently dropped: `banksy` (container; needs raw
counts), `cell2location` (needs a reference), `nnsvg` (AnnData write setting), `spatialde`
(module unavailable).

## Headline findings

### 1. Signal strength (`marker_gene_lift`) is the sharpest failure axis
This axis separates the domain-detection methods most cleanly — exactly where a
benchmark-only ("best score") view would hide the difference. Boundary `x*` = the weakest
signal each method tolerates before ARI drops below 0.7 (lower = more robust):

| Method | Fails below `x*` | Best ARI |
|---|---|---|
| weave_multiscale_consensus_domains | **0.83** | 0.89 |
| banksy_py | **0.83** | 0.93 |
| weave_uncertainty_domains | 0.93 | 0.91 |
| gaussian_mixture | 1.40 | 0.93 |
| kmeans | 1.48 | 1.00 |
| weave_topology_regularized_domains | 1.73 | 0.91 |
| bisecting_kmeans | 1.85 | 0.91 |
| agglomerative / birch | 2.40 | 0.84 |
| weave_boundary_aware_domains | **3.04** | 0.86 |

Read this as a trade-off, not a ranking: `kmeans` has the highest ceiling (ARI ≈ 1.0 on
easy data) but needs signal ≥ ~1.5, whereas the spatially-aware `banksy_py` and
`weave_multiscale_consensus_domains` keep working down to signal ≈ 0.83 — they are the
methods to trust in weak-signal tissue.

### 2. Density-based clustering is categorically unsuited to blob/Voronoi domains
`dbscan`, `mean_shift`, and `optics` **never reach τ = 0.7 on any axis** (ARI ≈ 0). This is
a structural mismatch (they seek density-separated clusters; the synthetic domains are
Voronoi-tessellated and touch), and the map states it plainly rather than reporting a
boundary. `minibatch_kmeans` is also fragile (never acceptable on `marker_gene_lift`,
`n_domains`; boundary only in a narrow band otherwise).

### 3. Ability to scale to many domains is method-specific
On `n_domains` most partition methods are locked to the true count and fail as soon as the
number of domains rises above the baseline, but two methods scale far:
`kmeans` (safe to **≈ 12** domains, interpolated `x*` ≈ 12.7) and `banksy_py` (safe to
**≈ 10** domains, interpolated `x*` ≈ 11.6). If your
tissue has many domains, these are the safe choices.

### 4. Multiplicative noise alone rarely breaks methods
On the `noise` axis most domain-detection methods are **robust across the entire tested
range** (σ up to 1.9). Signal strength and domain count are far more dangerous knobs than
lognormal expression noise — a concrete, testable design insight for anyone stress-testing
these pipelines.

### 5. SVG: classical statistics are rock-solid; two experimental methods fail
With the corrected precision@k, `morans_i`, `gearys_c`, `spatial_variance_ratio`,
`weave_multiscale_svg`, `weave_hotspot_svg`, and `weave_bootstrap_robust_svg` are robust
across both signal and noise sweeps. In contrast, **`weave_anisotropy_svg` and
`weave_boundary_svg` never reach precision 0.7** even on the easiest data — a genuine
weakness the corrected scorer exposes.

### 6. Deconvolution is robust within the resolvable regime (a caveat, not a win)
Both runnable deconvolution methods (`marker_deconv`, `weave_spatial_simplex_deconv`) stay
above τ across the tested `noise` (≤ 1.15) and `n_cell_types` (≤ 12) ranges. This is partly
because marker-based deconvolution is inherently strong on **marker-separable** synthetic
mixtures; their true failure boundary lies beyond the generator's clean capacity (see
caveats). We capped `n_cell_types` at 12 because at ≥ 13 the mixture generator can only
synthesise 12 marker-distinct programs (60 genes / 5 per type), so higher values would
measure a generator limit, not method robustness.

## Figures

- `figures/curve_<task>_<param>.svg/.png` — score vs parameter (mean ± std over seeds) for
  every method, with the τ line and `x*` markers. One per axis (8 total).
- `figures/summary_boundary_heatmap.svg/.png` — cross-method **safe-operating margin**:
  the fraction of each axis's difficulty range where a method stays ≥ τ (green = robust,
  red = fails early).

## How to reproduce / extend

```bash
# Full study
python failure_boundary_mapping/run_failure_mapping.py --seeds 5 --tau 0.7 --out results
python failure_boundary_mapping/make_figures.py

# One method/axis from the CLI (added subcommand)
histoweave benchmark-boundary --task domain_detection --axis marker_gene_lift \
    --methods kmeans,banksy_py --tau 0.7 --seeds 5
```

## Caveats & limitations

1. **Synthetic-only.** Boundaries are defined on controlled synthetic generators, which is
   the point (single-knob causal control), but absolute `x*` values are specific to these
   generators' parameterization and baseline, not real-tissue units.
2. **τ = 0.7 is a policy choice.** It is exposed as a parameter; a different τ shifts every
   boundary. The engine can re-run at any threshold.
3. **Baseline dependence.** Each sweep pins non-swept knobs to one baseline; boundaries can
   move under a different baseline. Multi-parameter (2-D) failure surfaces are a natural
   next step and are out of scope here.
4. **Deconvolution range ceiling.** The two deconvolution methods did not fail within the
   generator's clean range; a harder mixture generator (more overlapping profiles, rare
   cell types, lower marker specificity) would be needed to locate their true boundary.
5. **Backend-gated methods** (`banksy`, `cell2location`, `nnsvg`, `spatialde`) were not
   evaluated in this environment and are listed as such in the cards.

*Full per-method cards: `failure_boundary_mapping/results/safe_operating_cards.md`;
machine-readable summary: `safe_operating_cards.csv` / `.json`; raw measurements:
`boundary_long.csv`.*
