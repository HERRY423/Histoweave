# Method failure fingerprint atlas

Failure-boundary mapping answers *where* a method breaks (the parameter value at
which score drops below τ).  **Failure fingerprints** answer *how* it breaks.

## Four failure modes

Classification uses the contingency table between planted truth and predicted
labels (label-permutation invariant — only co-occurrence structure matters):

| Mode | Definition | Severity cue |
|------|------------|--------------|
| **Fragmentation** | One true domain is split into ≥5 predicted fragments | max over true domains of #fragments |
| **Merge** | One predicted cluster absorbs ≥3 true domains | max over predicted clusters of #true domains |
| **Noise** | Predicted clusters each covering &lt;5% of observations | micro-cluster count + mass |
| **Structural** | Recovers on easy data but collapses on high noise / many domains | ARI(easy) ≥ τ and ARI(hard) near 0 |

Each method receives a fingerprint vector

```
[fragmentation, merge, noise, structural] ∈ [0, 1]⁴
```

describing how performance degrades when approaching its failure boundary.
The **dominant mode** is the largest component.

## CLI

```bash
# Standalone fingerprint probe (easy / hard-noise / multi-domain)
histoweave failure-fingerprint \
  --methods kmeans,spectral,gaussian_mixture \
  --seeds 3 \
  --out-dir failure_fingerprints

# Included by default in boundary studies
histoweave benchmark-boundary \
  --task domain_detection \
  --methods kmeans,spectral \
  --out boundary_out
# → also writes failure_fingerprints.json / .md
# Skip with --no-fingerprints
```

## Python API

```python
import logging

from histoweave.benchmark import (
    classify_domain_failure,
    run_failure_fingerprint_probe,
    write_fingerprint_atlas,
)

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# From any truth / prediction pair
profile = classify_domain_failure(truth_labels, pred_labels)
_LOGGER.info(
    "frag=%s merge=%s noise=%s",
    profile.fragmentation_flag,
    profile.merge_flag,
    profile.noise_flag,
)

# Full atlas
atlas = run_failure_fingerprint_probe(methods=["kmeans", "spectral"], seeds=(0, 1, 2))
_LOGGER.info("%s", atlas.summary())
write_fingerprint_atlas(atlas, "failure_fingerprints")
_LOGGER.info("%s", atlas.matrix())  # method → [frag, merge, noise, structural]
```

## Probe design

| Condition | Role |
|-----------|------|
| `easy` | Low noise, few domains — baseline recovery |
| `hard_noise` | High multiplicative noise |
| `hard_domains` | Many spatial domains |

Fragmentation / merge / noise are averaged over hard conditions.  Structural
severity compares mean ARI(easy) vs mean ARI(hard).

## Reading a fingerprint

| Pattern | Interpretation |
|---------|----------------|
| High fragmentation, low merge | Over-clustering; lower `n_domains` or stronger spatial smoothing |
| High merge, low fragmentation | Under-clustering; raise `k` or reduce spatial weight |
| High noise | Unstable micro-clusters; QC / min-cluster-size post-processing |
| High structural | Works on clean data only; avoid on noisy multi-domain tissues |
