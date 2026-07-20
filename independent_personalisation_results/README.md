# Independent study/donor personalisation — reference artefacts

**Protocol:** `histoweave.independent_personalisation.v1`  
**Role:** frozen **in-repo** summary evidence for study-level gated personalisation  
(not a pre-execution predictor, not wet-lab data).

These files are **tracked in git** (small JSON/MD). Large raw H5AD inputs remain
local (`/data/`, `*.h5ad` gitignored). Rebuild with:

```bash
python scripts/run_independent_personalisation.py
# optional panel expansion:
python scripts/expand_real_independent_studies.py
```

## Tracked files

| File | Role |
|------|------|
| `independent_personalisation_summary.json` | **Primary** machine-readable endpoints |
| `independent_personalisation_report.md` | Human-readable summary table |
| `cross_lab_reproducibility.json` | Gated−global Δ, Kendall *W*, CIs |
| `personalisation_policies.json` | Policy definitions / margins |
| `independent_unit_landscape.json` | Per-unit landscape used by the panel |
| `real_independent_unit_landscape.json` | Real-unit subset landscape |

## Citation rule

1. Quote the **protocol string** and the **primary policy** (`gated_personalisation`).
2. Prefer `independent_personalisation_summary.json` over prose approximations.
3. Report **non-inferiority**, not superiority, unless the summary says otherwise.

See [Reference artefacts](../docs/reference-artefacts.md).
