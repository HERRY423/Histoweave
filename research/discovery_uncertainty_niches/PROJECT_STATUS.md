# Discovery project status (biological story freeze)

```json
{
  "protocol": "histoweave.biological_story.v1",
  "story": "BIOLOGICAL_STORY.md",
  "compose": "python research/discovery_uncertainty_niches/compose_biological_story.py",
  "headline_discoveries": {
    "D1_primary": {
      "id": "D1_L6_myelin_intralayer_niche",
      "claim": "Intra-Layer-6 myelin-concentrated cryptic niche (151508 n=154; myelin shift p=0.005; hard_pass on 2 L6 components)",
      "claim_level": "2b_if_ready",
      "protein_if": "PENDING — ROI_151508_L6_n154 MBP"
    },
    "D3_orthogonal": {
      "id": "D3_xenium_LN_Ca_signaling_cryptic_niche",
      "claim": "Intra-LN cryptic niche with KCNN4/ORAI3/MAP2K5/MEF2A same-domain hard DE on official Xenium counts",
      "claim_level": 2,
      "replication": "single section"
    },
    "D2_support": {
      "id": "D2_L3_directional_cryptic_program",
      "claim": "Cross-donor L3 direction (14/15 comps; CI excludes 0); same-layer hard FAIL",
      "claim_level": 1
    }
  },
  "cohort": {
    "n_slices": 12,
    "n_L3_components": 15,
    "n_L3_direction_ok": 14,
    "n_L6_hard_pass": 2
  },
  "donor_bootstrap_l3": {
    "n_donors": 3,
    "l3_delta_rest_point": 0.2878,
    "l3_delta_rest_ci95": [0.2221, 0.3442],
    "myelin_delta_rest_point": -0.3544,
    "myelin_delta_rest_ci95": [-0.3778, -0.3237],
    "ci_excludes_zero_both_directions": true
  },
  "functional_validation": {
    "protocol": "histoweave.functional_validation.v1",
    "doc": "FUNCTIONAL_VALIDATION.md",
    "levels": {
      "D1_L6_myelin": "F2_dual_axis",
      "D2_L3_plasticity": "F2_dual_axis",
      "D3_LN_ca2": "F2_dual_axis"
    },
    "disease_axes": {
      "D1": "myelin maintenance / demyelination vulnerability + mito trade-off",
      "D2": "mid-layer plasticity stress; myelin deplete vs rest",
      "D3": "Ca2+/MAPK activation tone in LN parenchyma"
    },
    "organisation_redefinitions": {
      "D1": "Layer 6 is multi-compartment (myelin micro-domain inside manual L6)",
      "D2": "Layer 3 is multi-compartment (plasticity niche ≠ glial boundary ribbon)",
      "D3": "LN parenchyma hosts non-GC Ca2+ micro-niches missed by pathology polygons"
    },
    "F3_next": "orthogonal platform + lineage + protein IF (FUNCTIONAL_EXPERIMENTS.md)",
    "F4_next": "CRISPR / drug / demyelination perturbation (pre-registered registry)"
  },
  "functional_experiments": {
    "protocol": "histoweave.functional_experiments.v1",
    "doc": "FUNCTIONAL_EXPERIMENTS.md",
    "n_experiments": 12,
    "classes": ["perturbation", "lineage", "orthogonal"],
    "prepare": "python research/discovery_uncertainty_niches/prepare_functional_experiment_package.py",
    "analyze": "python research/discovery_uncertainty_niches/analyze_functional_return.py",
    "returns_dir": "results/functional_experiments/returns/",
    "claim_status": "results/functional_experiments/CLAIM_STATUS.json"
  },
  "wet_lab_next": [
    "IF MBP on ROI_151508_L6_n154 (primary protein gate for D1)",
    "IF ENC1/HOPX/MBP on 151508+151669 L3 ROIs (D2 upgrade path)",
    "CODEX/IF KCNN4+ORAI3 vs BCL6 on Xenium LN rank3 (D3)",
    "Orthogonal MERFISH/Xenium myelin panel (D1_orthogonal_merfish_xenium_myelin)",
    "OPC lineage reporter density in L6 ROI (D1_lineage_opc_reporter)",
    "CRISPRi MYRF/OLIG2 or cuprizone demyelination arm (D1 F4)",
    "Return CSVs to results/if_return/ and results/functional_experiments/returns/"
  ],
  "forbidden": [
    "Cite simulated IF as protein validation",
    "Name a unified L3+L6 cryptic cell state",
    "Claim new cell type without Level 3 protein pass",
    "Claim patient-level disease mechanism without F4 cohort data",
    "Register F2 computational maps as atlas cell types without F3"
  ]
}
```

## Documents

| Doc | Role |
|-----|------|
| **[BIOLOGICAL_STORY.md](BIOLOGICAL_STORY.md)** | Full narrative: D1 + D3 headline discoveries |
| [CLAIM_LADDER.md](CLAIM_LADDER.md) | Claim levels |
| [IF_PROTOCOL.md](IF_PROTOCOL.md) | Wet-lab hand-off |
| [PANEL_VALIDATION_REPORT.md](PANEL_VALIDATION_REPORT.md) | Panel gates |
| [COHORT_META_REPORT.md](COHORT_META_REPORT.md) | 12-slice meta |
| [../discovery_xenium_lymph/LYMPH_DISCOVERY_REPORT.md](../discovery_xenium_lymph/LYMPH_DISCOVERY_REPORT.md) | Xenium track |

## Recompose

```bash
python research/discovery_uncertainty_niches/compose_biological_story.py
```
