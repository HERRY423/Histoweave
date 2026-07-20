# Independent study/donor personalisation

- **Protocol**: `histoweave.independent_personalisation.v1`
- **Independent queries**: 17 (target ≥15: met)
- **Non-inferiority margin**: 0.02
- **Mean global-best regret**: 0.016199984076815492
- **Mean unconstrained k-NN regret**: 0.016816073623454542 (non-inferior=True)
- **Mean gated personalisation regret**: 0.020195692236994165 (point-NI=True, stat-NI=True, superior=False)
- **Gated personalisation rate**: 0.6470588235294118
- **Primary policy**: gated_personalisation (primary-NI=True, superior=False)

Primary claim uses gated personalisation (fallback to global-best when the local proxy does not clear the gate). Statistical non-inferiority requires the bootstrap upper CI of mean(gated−global) ≤ margin. Unconstrained k-NN is a diagnostic, not the deployment policy.

| Unit | Class | kNN reg | Global reg | Gated reg | Action |
|------|-------|---------|------------|-----------|--------|
| allen_merfish_brain_section | external_study | 0.0000 | 0.0000 | 0.0000 | personalised |
| dlpfc_donor_Br5292 | biological_donor | 0.0195 | 0.0195 | 0.0195 | personalised |
| dlpfc_donor_Br5595 | biological_donor | 0.0000 | 0.0000 | 0.0000 | personalised |
| dlpfc_donor_Br8100 | biological_donor | 0.0677 | 0.0677 | 0.0677 | personalised |
| platform_merfish | cross_platform_study | 0.0000 | 0.0000 | 0.0000 | global_default |
| platform_slideseqv2 | cross_platform_study | 0.0140 | 0.0140 | 0.0140 | global_default |
| platform_xenium | cross_platform_study | 0.0031 | 0.0319 | 0.0319 | global_default |
| slideseq_puck_200115_08 | external_study | 0.0622 | 0.0000 | 0.0622 | personalised |
| squidpy_four_i | external_study | 0.0011 | 0.0211 | 0.0211 | global_default |
| squidpy_imc | external_study | 0.0677 | 0.0677 | 0.0677 | personalised |
| squidpy_mibitof | external_study | 0.0449 | 0.0449 | 0.0449 | global_default |
| squidpy_seqfish | external_study | 0.0000 | 0.0086 | 0.0086 | global_default |
| visium_hd_crc | external_study | 0.0000 | 0.0000 | 0.0000 | personalised |
| visium_mouse_brain | external_study | 0.0000 | 0.0000 | 0.0000 | personalised |
| xenium_human_lymph_node | external_study | 0.0057 | 0.0000 | 0.0057 | personalised |
| xenium_lung_cancer | external_study | 0.0000 | 0.0000 | 0.0000 | personalised |
| xenium_ovarian_cancer | external_study | 0.0000 | 0.0000 | 0.0000 | personalised |

## Cross-lab reproducibility

- Gated−global mean Δ regret: **0.003995708160178672** (95% CI [0.0, 0.011651862206240995]; P(Δ≤0)=0.1145)
- Kendall's W (rank concordance): **0.21425040604477086** across 17 units
- Independence class counts: {'external_study': 11, 'biological_donor': 3, 'cross_platform_study': 3}

## By independence class

- **biological_donor** (n=3): gated=0.0290, global=0.0290, personalised_rate=100%
- **cross_platform_study** (n=3): gated=0.0153, global=0.0153, personalised_rate=0%
- **external_study** (n=11): gated=0.0191, global=0.0129, personalised_rate=73%
