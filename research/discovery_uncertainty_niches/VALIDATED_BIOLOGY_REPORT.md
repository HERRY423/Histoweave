# Validated biology report (RNA simulation dry-run)

> ⚠️ **SIMULATED FROM RNA — NOT PROTEIN IF.** This dry-run only proves the analysis path. Do not cite as validated biology.

**Status tag:** `SIMULATED_PROXY`

## Gate results

| Niche | Class | Protein/Proxy pass | Detail |
|-------|-------|:------------------:|--------|
| `151508_L3` | L3_program | **FAIL** | `{'niche_id': '151508_L3', 'class': 'L3_program', 'pass': False, 'enc1_up': False, 'hopx_up': False, 'mbp_not_up': True, 'level': 3}` |
| `151508_L6` | L6_myelin | **PASS** | `{'niche_id': '151508_L6', 'class': 'L6_myelin', 'pass': True, 'mbp_up_vs_rest': True, 'level': 3}` |
| `151669_L3` | L3_program | **FAIL** | `{'niche_id': '151669_L3', 'class': 'L3_program', 'pass': False, 'enc1_up': False, 'hopx_up': False, 'mbp_not_up': True, 'level': 3}` |

## Narrative upgrade

Simulation complete. Replace with real IF CSVs in `results/if_return/` and re-run **without** `--simulate-from-rna`.

## Contrasts (padj)

| Niche | Gene | Contrast | Δ | padj |
|-------|------|----------|--:|-----:|
| 151508_L3 | ENC1 | vs_rest | 0.3207 | 4.60e-02 |
| 151508_L3 | ENC1 | vs_same_layer | -0.0517 | 2.15e-01 |
| 151508_L3 | HOPX | vs_rest | 0.2816 | 4.18e-03 |
| 151508_L3 | HOPX | vs_same_layer | 0.0190 | 5.40e-01 |
| 151508_L3 | MBP | vs_rest | -0.4418 | 4.59e-05 |
| 151508_L3 | MBP | vs_same_layer | -0.1385 | 2.77e-01 |
| 151508_L6 | ENC1 | vs_rest | -0.0215 | 8.54e-01 |
| 151508_L6 | ENC1 | vs_same_layer | 0.3288 | 7.40e-03 |
| 151508_L6 | HOPX | vs_rest | -0.2837 | 5.23e-03 |
| 151508_L6 | HOPX | vs_same_layer | 0.0912 | 4.67e-01 |
| 151508_L6 | MBP | vs_rest | 0.5938 | 9.70e-13 |
| 151508_L6 | MBP | vs_same_layer | -0.0856 | 8.54e-01 |
| 151669_L3 | ENC1 | vs_rest | 0.2604 | 1.14e-03 |
| 151669_L3 | ENC1 | vs_same_layer | 0.1476 | 5.16e-02 |
| 151669_L3 | HOPX | vs_rest | 0.2300 | 7.32e-03 |
| 151669_L3 | HOPX | vs_same_layer | 0.0558 | 6.81e-01 |
| 151669_L3 | MBP | vs_rest | -0.2349 | 1.21e-03 |
| 151669_L3 | MBP | vs_same_layer | -0.0380 | 6.81e-01 |

## Methods note

Pre-registered in `IF_PROTOCOL.md` / `prepare_if_lab_package.py`. Mann–Whitney U + BH-FDR within the tested contrast set.
