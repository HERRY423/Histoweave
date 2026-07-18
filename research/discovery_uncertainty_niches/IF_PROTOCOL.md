# IF / spatial protein validation protocol (hand-off)

## Objective

Protein-level test of the two cryptic-niche classes discovered computationally:

| ROI set | File | Hypothesis |
|---------|------|------------|
| L3 niche (151508) | `results/panel_validation/ROI_151508_L3_n138.csv` | ENC1/HOPX high, MBP low vs other L3 |
| L3 niche (151669) | `results/panel_validation/ROI_151669_L3_n137.csv` | same direction (cross-donor) |
| L3 niche (151673) | `results/panel_validation/ROI_151673_L3_n47.csv` | **third donor** same L3 direction |
| L6 niche (151508) | `results/panel_validation/ROI_151508_L6_n154.csv` | MBP high vs whole grey; optional vs other L6 |
| L6 niche (151673) | `results/panel_validation/ROI_151673_L6_n24.csv` | small; direction-only RNA support |

**Do not pool L3 and L6 ROIs into one “cryptic state”.**

## Minimal antibody panel

| Target | Role | Notes |
|--------|------|-------|
| **ENC1** | L3 program | Primary mid-layer / neuronal-associated |
| **HOPX** | L3 program | Cross-donor direction hit |
| **MBP** | Myelin | High in L6 class; low in L3 class |
| PLP1 (optional) | Myelin backup | |
| DAPI + NeuN or MAP2 (optional) | Cellular context | |

## Image registration

1. Retrieve Space Ranger `spatial/` for sections **151508** and **151669**
   (tissue image + `scalefactors_json.json` + `tissue_positions*`).
2. ROI CSVs use the same pixel coordinates as HistoWeave `obsm['spatial']`
   (`x` = pxl_col-like, `y` = pxl_row-like — verify against `tissue_positions`).
3. Transform Visium pixel → IF pyramid with the same affine used for H&E–Visium overlay.

## Quantification (pre-registered)

For each barcode centroid:

1. Disk radius = 0.55 × Visium spot diameter from scalefactors (~55 µm default if diameter ≈ 100 µm center-to-center).
2. Mean background-subtracted IF intensity per channel.
3. Contrasts (two-sided Mann–Whitney + BH within panel):
   - **L3 ROI vs non-ROI spots with manual Layer 3** (same-layer; primary).
   - **L3 ROI vs all non-ROI** (secondary).
   - **L6 ROI vs all non-ROI** (primary for myelin).
   - **L6 ROI vs non-ROI Layer 6** (secondary; may be non-significant).

### Pass criteria (protein)

| Class | Pass if |
|-------|---------|
| L3 | ENC1 **or** HOPX higher in ROI vs same-layer L3 (padj≤0.05); MBP not higher in ROI |
| L6 | MBP higher in ROI vs rest (padj≤0.05) |
| Cross-donor | Both L3 sections meet L3 protein criteria |

## Computational status already achieved (Visium RNA)

See `PANEL_VALIDATION_REPORT.md`:

- L6 myelin composite vs rest: **PASS** (shift p=0.005)
- L3 program vs rest, both donors: **direction OK** (shift p≤0.05)
- L3 program vs **same-layer** L3: direction positive but **shift p > 0.05** (hard gate FAIL)

→ RNA supports exporting ROIs for IF; protein is required before naming a cell state.

## Deliverables back to repo

Please return:

1. Per-spot IF intensity table (`barcode`, `ENC1`, `HOPX`, `MBP`, …)
2. Registration QC (overlay PNG)
3. Short stats notebook or CSV of contrast tests

Place under `research/discovery_uncertainty_niches/results/if_return/`.
