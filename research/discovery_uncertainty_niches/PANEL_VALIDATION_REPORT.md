# Panel validation & IF ROI export

**Pre-registered panels**

- L3 program: `ENC1, HOPX, GAP43, GRIA2, CARTPT`
- Myelin: `MBP, PLP1, MOBP`
- Exploratory (not for primary claim): `GFAP, SST, SAA1, SCGB2A2`

## Gate results

| Component | Class | n | L3 Δ same-layer | L3 shift p | Myelin Δ rest | Myelin shift p | direction_ok | **pass** |
|-----------|-------|--:|----------------:|-----------:|--------------:|---------------:|:------------:|:--------:|
| 151508_L6_n154 | L6_myelin | 154 | 0.066 | 0.105 | 0.497 | 0.005 | Y | **PASS** |
| 151508_L3_n138 | L3_program | 138 | 0.064 | 0.190 | -0.339 | 0.960 | Y | FAIL |
| 151669_L3_n137 | L3_program | 137 | 0.058 | 0.370 | -0.247 | 0.905 | Y | FAIL |
| 151673_L3_n47 | L3_program | 47 | 0.038 | 0.470 | -0.609 | 0.965 | Y | FAIL |
| 151673_L6_n24 | L6_myelin | 24 | 0.025 | 0.510 | 0.033 | 0.340 | Y | FAIL |

**L3 direction OK:** `3/3` donors/components (need L3-program ↑ vs rest and myelin ↓ vs rest).

**Cross-donor L3 direction (all listed L3 niches):** `YES`.

**Cross-donor L3 hard gate (same-layer shift p≤0.05 on all):** `NO`.

**L6 myelin direction OK:** `2/2`; **L6 hard pass:** `1/2`.

## Composite scores (detail)

### 151508_L6_n154

- **L3_program** genes=['ENC1', 'HOPX', 'GAP43', 'GRIA2', 'CARTPT']: mean_in=-0.162, rest=0.006, same-layer out=-0.229; Δrest=-0.168 (shift p=0.940); Δsame-layer=0.066 (shift p=0.105)
- **myelin** genes=['MBP', 'PLP1', 'MOBP']: mean_in=0.480, rest=-0.017, same-layer out=0.457; Δrest=0.497 (shift p=0.005); Δsame-layer=0.023 (shift p=0.320)

### 151508_L3_n138

- **L3_program** genes=['ENC1', 'HOPX', 'GAP43', 'GRIA2', 'CARTPT']: mean_in=0.380, rest=-0.012, same-layer out=0.316; Δrest=0.392 (shift p=0.005); Δsame-layer=0.064 (shift p=0.190)
- **myelin** genes=['MBP', 'PLP1', 'MOBP']: mean_in=-0.328, rest=0.011, same-layer out=-0.299; Δrest=-0.339 (shift p=0.960); Δsame-layer=-0.030 (shift p=0.715)

### 151669_L3_n137

- **L3_program** genes=['ENC1', 'HOPX', 'GAP43', 'GRIA2', 'CARTPT']: mean_in=0.217, rest=-0.008, same-layer out=0.159; Δrest=0.225 (shift p=0.040); Δsame-layer=0.058 (shift p=0.370)
- **myelin** genes=['MBP', 'PLP1', 'MOBP']: mean_in=-0.238, rest=0.009, same-layer out=-0.229; Δrest=-0.247 (shift p=0.905); Δsame-layer=-0.008 (shift p=0.535)

### 151673_L3_n47

- **L3_program** genes=['ENC1', 'HOPX', 'GAP43', 'GRIA2', 'CARTPT']: mean_in=0.486, rest=-0.006, same-layer out=0.448; Δrest=0.492 (shift p=0.035); Δsame-layer=0.038 (shift p=0.470)
- **myelin** genes=['MBP', 'PLP1', 'MOBP']: mean_in=-0.602, rest=0.008, same-layer out=-0.583; Δrest=-0.609 (shift p=0.965); Δsame-layer=-0.019 (shift p=0.615)

### 151673_L6_n24

- **L3_program** genes=['ENC1', 'HOPX', 'GAP43', 'GRIA2', 'CARTPT']: mean_in=-0.148, rest=0.001, same-layer out=-0.172; Δrest=-0.148 (shift p=0.690); Δsame-layer=0.025 (shift p=0.510)
- **myelin** genes=['MBP', 'PLP1', 'MOBP']: mean_in=0.033, rest=-0.000, same-layer out=0.392; Δrest=0.033 (shift p=0.340); Δsame-layer=-0.359 (shift p=0.930)

## Per-gene primary panel (same-layer contrast)

| Component | Gene | Panel | Δ same-layer | p same-layer | shift p | Δ rest |
|-----------|------|-------|-------------:|-------------:|--------:|-------:|
| 151508_L3_n138 | `CARTPT` | L3_program | -0.009 | 9.11e-01 | 0.550 | 0.141 |
| 151508_L3_n138 | `ENC1` | L3_program | -0.052 | 1.43e-01 | 0.720 | 0.551 |
| 151508_L3_n138 | `GAP43` | L3_program | 0.152 | 4.19e-02 | 0.030 | 0.486 |
| 151508_L3_n138 | `GRIA2` | L3_program | 0.207 | 1.73e-02 | 0.010 | 0.353 |
| 151508_L3_n138 | `HOPX` | L3_program | 0.019 | 5.40e-01 | 0.500 | 0.452 |
| 151508_L3_n138 | `MBP` | myelin | -0.138 | 2.31e-01 | 0.855 | -0.602 |
| 151508_L3_n138 | `MOBP` | myelin | 0.004 | 7.78e-01 | 0.485 | -0.179 |
| 151508_L3_n138 | `PLP1` | myelin | 0.021 | 7.91e-01 | 0.500 | -0.396 |
| 151508_L6_n154 | `CARTPT` | L3_program | -0.024 | 5.09e-01 | 0.745 | -0.142 |
| 151508_L6_n154 | `ENC1` | L3_program | 0.329 | 3.70e-03 | 0.015 | -0.111 |
| 151508_L6_n154 | `GAP43` | L3_program | 0.157 | 2.49e-01 | 0.090 | 0.040 |
| 151508_L6_n154 | `GRIA2` | L3_program | -0.107 | 2.58e-01 | 0.900 | -0.157 |
| 151508_L6_n154 | `HOPX` | L3_program | 0.091 | 3.11e-01 | 0.190 | -0.367 |
| 151508_L6_n154 | `MBP` | myelin | -0.086 | 8.54e-01 | 0.760 | 0.753 |
| 151508_L6_n154 | `MOBP` | myelin | 0.064 | 4.42e-01 | 0.245 | 0.295 |
| 151508_L6_n154 | `PLP1` | myelin | 0.072 | 8.53e-01 | 0.250 | 0.659 |
| 151669_L3_n137 | `CARTPT` | L3_program | 0.114 | 2.02e-01 | 0.100 | 0.340 |
| 151669_L3_n137 | `ENC1` | L3_program | 0.148 | 3.44e-02 | 0.070 | 0.307 |
| 151669_L3_n137 | `GAP43` | L3_program | -0.087 | 1.88e-01 | 0.745 | 0.007 |
| 151669_L3_n137 | `GRIA2` | L3_program | 0.009 | 9.41e-01 | 0.435 | -0.010 |
| 151669_L3_n137 | `HOPX` | L3_program | 0.056 | 6.60e-01 | 0.290 | 0.314 |
| 151669_L3_n137 | `MBP` | myelin | -0.038 | 6.81e-01 | 0.650 | -0.308 |
| 151669_L3_n137 | `MOBP` | myelin | 0.056 | 2.87e-01 | 0.095 | -0.092 |
| 151669_L3_n137 | `PLP1` | myelin | -0.079 | 2.02e-01 | 0.840 | -0.297 |
| 151673_L3_n47 | `CARTPT` | L3_program | -0.028 | 9.65e-01 | 0.580 | 0.222 |
| 151673_L3_n47 | `ENC1` | L3_program | -0.011 | 6.95e-01 | 0.540 | 0.572 |
| 151673_L3_n47 | `GAP43` | L3_program | 0.232 | 8.36e-03 | 0.020 | 0.587 |
| 151673_L3_n47 | `GRIA2` | L3_program | 0.038 | 7.69e-01 | 0.460 | 0.224 |
| 151673_L3_n47 | `HOPX` | L3_program | -0.052 | 3.50e-01 | 0.680 | 0.459 |
| 151673_L3_n47 | `MBP` | myelin | -0.109 | 3.44e-01 | 0.870 | -0.904 |
| 151673_L3_n47 | `MOBP` | myelin | 0.021 | 8.62e-01 | 0.360 | -0.513 |
| 151673_L3_n47 | `PLP1` | myelin | 0.022 | 9.50e-01 | 0.395 | -0.770 |
| 151673_L6_n24 | `CARTPT` | L3_program | -0.056 | 5.46e-01 | 0.815 | -0.202 |
| 151673_L6_n24 | `ENC1` | L3_program | 0.071 | 6.38e-01 | 0.470 | -0.086 |
| 151673_L6_n24 | `GAP43` | L3_program | 0.015 | 7.99e-01 | 0.380 | -0.019 |
| 151673_L6_n24 | `GRIA2` | L3_program | 0.266 | 1.11e-01 | 0.105 | 0.191 |
| 151673_L6_n24 | `HOPX` | L3_program | -0.164 | 3.19e-01 | 0.835 | -0.433 |
| 151673_L6_n24 | `MBP` | myelin | -0.334 | 2.79e-02 | 0.795 | 0.259 |
| 151673_L6_n24 | `MOBP` | myelin | -0.440 | 3.10e-02 | 0.915 | -0.187 |
| 151673_L6_n24 | `PLP1` | myelin | -0.504 | 1.57e-02 | 0.890 | 0.078 |

## IF / imaging hand-off

ROI CSVs (one row per Visium spot in the component):

- `results/panel_validation/ROI_151508_L6_n154.csv`
- `results/panel_validation/ROI_151508_L3_n138.csv`
- `results/panel_validation/ROI_151669_L3_n137.csv`
- `results/panel_validation/ROI_151673_L3_n47.csv`
- `results/panel_validation/ROI_151673_L6_n24.csv`

**Recommended IF panel (minimal):** ENC1, HOPX, MBP (optional + PLP1, GAP43).

**Sectioning notes:**

1. Align Visium pixel coordinates (`x`,`y` in ROI CSV) to H&E via the original Space Ranger `spatial/` folder for that section.
2. Score IF mean intensity inside a 55 µm radius of each barcode centroid (Visium center-to-center ~100 µm; use spot diameter from scalefactors).
3. Primary stats: L3 ROIs should be ENC1/HOPX-high and MBP-low vs same-layer non-ROI L3; L6 ROI myelin-high vs whole section (same-layer optional).

## Claim bounds

1. Visium expression is a **proxy**, not IF. Hard biological claim still needs protein.
2. Same-layer shift-null is stricter than whole-slice DE; failures here block naming a new cell state.
3. SCGB/SAA-family genes remain exploratory / artifact-suspect and are excluded from primary panels.
