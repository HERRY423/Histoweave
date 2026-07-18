# Cryptic-component对照：151508 最大 / 次大 vs 151669 最大

**Date:** 2026-07-17
**Pipeline:** `analyze_largest_component.py` (same DE + adjacency code for all three)

| Case | Slice | Rank | n spots | Out dir |
|------|-------|-----:|--------:|---------|
| A | `dlpfc_151508` | 0 (largest) | 154 | `results/dlpfc_151508/largest_component/` |
| B | `dlpfc_151508` | 1 (second) | 138 | `results/dlpfc_151508/component_rank1_n138/` |
| C | `dlpfc_151669` | 0 (largest) | 137 | `results/dlpfc_151669/largest_component/` |

---

## Side-by-side geometry

| Metric | A · 151508 #1 (154) | B · 151508 #2 (138) | C · 151669 #1 (137) |
|--------|---------------------|---------------------|---------------------|
| **domain_truth inside** | **100% Layer 6** | **100% Layer 3** | **100% Layer 3** |
| Top abutting layer | Layer 6 only | Layer 3 only | Layer 3 only |
| External contacts | 297 → L6 (100%) | 260 → L3 (100%) | 288 → L3 (100%) |
| Contact enrichment vs bg | ×11.4 (L6) | ×3.4 (L3) | ×3.9 (L3) |
| Primary abutment | L6 128 / internal 26 | L3 115 / internal 23 | L3 116 / internal 21 |
| Internal edge fraction | 67.9% | 68.6% | 65.0% |
| Geometry class | **Intra-L6 compact** | **Intra-L3 compact** | **Intra-L3 compact** |

**Shared pattern:** every large cryptic component analysed so far is a **pure single manual layer**, fully surrounded by the **same** layer — not a WM↔grey or L5↔L6 ribbon.

---

## Side-by-side markers (component vs rest of slice)

| | A · L6 niche (154) | B · L3 niche (138) | C · L3 niche (137) |
|--|--------------------|--------------------|--------------------|
| padj≤0.05 | 63 (14↑ / 49↓) | 51 (45↑ / 6↓) | 19 (13↑ / 6↓) |
| **Top ↑** | MBP, SCGB2A2, PLP1, KRT8, S100A11, MOBP, SPP1 | **SAA1, MGP, HOPX, GAP43, CALM2, ENC1, GRIA2** | **CARTPT, ENC1, CHN1, CKB, NSG2, HOPX** |
| **Top ↓** | SST, FABP7, HOPX, … | **MBP, GFAP, PTGDS, S100B, PLP1** | **SCGB2A2, MBP, PLP1, DBI** |
| Program sketch | Myelin / oligo-like **up** vs whole slice | Neuronal / mid-layer **up**; **myelin down** | Mid-layer neuronal **up**; **myelin down** |
| vs abutting same layer | **0** up genes padj≤0.05 (vs L6) | **10** up genes padj≤0.05 (vs L3) | **0** up genes padj≤0.05 (vs L3) |

### Interpretation

1. **A vs B (same slice, two components)** are **not the same biology**:
   - #1 is deep (L6) with relative myelin enrichment vs the rest of the slice.
   - #2 is mid-cortical (L3) with neuronal/plasticity markers (GAP43, GRIA2, ENC1, HOPX) and **suppressed** MBP/PLP1/GFAP — opposite myelin polarity.

2. **B vs C (L3 niches across two donors)** partially **replicate geometry**:
   - Both pure L3, only abut L3, compact (~65–69% internal edges).
   - Shared direction: **ENC1**, **HOPX** up; **MBP/PLP1** down vs rest.
   - C has fewer DE genes and **no** significant up-markers vs surrounding L3 (like A vs L6).
   - B uniquely shows 10 genes up even vs abutting L3 (includes SCGB/MUC1/SAA1 — treat epithelial-like hits as **artifact risk** until IF validates).

3. **SCGB2A2 / SCGB1D2** flip sign across components (up in A and B, down in C) → unstable / likely technical; do **not** use as discovery markers.

---

## Biological takeaway (honest)

| Claim | Status |
|-------|--------|
| Large cryptic niches are often **intra-layer** substructure | **Supported** (A, B, C + 151673) |
| Same-slice components can mark **different layers** with **opposite** marker polarity | **Supported** (A vs B) |
| L3-type cryptic niche **geometry** replicates across donors | **Supported** (508, 669, **673**) |
| L3-type **molecular direction** (L3↑ myelin↓ vs rest) | **Supported 3/3** |
| L3 same-layer hard gate | **Fails all three** — not a named state yet |
| Named new cell state / mechanism | **Not claimed** — need IF |

### Third donor `dlpfc_151673` (panel direction check)

| Component | n | Truth | L3 Δrest (shift p) | Myelin Δrest | direction_ok | hard pass |
|-----------|--:|-------|-------------------:|-------------:|:------------:|:---------:|
| L3 largest | 47 | 100% L3 | **+0.492 (0.035)** | **−0.609** | **Y** | N |
| L6 rank-1 | 24 | 100% L6 | −0.148 | +0.033 (p=0.34) | Y (weak) | N |

→ **L3 direction fully replicates on a third independent section.**

**Best next experiment:** IF for **ENC1 + HOPX + MBP** on 151508 / 151669 / **151673** ROIs (`IF_PROTOCOL.md`).

### Status of panel package

Pipeline: `validate_panel_and_rois.py` → `PANEL_VALIDATION_REPORT.md` + `IF_PROTOCOL.md`.

| Check | Result |
|-------|--------|
| L3 panel vs rest | ↑ on **3/3** (shift p 0.005 / 0.040 / **0.035**) |
| Myelin on L3 niches | ↓ on **3/3** |
| L6 myelin hard pass | **1/2** (508 PASS; 673 n=24 FAIL) |
| Hard same-layer L3 | **FAIL** all three |
| Cross-donor L3 **direction** | **YES (3/3)** |
| ROI CSVs | includes `ROI_151673_L3_n47.csv`, `ROI_151673_L6_n24.csv` |

**Implication:** third donor **confirms L3 direction**; still **no** same-layer hard pass — IF required before naming a state.

---

## How to reproduce

```bash
python research/discovery_uncertainty_niches/analyze_largest_component.py dlpfc_151508 0
python research/discovery_uncertainty_niches/analyze_largest_component.py dlpfc_151508 1
python research/discovery_uncertainty_niches/analyze_largest_component.py dlpfc_151669 0
python research/discovery_uncertainty_niches/analyze_largest_component.py dlpfc_151673 0
python research/discovery_uncertainty_niches/analyze_largest_component.py dlpfc_151673 1
python research/discovery_uncertainty_niches/validate_panel_and_rois.py
```

Per-component reports:

- `COMPONENT_REPORT_dlpfc_151508_rank0_n154.md`
- `COMPONENT_REPORT_dlpfc_151508_rank1_n138.md`
- `COMPONENT_REPORT_dlpfc_151669_rank0_n137.md`
- `COMPONENT_REPORT_dlpfc_151673_rank0_n47.md`
- `COMPONENT_REPORT_dlpfc_151673_rank1_n24.md`
