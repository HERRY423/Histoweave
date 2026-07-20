# Functional experiments — perturbation, lineage, orthogonal platforms

**Protocol:** `histoweave.functional_experiments.v1`  
**n experiments:** 12  
**Classes:** perturbation=6, lineage=3, orthogonal=3

> **Scope.** Pre-registered **F3/F4** experiments that can upgrade
> computational F2 cryptic-state claims. No wet-lab outcomes are claimed
> until returns pass `analyze_functional_return.py`.

---

## How this upgrades claim levels

| Level | Evidence class | Typical experiment |
|------:|----------------|--------------------|
| F2 | Computational dual-axis | modules on DE (done) |
| **F3** | Orthogonal assay / lineage map | MERFISH/Xenium/CODEX; lineage density |
| **F4** | Perturbation causality | CRISPR; drug; demyelination model |

```
F2 computational map
        │
        ├─► Orthogonal platform (F3) ──► same program, new assay
        ├─► Lineage tracing (F3) ─────► cell-of-origin / descendant map
        └─► CRISPR / drug (F4) ───────► necessity / disease mechanism
```

---

## Experiment matrix (summary)

| ID | Disc | Class | Modality | Prio | Level |
|----|------|-------|----------|-----:|-------|
| `D1_crispri_myrf_olig2` | D1_L6_myelin | perturbation | crispr | 1 | F4 |
| `D1_lineage_opc_reporter` | D1_L6_myelin | lineage | lineage_reporter | 1 | F3 |
| `D1_orthogonal_merfish_xenium_myelin` | D1_L6_myelin | orthogonal | merfish_or_xenium | 1 | F3 |
| `D1_drug_demyelination_cuprizone` | D1_L6_myelin | perturbation | drug | 2 | F4 |
| `D2_crispri_enc1_hopx` | D2_L3_plasticity | perturbation | crispr | 1 | F4 |
| `D2_orthogonal_multiome_l3` | D2_L3_plasticity | orthogonal | multiome_or_snrna_spatial | 1 | F3 |
| `D2_drug_activity_block_ttx` | D2_L3_plasticity | perturbation | drug | 2 | F4 |
| `D2_lineage_hopx_ip` | D2_L3_plasticity | lineage | lineage_reporter | 2 | F3 |
| `D3_crispr_kcnn4_orai3` | D3_LN_ca2 | perturbation | crispr | 1 | F4 |
| `D3_lineage_immune_barcode` | D3_LN_ca2 | lineage | lineage_barcode | 1 | F3 |
| `D3_orthogonal_codex_second_ln` | D3_LN_ca2 | orthogonal | codex_or_xenium | 1 | F3 |
| `D3_drug_ca_mek_inhibitors` | D3_LN_ca2 | perturbation | drug | 2 | F4 |

Full table: `results/functional_experiments/EXPERIMENT_MATRIX.csv`.

---

## D1 — L6 myelin microcompartment

| Class | Experiment | Key targets | Pass gist |
|-------|------------|-------------|-----------|
| Orthogonal | MERFISH/Xenium myelin | MBP PLP1 MOBP | myelin Δrest>0, shift p≤0.05 on new platform |
| Lineage | OPC reporter | PDGFRA/OLIG2 | lineage density ↑ in ROI |
| CRISPR | MYRF/OLIG2/SOX10 CRISPRi | oligo drivers | myelin program ↓, layer ID intact |
| Drug | Cuprizone/LPC demyelination | myelin integrity | niche myelin program shrinks then recovers |

## D2 — L3 plasticity microcompartment

| Class | Experiment | Key targets | Pass gist |
|-------|------------|-------------|-----------|
| Orthogonal | multiome/snRNA+spatial | ENC1 HOPX GAP43 | mid-layer state maps into L3 ROI |
| Lineage | HOPX-CreERT2 | HOPX | lineage enriched in L3 cryptic ROI |
| CRISPR | ENC1/HOPX CRISPRi | plasticity | module ↓ |
| Drug | TTX / experience | activity | module moves in pre-registered direction |

## D3 — LN Ca²⁺ micro-niche

| Class | Experiment | Key targets | Pass gist |
|-------|------------|-------------|-----------|
| Orthogonal | CODEX / 2nd Xenium | KCNN4 ORAI3 BCL6 | protein or 2nd donor same-domain ↑; not GC |
| Lineage | CITE-seq / immune barcode | lineage + Ca²⁺ | lineage composition non-random |
| CRISPR | KCNN4/ORAI3 KO | Ca²⁺ | module ↓; GC counter holds |
| Drug | SOCE / MEK inhibitors | Ca²⁺ MAPK | module ↓ at pre-registered dose |

---

## Detailed registry

### `D1_crispri_myrf_olig2`

- **Discovery:** `D1_L6_myelin`
- **Class / modality:** perturbation / crispr
- **Claim on pass:** F4 (priority 1)
- **Title:** CRISPRi of MYRF / OLIG2 in human cortical organoids or slice culture
- **Hypothesis:** Reducing oligodendrocyte-lineage drivers shrinks or transcriptionally erases the L6-myelin microcompartment program (MBP/PLP1/MOBP down in ROI-matched deep-layer zones) without collapsing bulk Layer-6 identity.
- **System:** human iPSC cortical organoid ± slice culture; optional mouse L6 slice
- **Targets:** `MYRF`, `OLIG2`, `SOX10`
- **Readouts:** spatial transcriptomics or smFISH panel MBP/PLP1/MOBP; IF MBP + SOX10; layer marker controls (e.g. FOXP2/TLE4 for deep layers)
- **Contrast:** perturbed vs non-targeting gRNA; ROI-matched deep layer vs rest
- **Controls:** non-targeting gRNA, scrambled, vehicle for any drug arm
- **Min n:** 3
- **Pass criteria:**
  - MBP and/or PLP1 ROI-vs-rest Δ decreases vs control (padj≤0.05, n≥3)
  - deep-layer identity markers not globally ablated (fold-change within ±20% or padj>0.05)
  - effect direction pre-registered: myelin program ↓ under CRISPRi
- **Related ROI:** `ROI_151508_L6_n154`
- **Notes:** Causal test of oligodendrocyte program necessity for the niche signature.

### `D1_drug_demyelination_cuprizone`

- **Discovery:** `D1_L6_myelin`
- **Class / modality:** perturbation / drug
- **Claim on pass:** F4 (priority 2)
- **Title:** Cuprizone (or lysolecithin) demyelination with spatial readout
- **Hypothesis:** Induced demyelination remaps multi-method uncertainty niches in deep cortex and reduces L6-myelin microcompartment markers; recovery phase partially restores the niche.
- **System:** mouse cortex (cuprizone diet) or local LPC lesion; spatial Visium/Xenium
- **Targets:** `myelin integrity`
- **Readouts:** HistoWeave uncertainty niche pipeline on treated vs control sections; MBP/PLP1 IF; cryptic component size and myelin panel Δrest
- **Contrast:** treated demyelinated vs control age-matched; recovery time course
- **Controls:** vehicle diet, contralateral unlesioned hemisphere for LPC
- **Min n:** 4
- **Pass criteria:**
  - myelin panel Δrest in deep-layer cryptic components decreases under demyelination (padj≤0.05)
  - cryptic component geometry remains measurable (not pure dropout of all spots)
  - partial recovery of myelin panel at remyelination time point (direction pre-registered)
- **Related ROI:** `ROI_151508_L6_n154`
- **Notes:** Disease-mechanism stress test linking niche to demyelination vulnerability.

### `D1_lineage_opc_reporter`

- **Discovery:** `D1_L6_myelin`
- **Class / modality:** lineage / lineage_reporter
- **Claim on pass:** F3 (priority 1)
- **Title:** OPC lineage reporter mapped onto L6 cryptic ROI geometry
- **Hypothesis:** PDGFRA+ or OLIG2-lineage cells are enriched inside L6-myelin cryptic components relative to adjacent L6 non-ROI.
- **System:** Pdgfra-CreERT2;Rosa-tdTomato (or Olig2-lineage) mouse; adult pulse-chase
- **Targets:** `PDGFRA`, `OLIG2`, `MBP`
- **Readouts:** reporter+ cell density in ROI vs same-layer non-ROI; co-IF MBP; optional spatial RNA of reporter sorted cells
- **Contrast:** reporter density ROI vs same-layer L6 non-ROI
- **Controls:** oil vehicle no tamoxifen, non-cryptic L6 ROIs
- **Min n:** 3
- **Pass criteria:**
  - reporter+ density higher in ROI (padj≤0.05, n≥3 animals)
  - ≥30% of ROI MBP+ area co-localizes with lineage label OR lineage cells show MBP program
- **Related ROI:** `ROI_151508_L6_n154`

### `D1_orthogonal_merfish_xenium_myelin`

- **Discovery:** `D1_L6_myelin`
- **Class / modality:** orthogonal / merfish_or_xenium
- **Claim on pass:** F3 (priority 1)
- **Title:** Imaging-based ST (MERFISH/Xenium) confirmation of L6 myelin microcompartment
- **Hypothesis:** On an independent brain section/platform, multi-method uncertainty recovers a pure deep-layer cryptic niche with myelin-panel elevation surviving spatial-shift null.
- **System:** human DLPFC or mouse homologous deep cortex; MERFISH or Xenium myelin panel
- **Targets:** `MBP`, `PLP1`, `MOBP`, `SOX10`, `deep-layer controls`
- **Readouts:** HistoWeave discovery pipeline on orthogonal counts; myelin panel Δrest + shift p; component purity for deep layer / L6-homologue
- **Contrast:** cryptic component vs rest; vs same deep-layer background
- **Controls:** technical replicate section, white-matter positive control for MBP
- **Min n:** 2
- **Pass criteria:**
  - ≥1 pure deep-layer cryptic component with myelin Δrest>0 and shift p≤0.05
  - SCGB/SAA artifact genes not required for pass
  - platform ≠ original Visium 151508 bundle
- **Related ROI:** `ROI_151508_L6_n154`
- **Notes:** Orthogonal platform confirmation without requiring CRISPR.

### `D2_crispri_enc1_hopx`

- **Discovery:** `D2_L3_plasticity`
- **Class / modality:** perturbation / crispr
- **Claim on pass:** F4 (priority 1)
- **Title:** CRISPRi ENC1 and/or HOPX in mid-layer cortical models
- **Hypothesis:** Knockdown of ENC1/HOPX reduces the L3-plasticity niche signature (GAP43/GRIA2/NRGN program) inside mid-layer ROIs more than bulk L3 markers.
- **System:** human cortical organoid or primary culture; optional in vivo AAV-CRISPRi
- **Targets:** `ENC1`, `HOPX`, `GAP43`
- **Readouts:** smFISH/IF ENC1 HOPX GAP43 GRIA2; spatial or ROI-bulk RNA plasticity module score
- **Contrast:** CRISPRi vs non-targeting; mid-layer ROI vs rest
- **Controls:** non-targeting gRNA
- **Min n:** 3
- **Pass criteria:**
  - plasticity module score ↓ in mid-layer under CRISPRi (padj≤0.05, n≥3)
  - MBP remains not elevated (no ectopic myelin program)
- **Related ROI:** `ROI_151508_L3_n138`

### `D2_drug_activity_block_ttx`

- **Discovery:** `D2_L3_plasticity`
- **Class / modality:** perturbation / drug
- **Claim on pass:** F4 (priority 2)
- **Title:** Activity blockade (TTX) or plasticity challenge on mid-layer niches
- **Hypothesis:** Silencing network activity shrinks L3-plasticity cryptic programs; enriched sensory experience expands them — linking niche to plasticity state.
- **System:** acute cortical slice or organoid; optional chronic monocular deprivation analogue
- **Targets:** `network activity`, `plasticity genes`
- **Readouts:** GAP43/NRGN/ENC1 module; uncertainty niche size; IF HOPX
- **Contrast:** TTX vs vehicle; or enriched vs standard housing (pre-registered arm)
- **Controls:** vehicle, time-matched untreated
- **Min n:** 3
- **Pass criteria:**
  - plasticity module Δ in L3 cryptic ROI moves in pre-registered direction (padj≤0.05)
  - geometry of layer boundaries stable (not a global tissue collapse)
- **Related ROI:** `ROI_151508_L3_n138`

### `D2_lineage_hopx_ip`

- **Discovery:** `D2_L3_plasticity`
- **Class / modality:** lineage / lineage_reporter
- **Claim on pass:** F3 (priority 2)
- **Title:** HOPX+ intermediate progenitor lineage contribution to L3 cryptic niches
- **Hypothesis:** HOPX-lineage descendants are enriched in L3 cryptic plasticity niches versus adjacent L3 non-ROI.
- **System:** Hopx-CreERT2 lineage mouse (developmental pulse) or human organoid barcoded iPSC
- **Targets:** `HOPX`, `ENC1`, `SATB2`
- **Readouts:** lineage density ROI vs same-layer; co-expression with plasticity panel
- **Contrast:** lineage+ density in L3 ROI vs L3 non-ROI
- **Controls:** no-tamoxifen, L2/L4 control ROIs
- **Min n:** 3
- **Pass criteria:**
  - lineage enrichment in ROI (padj≤0.05, n≥3)
  - lineage cells show higher plasticity module than non-lineage L3 neighbours
- **Related ROI:** `ROI_151508_L3_n138`

### `D2_orthogonal_multiome_l3`

- **Discovery:** `D2_L3_plasticity`
- **Class / modality:** orthogonal / multiome_or_snrna_spatial
- **Claim on pass:** F3 (priority 1)
- **Title:** snRNA-seq/multiome + spatial joint confirmation of L3 sub-compartments
- **Hypothesis:** Independent single-cell modality recovers a mid-layer state with ENC1/HOPX/GAP43 program that spatially maps into cryptic L3 components.
- **System:** matched human DLPFC multiome or snRNA + Visium/Xenium same donor
- **Targets:** `ENC1`, `HOPX`, `GAP43`, `GRIA2`, `MBP`
- **Readouts:** cluster marker table; spatial deconvolution into L3 ROI; module scores
- **Contrast:** state enriched in L3 ROI vs other L3 cells
- **Controls:** white-matter nuclei, L6 control
- **Min n:** 2
- **Pass criteria:**
  - ≥1 mid-layer state with plasticity module ↑ and myelin ↓ vs other L3 (padj≤0.05)
  - spatial mapping enriches that state inside pre-registered L3 cryptic ROIs
- **Related ROI:** `ROI_151508_L3_n138`

### `D3_crispr_kcnn4_orai3`

- **Discovery:** `D3_LN_ca2`
- **Class / modality:** perturbation / crispr
- **Claim on pass:** F4 (priority 1)
- **Title:** CRISPR KO/KD of KCNN4 and/or ORAI3 in LN organoids or tonsil explants
- **Hypothesis:** Loss of KCNN4/ORAI3 collapses the Ca²⁺/MAPK niche signature and reduces activation tone without converting the niche into a classical GC program.
- **System:** tonsil explant, LN organoid, or activated B/T co-culture with spatial readout
- **Targets:** `KCNN4`, `ORAI3`, `MAP2K5`
- **Readouts:** Ca²⁺ flux imaging; panel KCNN4/ORAI3/MEF2A/BCL6; phospho-ERK optional
- **Contrast:** KO/KD vs AAVS1/safe-harbor control
- **Controls:** safe-harbor gRNA, untreated explant
- **Min n:** 3
- **Pass criteria:**
  - Ca²⁺ module score ↓ (padj≤0.05, n≥3)
  - BCL6/MKI67 not significantly up (GC counter still holds)
  - viability >70% of control
- **Related ROI:** `xenium_rank3_n31`

### `D3_drug_ca_mek_inhibitors`

- **Discovery:** `D3_LN_ca2`
- **Class / modality:** perturbation / drug
- **Claim on pass:** F4 (priority 2)
- **Title:** Ca²⁺ flux and MEK/ERK inhibitors on LN spatial niches
- **Hypothesis:** Pharmacologic blockade of store-operated Ca²⁺ entry or MEK reduces the cryptic Ca²⁺ niche program in situ.
- **System:** human tonsil/LN explant culture with CODEX or Xenium end-point
- **Targets:** `SOCE`, `MEK1/2`
- **Readouts:** KCNN4/ORAI3/MEF2A module; pERK; GC panel BCL6/Ki67
- **Contrast:** inhibitor vs vehicle; dose pre-registered
- **Controls:** vehicle, inactive analogue if available
- **Min n:** 3
- **Pass criteria:**
  - Ca²⁺ module ↓ at pre-registered dose (padj≤0.05)
  - GC panel not increased (no compensatory GC conversion)
- **Related ROI:** `xenium_rank3_n31`
- **Notes:** Example tool compounds: 2-APB/SOCE blockers; trametinib/MEK class — final choice by local pharmacology SOP.

### `D3_lineage_immune_barcode`

- **Discovery:** `D3_LN_ca2`
- **Class / modality:** lineage / lineage_barcode
- **Claim on pass:** F3 (priority 1)
- **Title:** Immune lineage barcoding / CITE-seq to assign niche cell of origin
- **Hypothesis:** The Ca²⁺ cryptic niche is enriched for a specific immune lineage (e.g. B or innate-like) rather than random LN parenchyma mixture.
- **System:** human LN/tonsil CITE-seq + spatial; or mouse lineage-traced immune subsets
- **Targets:** `MS4A1`, `CD3E`, `CD68`, `KCNN4`, `ORAI3`
- **Readouts:** lineage fraction in niche vs rest; module score per lineage
- **Contrast:** lineage composition niche vs bulk LN
- **Controls:** GC polygon cells, adipose domain if present
- **Min n:** 3
- **Pass criteria:**
  - ≥1 lineage shows enrichment OR exclusive high Ca²⁺ module (padj≤0.05)
  - niche is not a random sample of bulk LN (χ² or permutation p≤0.05)
- **Related ROI:** `xenium_rank3_n31`

### `D3_orthogonal_codex_second_ln`

- **Discovery:** `D3_LN_ca2`
- **Class / modality:** orthogonal / codex_or_xenium
- **Claim on pass:** F3 (priority 1)
- **Title:** CODEX/IMC protein or second-donor Xenium confirmation
- **Hypothesis:** Protein-level KCNN4/ORAI3 (or second LN Xenium) recovers a non-GC cryptic niche with same-domain hard DE / protein elevation.
- **System:** second human LN donor; CODEX/IMC or Xenium
- **Targets:** `KCNN4`, `ORAI3`, `MEF2A`, `BCL6`, `CD20`, `CD3`
- **Readouts:** protein intensity ROI; or Xenium same-domain DE
- **Contrast:** niche vs same LN domain; vs true GC
- **Controls:** pathologist-marked GC, T-zone
- **Min n:** 2
- **Pass criteria:**
  - KCNN4 or ORAI3 ↑ vs same-domain (padj≤0.05)
  - BCL6 not GC-like high in niche vs true GC
  - independent donor or independent platform from first Xenium run
- **Related ROI:** `xenium_rank3_n31`

---

## Return analysis

```bash
# Build / refresh this package
python research/discovery_uncertainty_niches/prepare_functional_experiment_package.py

# Schema check (no data)
python research/discovery_uncertainty_niches/analyze_functional_return.py --dry-run

# Score real returns
python research/discovery_uncertainty_niches/analyze_functional_return.py
```

Analyzer writes `results/functional_experiments/RETURN_REPORT.md` and
updates claim status JSON. Simulated data must be labelled
`notes` containing `SIMULATED` or use `--simulate` (explicitly non-claim).

---

## Artifact & stats continuity

- SCGB/SAA are **not** valid primary readouts (see FUNCTIONAL_VALIDATION.md).
- D3 GC counter remains a non-enrichment control; orthogonal CODEX must show
  BCL6 not GC-like high in the niche.
- All pass criteria are pre-registered in `functional_experiments.py`.

## Honesty banner

* Package ≠ completed F3/F4 validation.
* CRISPR/drug parameters require local biosafety and pharmacology approval.
* Do not cite empty returns or `--simulate` as causal proof.
