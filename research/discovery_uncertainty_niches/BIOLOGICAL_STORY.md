# Biological discovery story — HistoWeave uncertainty niches

**Protocol:** `histoweave.biological_story.v1` · composed `2026-07-20`  
**Headline discoveries:** **2** (D1 primary · D3 orthogonal experimental) + multi-donor L3 support (D2)

> **Honesty banner.** Wet-lab **protein IF** remains pending. D1 is frozen at
> **RNA panel + spatial-shift null PASS** with a pre-registered IF package.
> Simulated RNA→IF proxy must **not** be cited as protein validation.
> D3 uses **experimental Xenium** measurements with same-domain hard DE.

---

## Executive claim (what other tools cannot produce)

| # | Discovery | Tissue / assay | Highest honest claim | Why not Scanpy/Squidpy alone |
|--:|-----------|----------------|----------------------|------------------------------|
| **D1** | Intra-**Layer 6** myelin-concentrated cryptic niche | DLPFC Visium | RNA + spatial-null **PASS** on 151508 (n=154); hard_pass on **2** L6 components; **IF package ready** | No multi-method disagreement map inside L6; no cryptic=high-U∧¬boundary; no pre-registered myelin spatial null |
| **D3** | Intra-**LN** Ca²⁺-signaling cryptic niche (`KCNN4`/`ORAI3`/…) | Xenium human lymph node | **Same-domain hard DE** padj≤0.05 on 4 genes (single experimental section) | Pathology polygons label bulk LN only; no uncertainty-driven cryptic components |
| D2 (support) | Cross-donor **Layer 3** directional niches | DLPFC 12 sections / 3 donors | Donor-stratified CI excludes 0 (L3↑ myelin↓); same-layer hard **FAIL** | Direction without IF ≠ new cell state — HistoWeave *blocks* overclaim |

![fig1_cohort_panel_gates](results/biological_story/figures/fig1_cohort_panel_gates.png)
![fig2_capability_matrix](results/biological_story/figures/fig2_capability_matrix.png)
![fig3_headline_discoveries](results/biological_story/figures/fig3_headline_discoveries.png)

---

## Discovery 1 — Intra-L6 myelin niche (primary IF-ready finding)

### Statement

On DLPFC section **151508**, multi-method boundary uncertainty recovers a **compact, pure Layer-6** cryptic component (**n = 154** spots) whose external contacts are **100% Layer 6**. This is **intra-layer substructure**, not a layer-edge ribbon.

A pre-registered **myelin panel** (`MBP`, `PLP1`, `MOBP`) is elevated versus the rest of the section:

| Metric | Value | Gate |
|--------|------:|------|
| Myelin Δ vs rest | **+0.497** | direction |
| Spatial-shift *p* (rest) | **0.005** | **PASS** (≤0.05) |
| Internal edge fraction | 0.68 | compact blob |
| Cohort L6 hard_pass | **2 / 9** | multi-slice |

Independent hard_pass also appears on **151672** L6 (n=26; myelin shift p = 0.03).

### Biological interpretation

The niche is consistent with a **myelin-rich micro-domain inside deep cortical Layer 6** — e.g. local oligodendrocyte / myelinated-fibre enrichment — that standard single-partition domain methods fold into a monolithic “Layer 6” label. Manual anatomy and single-method ARI benchmarks therefore **cannot** surface it.

### Experimental validation status

| Level | Evidence | Status |
|------:|----------|--------|
| 0 Geometry | pure L6 contiguous cryptic component | **DONE** |
| 1 RNA direction | myelin ↑ vs rest | **DONE** |
| 2 Spatial null | shift p = 0.005 | **DONE** |
| 2b Multi-slice hard_pass | 151508 + 151672 | **DONE** |
| 2c RNA-proxy IF pipeline | MBP gate PASS (simulated) | pipeline only |
| **3 Protein IF** | MBP on ROI vs rest (padj≤0.05) | **PENDING** — see `IF_PROTOCOL.md` |

**IF hand-off (pre-registered):**

- ROI: `results/panel_validation/ROI_151508_L6_n154.csv`
- Antibody: **MBP** (+ optional PLP1)
- Pass: MBP higher in ROI vs non-ROI (Mann–Whitney + BH)

### Why HistoWeave was required

- Single-method layer clustering returns one Layer-6 label — no intra-layer disagreement map.
- Target-free multi-method boundary_uncertainty finds compact high-U blobs inside Layer 6.
- Cryptic mask (high-U ∧ ¬ known boundary) excludes layer-edge ribbons by construction.
- Pre-registered myelin panel + spatial-shift null blocks overclaiming from raw DE.

---

## Discovery 3 — Xenium LN Ca²⁺-signaling niche (orthogonal experiment)

### Statement

Applying the **identical** uncertainty-niche architecture to **official 10x Xenium** human lymph node counts yields contiguous cryptic components almost entirely **off** coarse pathology boundaries (AUROC(U→boundary) ≈ 0.44; cryptic ≈ 99% of high-U cells).

Component **rank-3 (n=31)** lies entirely inside pathology **“Lymph node”** (not the GC polygon) yet shows a **same-domain hard** marker program:

| Gene | log2FC vs same LN domain | padj |
|------|-------------------------:|-----:|
| `KCNN4` | 2.12 | 1.8×10⁻³ |
| `MAP2K5` | 2.13 | 1.2×10⁻² |
| `ORAI3` | 1.99 | 1.2×10⁻² |
| `MEF2A` | 1.99 | 1.2×10⁻² |

Classical GC / B / T panels are **not** enriched — this is **not** a missed germinal center under a coarse polygon. It is a **Ca²⁺ / MAPK-linked transcriptional niche** inside bulk LN parenchyma that multi-method disagreement isolates.

**Literature + spatial neighbourhood (deep dive):**
[`../discovery_xenium_lymph/KCNN4_ORAI3_NEIGHBORHOOD.md`](../discovery_xenium_lymph/KCNN4_ORAI3_NEIGHBORHOOD.md)

- **KCNN4 (KCa3.1)** sustains Ca²⁺ driving force in activated T/B cells; **ORAI3** is a CRAC-channel subunit shaping store-operated Ca²⁺ entry — co-elevation = high-throughput activation Ca²⁺ logic.
- Niche interior is **multi-lineage** (B/T/myeloid/stromal mix), not a pure cluster.
- External kNN: **T_like ~31%** (enriched vs random LN null); **GC_like ~7%** (not enriched) → favours a **parenchymal T-contact activation zone** over missed GC or sinus-dominant stroma.

### Experimental status

| Item | Detail |
|------|--------|
| Platform | Xenium (imaging-based spatial transcriptomics) — **experimental measurement** |
| Expression source | `official_10x_cell_feature_matrix` |
| Hard gate | same-domain DE **PASS** (unlike DLPFC L3 same-layer) |
| Replication | single section — second LN sample pending |
| Protein IF | optional next (KCNN4 / ORAI3) |

### Why HistoWeave was required

- Pathology polygons alone label bulk LN — no intra-LN disagreement niches.
- Same pipeline as DLPFC (non-oracle K, multi-method U, cryptic mask) transfers cross-tissue.
- Same-domain hard DE is the LN analogue of same-layer hard gate — and *passes* here.

---

## Supporting discovery D2 — Multi-donor L3 directional program

Across **15** pure L3 cryptic components on **12** DLPFC sections:

- Direction OK (L3 panel ↑ and myelin ↓ vs rest): **14 / 15**
- Donor-stratified bootstrap (3 donors, 14 direction_ok components):
  - L3 Δrest **0.288** 95% CI **[0.222, 0.345]**
  - Myelin Δrest **-0.354** 95% CI **[-0.377, -0.325]**
  - Both CIs **exclude 0**
- Same-layer hard_pass: **0 / 15** → **no named cell state without IF**

Primary IF ROIs: 151508 L3 (n=138), 151669 L3 (n=137). Panel: **ENC1, HOPX, MBP**.

This is a **replicable geometric + directional RNA finding**, not yet protein-validated biology.

---

## Cross-tissue narrative (one architecture, two tissues)

```
                    non-oracle K
                         │
              multi-method domain ensemble
                         │
           target-free boundary_uncertainty
                         │
         cryptic = high-U ∧ ¬ known boundary
                         │
              contiguous components
                    ╱         ╲
           DLPFC Visium      Xenium LN
         L6 myelin niche    Ca²⁺ signaling niche
         L3 multi-donor dir   same-domain DE PASS
                │                   │
         IF package (MBP)     experimental Xenium
```

| Axis | D1 (DLPFC L6) | D3 (Xenium LN) |
|------|---------------|----------------|
| Hidden by single partition? | Yes (monolithic L6) | Yes (bulk LN polygon) |
| Survives spatial / same-domain hard gate? | Yes (shift p=0.005) | Yes (4 genes padj≤0.05) |
| Multi-slice / multi-donor? | 2 hard_pass L6 comps | 1 section |
| Wet-lab protein | IF ready | optional |

---

## Claim ladder (frozen)

| Level | Name | D1 L6 | D2 L3 | D3 Xenium LN |
|------:|------|:-----:|:-----:|:------------:|
| 0 | Geometric candidate | ✓ | ✓ | ✓ |
| 1 | RNA / panel direction | ✓ | ✓ multi-donor CI | ✓ |
| 2 | Spatial / same-domain hard null | ✓ | ✗ | ✓ |
| 2b | Multi-slice hard_pass | ✓ (2 comps) | ✗ | pending |
| 3 | Protein IF | **pending** | pending | optional |
| 4 | Multi-donor protein | — | pending | — |

**Allowed language for D1 today:**  
“IF-ready, multi-gate-validated **myelin-concentrated Layer-6 cryptic niche** discovered by multi-method uncertainty.”

**Forbidden language until IF returns:**  
“Protein-validated new cell type/state.”

---

## Methods snapshot (reproducible)

```bash
# DLPFC track
histoweave discovery run
histoweave discovery cohort
histoweave discovery bootstrap-ci
histoweave discovery panel
histoweave discovery if-package

# Xenium track
python research/discovery_xenium_lymph/run_discovery_ln.py
python research/discovery_xenium_lymph/analyze_gc_components.py

# Compose this story
python research/discovery_uncertainty_niches/compose_biological_story.py
```

Artifacts: `results/biological_story/story_metrics.json`, figures under `results/biological_story/figures/`.

---

## Global decision

Do not cite simulated IF as protein validation. D1 is IF-ready with RNA panel + spatial-null PASS on two slices. D3 is experimentally measured on Xenium with same-domain hard DE.

**Primary story for external presentation:** **D1 + D3**.  
**Wet-lab next step:** run IF on `ROI_151508_L6_n154` (MBP) and dual L3 ROIs (ENC1/HOPX/MBP); drop CSVs into `results/if_return/` and re-run `analyze_if_return.py` without `--simulate-from-rna`.

---

## Functional validation upgrade (F2)

Computational **dual-axis** mapping (disease mechanism + spatial-organisation
redefinition) for D1, D2, and D3 is frozen in
**[FUNCTIONAL_VALIDATION.md](FUNCTIONAL_VALIDATION.md)**.

| Discovery | Proposed state | Disease axis | Organisation redefinition | F-level |
|-----------|----------------|--------------|---------------------------|---------|
| D1 | L6-myelin microcompartment | Myelin core + mito trade-off | L6 ≠ single molecular mantle | **F2** |
| D2 | L3-plasticity microcompartment | Mid-layer plasticity program | L3 multi-compartment; not edge ribbon | **F2** |
| D3 | LN Ca²⁺-signaling micro-niche | KCNN4/ORAI3/MAP2K5/MEF2A | Bulk LN ≠ GC polygon field | **F2** |

```bash
python research/discovery_uncertainty_niches/run_functional_validation.py
```

F2 is **not** F3 protein validation and **not** F4 disease-cohort proof.

### F3 / F4 experiment package (perturbation · lineage · orthogonal)

Pre-registered upgrades live in **[FUNCTIONAL_EXPERIMENTS.md](FUNCTIONAL_EXPERIMENTS.md)**:

| Class | Examples | Level |
|-------|----------|-------|
| Orthogonal platform | MERFISH/Xenium myelin; multiome L3; CODEX LN | **F3** |
| Lineage tracing | OPC reporter → L6 ROI; HOPX lineage → L3; immune barcodes | **F3** |
| Perturbation | CRISPRi MYRF/OLIG2/ENC1/KCNN4; cuprizone; TTX; MEK/SOCE drugs | **F4** |

```bash
python research/discovery_uncertainty_niches/prepare_functional_experiment_package.py
python research/discovery_uncertainty_niches/analyze_functional_return.py --dry-run
```
