# KCNN4 / ORAI3 cryptic niche — function + abutting neighbourhood

**Dataset:** `xenium_human_lymph_node` · **component:** rank3 n=31  
**Protocol:** `histoweave.ca2_niche_neighborhood.v1`  
**Composed:** 2026-07-20

> Molecular **proxy** cell classes from gene panels — not protein-defined types.
> Pathology domain abutment remains 100% “Lymph node” (see prior GC deep-dive);
> this report asks what **molecular** neighbours sit outside the 31-cell niche.

![fig_niche_spatial_ca2](results/ca2_niche_neighborhood/figures/fig_niche_spatial_ca2.png)
![fig_abutting_classes](results/ca2_niche_neighborhood/figures/fig_abutting_classes.png)

---

## Literature integration: KCNN4 and ORAI3 in immune calcium signalling

### KCNN4 (KCa3.1 / IKCa1 / SK4)

**Molecular role.** KCNN4 encodes the intermediate-conductance Ca²⁺-activated K⁺
channel KCa3.1. Upon store-operated Ca²⁺ entry, KCa3.1 opens to hyperpolarize the
membrane and sustain the driving force for continued Ca²⁺ influx — a feed-forward
module of **activation-dependent calcium signalling** in lymphocytes
(Wulff et al., *J Clin Invest* / channel pharmacology literature; Cahalan & Chandy
reviews on K⁺ channels in T cells).

**Immune functions (established):**

| Context | Function |
|---------|----------|
| **T cells** | Sustains Ca²⁺ oscillations and NFAT-dependent activation after TCR engagement; pharmacologic KCa3.1 block dampens T-cell proliferation and cytokine production |
| **B cells** | Supports BCR-linked Ca²⁺ responses and aspects of B-cell activation / class switching in several models |
| **Other** | Also expressed in some myeloid and proliferative epithelial contexts — **not LN-specific**, so spatial context is required |

**Therapeutic / disease angle.** KCa3.1 inhibitors have been explored for autoimmune
and transplant indications precisely because they bias toward dampening
**pathologic lymphocyte activation** without globally deleting a lineage.

### ORAI3 (CRAC channel subunit)

**Molecular role.** ORAI proteins (ORAI1/2/3) form the plasma-membrane pore of the
**calcium-release-activated calcium (CRAC)** channel that opens after STIM sensors
detect ER Ca²⁺ store depletion. ORAI3 can hetero-multimerize with ORAI1 and
modulates CRAC amplitude and redox/sensitivity profiles (Feske; Prakriya & Lewis
CRAC reviews; ORAI3-specific studies in immune and non-immune cells).

**Immune functions (established / emerging):**

| Context | Function |
|---------|----------|
| **T-cell activation** | CRAC (primarily ORAI1-dominant in classic models) is **essential** for NFAT nuclear translocation and effector programs; ORAI3 contributes to channel diversity and can alter sustained Ca²⁺ plateaus |
| **B-cell maturation / activation** | Store-operated Ca²⁺ entry shapes BCR signalling thresholds; ORAI family members participate in B-cell Ca²⁺ signatures relevant to selection and activation |
| **Stromal / non-hematopoietic** | ORAI3 is also reported outside pure lymphoid lineages — so co-detection with immune vs stromal proxies is diagnostic |

**Together (KCNN4 + ORAI3).** Co-elevation is coherent with a **high Ca²⁺-throughput
activation niche**: ORAI-family store-operated entry + KCa3.1-mediated driving force.
In LN parenchyma this is a plausible signature of **locally activated lymphocytes**
and/or **stromal–immune contact zones** that sustain Ca²⁺ signalling, rather than a
classical dark-zone GC proliferation program (BCL6/MKI67-high).

### Stromal–immune interaction hypothesis (testable)

Lymph-node **fibroblastic reticular cells (FRCs)**, follicular dendritic cells (FDCs),
and conduits organize chemokine fields (CCL19/21, CXCL13) and present adhesive cues
(ICAM1/VCAM1) that trap and activate lymphocytes. KCNN4/ORAI3 co-elevation is
therefore most interesting when the niche sits at **lymphocyte–lymphocyte or
lymphocyte–stroma contacts** rather than inside BCL6-high GC dark zones.

**Pre-registered spatial predictions (tested below):**

| If external kNN enriched for… | Favoured reading |
|------------------------------|------------------|
| **T_like** | Parenchymal T-activation / help zone (Ca²⁺ machinery + T rim) |
| **B_like** | Extrafollicular B activation or T–B border |
| **stromal_like** | FRC/conduit-associated activation niche |
| **myeloid_like** | Sinus / macrophage interface |
| **GC_like** | Missed GC (would *weaken* the non-GC claim) |

**Key caveats.** Xenium gene panels give **proxy** cell classes, not gold-standard
protein phenotyping. KCNN4/ORAI3 can appear in non-lymphoid cells. Orthogonal CODEX
(KCNN4/ORAI3 + CD20/CD3/PDPN/CD68 + BCL6) remains the F3 protein gate.


---

## Expression inside the niche

Genes used on this assay for panels:  
{
  "B_like": [
    "MS4A1",
    "CD19",
    "CR2",
    "CD79A",
    "PAX5",
    "CD22",
    "CD79B",
    "FCER2"
  ],
  "T_like": [
    "CD3E",
    "CD4",
    "CD8A",
    "CCR7",
    "IL7"
  ],
  "GC_like": [
    "BCL6",
    "MKI67",
    "TOP2A",
    "PCNA",
    "LMO2",
    "CXCL13"
  ],
  "myeloid_like": [
    "CD68",
    "MARCO",
    "CD163",
    "CSF1R",
    "ITGAX"
  ],
  "stromal_like": [
    "PDPN",
    "CCL19",
    "VCAM1",
    "ICAM1",
    "PECAM1",
    "CXCL12"
  ],
  "ca2": [
    "KCNN4",
    "ORAI3",
    "STIM1",
    "MAP2K5",
    "MEF2A",
    "PRKCB",
    "NFATC1",
    "NFATC2"
  ]
}

### Ca²⁺ / activation genes

| Gene | present | mean_in | mean_out | frac+ in | frac+ out | log2FC |
|------|:-------:|--------:|---------:|---------:|----------:|-------:|
| `KCNN4` | Y | 1.097 | 0.243 | 0.29 | 0.07 | 2.17 |
| `ORAI3` | Y | 0.907 | 0.225 | 0.26 | 0.06 | 2.01 |
| `ORAI1` | N | — | — | — | — | — |
| `STIM1` | Y | 0.233 | 0.364 | 0.06 | 0.10 | -0.64 |
| `MAP2K5` | Y | 0.876 | 0.197 | 0.23 | 0.06 | 2.15 |
| `MEF2A` | Y | 0.953 | 0.241 | 0.26 | 0.07 | 1.98 |
| `PRKCB` | Y | 1.204 | 0.524 | 0.32 | 0.15 | 1.20 |
| `MS4A1` | Y | 1.816 | 1.326 | 0.42 | 0.30 | 0.45 |
| `CD3E` | Y | 2.026 | 2.228 | 0.48 | 0.52 | -0.14 |
| `BCL6` | Y | 0.000 | 0.100 | 0.00 | 0.03 | -16.61 |
| `PDPN` | Y | 0.000 | 0.046 | 0.00 | 0.01 | -15.49 |
| `CD68` | Y | 0.314 | 0.359 | 0.10 | 0.10 | -0.19 |

**Read.** Prefer rows with `present=Y` and elevated `frac_pos_in` or `log2FC>0`.
KCNN4/ORAI3 (when present on the panel) anchor the Ca²⁺ story; MAP2K5/MEF2A
support MAPK/NFAT-axis coherence from the prior same-domain DE.

---

## What is *inside* the 31 cells (proxy class)?

Proxy argmax (B / T / GC / myeloid / stromal / mixed_low): **B_like:9, myeloid_like:5, GC_like:5, stromal_like:5, T_like:4, mixed_low:3**

If the niche itself is mixed_low or multi-class, the signal is a **local milieu**,
not a pure single lineage cluster — consistent with an interaction zone.

---

## What abuts the niche? (external kNN, k=8)

Geometry from prior deep-dive: external edges dominated by pathology label
“Lymph node” (not GC polygons). Below: **molecular** class of those external neighbours.

### Observed external neighbour fractions

| Neighbour class | Obs frac (external kNN) | Null mean [95% CI] | Enrichment (obs/null) |
|-----------------|------------------------:|-------------------:|----------------------:|
| `B_like` | 0.204 | 0.218 [0.145, 0.298] | 0.94 |
| `GC_like` | 0.068 | 0.083 [0.048, 0.117] | 0.82 |
| `T_like` | 0.315 | 0.210 [0.154, 0.274] | 1.50 |
| `mixed_low` | 0.216 | 0.192 [0.145, 0.246] | 1.13 |
| `myeloid_like` | 0.080 | 0.122 [0.085, 0.161] | 0.66 |
| `stromal_like` | 0.117 | 0.175 [0.129, 0.230] | 0.67 |

- Internal kNN edges (within niche): **86**
- External kNN edges: **162**
- Primary abutment (per niche cell majority neighbour class):
  `{"T_like": 16, "B_like": 8, "GC_like": 2, "mixed_low": 4, "stromal_like": 1}`

### Enrichment vs size-matched random LN cells (n_null=200)

| Class | obs_frac | null_mean | enrich | outside_95CI |
|-------|---------:|----------:|-------:|:------------:|
| `T_like` | 0.315 | 0.210 | 1.50 | **Y** |
| `mixed_low` | 0.216 | 0.192 | 1.13 | N |
| `B_like` | 0.204 | 0.218 | 0.94 | N |
| `stromal_like` | 0.117 | 0.175 | 0.67 | **Y** |
| `myeloid_like` | 0.080 | 0.122 | 0.66 | **Y** |
| `GC_like` | 0.068 | 0.083 | 0.82 | N |

**Interpretation rules**

| Pattern | Favoured biological reading |
|---------|-----------------------------|
| External **B_like** enriched | Extrafollicular B activation / T–B border–like milieu |
| External **T_like** enriched | T-helper / activation synapse zone in parenchyma |
| External **stromal_like** enriched | FRC/conduit-associated immune activation niche |
| External **myeloid_like** enriched | Sinus / macrophage interface |
| **GC_like** *not* enriched | Supports non-GC organisation (with GC counter) |
| No class outside null 95% CI | Neighbours ≈ bulk LN — milieu not compositionally special; Ca²⁺ program is intrinsic |

---

## Integrated model (data-constrained hypothesis)

1. **Activation Ca²⁺ module** — KCNN4 and ORAI3 are both present and strongly
   elevated in the niche (log2FC ≈ 2.0–2.2; ~4× higher positive fraction than
   bulk), with MAP2K5/MEF2A/PRKCB coherent. Literature places this module in
   **TCR/BCR-linked sustained Ca²⁺ entry** (KCa3.1 driving force + CRAC pore).
2. **Niche interior is multi-lineage** — proxy mix of B / T / myeloid / stromal /
   GC-like / mixed (not a pure Leiden B or T cluster). That favours a **local
   interaction milieu** over a single new “cell type” name.
3. **Rim is T-enriched, GC-poor** — external kNN are enriched for **T_like**
   (obs ≈ 31% vs null ≈ 21%, outside 95% CI) while **GC_like remains rare**
   (~7%, not enriched). Stromal- and myeloid-like rims are if anything
   *under*-represented vs random LN. Together this supports a
   **parenchymal T-contact / activation-help zone** more than an FRC-sinus or
   missed-GC model — while B_like contacts stay near bulk LN baseline (~20%).
4. **Not a missed GC polygon** — BCL6 mean_in = 0; GC neighbour class not
   enriched; pathology adjacency was already 100% “Lymph node”.
5. **F3 tests:** CODEX KCNN4 + ORAI3 + CD3 + CD20 + PDPN + CD68 + BCL6; test
   whether KCNN4/ORAI3 protein sits on T and/or B membranes at T-rich rims.

---

## Methods

```bash
python research/discovery_xenium_lymph/analyze_ca2_niche_neighborhood.py
```

* Load official Xenium LN bundle via `get_dataset`.
* Component indices from `results/gc_deep_dive/component_rank3_n31/component_spots.csv`.
* Panel scores: mean z of present genes (HistoWeave `composite_score`).
* Proxy class = argmax(B,T,GC,myeloid,stromal) with margin → `mixed_low`.
* External kNN (k=8) class fractions; null = 200 random
  LN subsets of size n=31.
* Artifact genes (SCGB/SAA) not used for classification.

Artifacts: `results/ca2_niche_neighborhood/`.
