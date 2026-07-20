# Claim ladder — geometric candidate → validated biology

| Level | Name | Status | Evidence |
|------:|------|--------|----------|
| 0 | Geometric candidate | **DONE** | Multi-method uncertainty + contiguous cryptic components (12 DLPFC + Xenium LN) |
| 1 | RNA direction (vs rest) | **DONE** | L3 14/15 cohort; donor-stratified CI excludes 0; L6 myelin direction |
| 2 | Spatial / same-domain hard null | **D1 PASS · D2 FAIL · D3 PASS** | L6 myelin shift p=0.005; L3 same-layer fail; Xenium same-domain DE (KCNN4/ORAI3/…) |
| 2b | Multi-slice hard_pass | **D1: 2 L6 comps** | 151508 L6 n=154 + 151672 L6 n=26 |
| 2c | RNA multiplex proxy on IF ROIs | **L6_proxy=PASS · L3_proxy=FAIL** | Pipeline dry-run only — not protein |
| 3 | **Protein IF same-layer / ROI** | **PENDING** | Wet-lab return → `analyze_if_return.py` (no `--simulate-from-rna`) |
| 4 | Multi-donor protein | PENDING | 151508 + 151669 L3 both IF-pass |

## Headline discoveries (see `BIOLOGICAL_STORY.md`)

| ID | Finding | Highest honest level |
|----|---------|----------------------|
| **D1** | Intra-L6 myelin-concentrated cryptic niche | **2b** (IF-ready); Level 3 pending protein MBP |
| **D3** | Xenium LN Ca²⁺-signaling cryptic niche | **2** experimental (single section) |
| D2 | Cross-donor L3 directional niches | **1** (CI excludes 0; hard gate FAIL) |

## Narrative rules

| If… | Allowed language |
|-----|------------------|
| Level ≤1 only | geometric / RNA-directional candidate |
| Level 2–2b pass, IF pending | multi-gate-validated molecular niche; **IF-ready**; **not** protein-validated cell state |
| Level 3 pass on 151508 L6 | **protein-validated L6 myelin niche on one section** |
| Level 3 pass on 151508 L3 **and** L6 | **validated dual-niche biology on one section** |
| Level 4 pass | cross-donor validated L3 niche program |

**Never** claim a single unified “cryptic cell state” for L3+L6.  
**Never** cite simulated IF (`--simulate-from-rna`) as protein validation.

---

## Functional validation ladder (disease + organisation)

See **[FUNCTIONAL_VALIDATION.md](FUNCTIONAL_VALIDATION.md)** (re-run:
`python research/discovery_uncertainty_niches/run_functional_validation.py`).

| Level | Name | Freeze status |
|------:|------|---------------|
| F0 | Geometry only | superseded for D1–D3 |
| F1 | Single-axis functional map | — |
| **F2** | **Dual-axis (disease + organisation)** | **D1, D2, D3 all PASS** |
| F3 | Orthogonal platform / lineage / protein IF | **PENDING** — see `FUNCTIONAL_EXPERIMENTS.md` |
| F4 | CRISPR / drug / demyelination perturbation | **PENDING** — pre-registered; no claim until returns PASS |

| Discovery | Disease axis | Organisation redefinition | F-level |
|-----------|--------------|---------------------------|---------|
| D1 L6 myelin microcompartment | Myelin core + mito trade-off | Intra-L6 not uniform mantle | **F2** |
| D2 L3 plasticity microcompartment | Mid-layer plasticity stress + myelin deplete | Intra-L3 multi-compartment + anti-glial-boundary | **F2** |
| D3 LN Ca²⁺ micro-niche | KCNN4/ORAI3/MAP2K5/MEF2A | Bulk LN ≠ GC; non-GC micro-niche | **F2** |

F2 is **computational functional mapping**, not atlas-grade new cell-type registration
and not patient-level disease proof.

### F3 / F4 experiment classes (pre-registered)

| Class | Modalities | Claim on PASS | Entry point |
|-------|------------|---------------|-------------|
| Orthogonal | MERFISH, Xenium, CODEX, multiome | **F3** | `functional_experiments.py` |
| Lineage | OPC/HOPX reporters, immune barcodes | **F3** | same |
| Perturbation | CRISPR, drugs, cuprizone | **F4** | same |

```bash
python research/discovery_uncertainty_niches/prepare_functional_experiment_package.py
python research/discovery_uncertainty_niches/analyze_functional_return.py --dry-run
# after lab returns:
python research/discovery_uncertainty_niches/analyze_functional_return.py
```

Never cite `--simulate` or notes containing `SIMULATED` as F3/F4.
