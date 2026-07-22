# Fine immune niches are systematically under-resolved by spatial-domain clustering

**Exploratory discovery — HistoWeave Phase B**
Sample: 10x Visium FFPE human breast cancer (public, CC-BY-4.0), 2,518 spots × 17,943 genes.
Framing: exploratory niche-recovery analysis on a single public sample. This is a **methodological / niche-recovery observation**, not a validated clinical finding. Every claim below is limited to what the markers and spatial statistics on this one sample support.

---

## TL;DR

1. The tertiary-lymphoid-structure (**TLS**) immune niche in this tumor is **real and spatially organized** — the canonical 13-gene B/T/chemokine signature has Moran's I = **0.665**, and the 99 TLS-foci spots form spatially contiguous clusters (contiguity = **0.727**).
2. A naive **multi-method consensus did *not* recover the niche better than single methods** — in fact it did *worse* (best-domain F1 **0.130** vs best single method **0.416**). This negative result is reported as-is.
3. The reason is the genuine finding: **the TLS niche (3.9% of spots) is far smaller than any domain produced at standard tissue-architecture granularity (k ≈ 7–8 → recovered domain ≈ 11% of tissue, ~3× too large).** No spatial-domain method — single or consensus — cleanly isolates it. Best-domain F1 only approaches niche-matched precision at k ≈ 30–40, where domain size finally equals the ~99-spot niche.
4. **Spatial smoothing actively hurts.** Adding spatial coordinates to features (sw = 0.3) *lowers* niche recovery at typical k (F1 0.11–0.14 for k ≤ 15) because smoothing merges the small immune niche into large contiguous tissue blocks.
5. Wherever the niche *is* captured, it is biologically validated: within the recovered domain, B markers are enriched **3.13×**, T markers **2.60×**, chemokines **2.33×**, and an *independent* 14-gene immune signature **1.69×**; TLS score inside vs outside the domain differs at Mann–Whitney U **p = 8.2 × 10⁻¹³⁹**.

**Implication for HistoWeave:** generic spatial-domain clustering at architecture-level resolution is the wrong tool for fine immune niches. This motivates a *signature-targeted* niche-detection module (score → threshold → spatial test) as a first-class HistoWeave capability, orthogonal to the domain-segmentation methods it already orchestrates.

---

## 1. Target definition (independent of any clustering)

The TLS niche is defined **without** reference to any clustering result, to avoid circularity:

- **TLS signature (13 genes):** B cell = MS4A1, CD79A, CD79B, CD19, CR2, LTB; T cell = CD3D, CD3E, CD8A; chemokines = CXCL13, CCL19, CCL21, SELL.
- **TLS foci** = spots co-high in **both** a B-cell axis **and** a T-cell axis (each > 90th percentile) → **99 spots (3.9%)**.
- **Spatial reality check:** Moran's I of the TLS signature (k = 6 neighbors) = **0.665** (strongly autocorrelated); foci spatial contiguity = **0.727** (most foci have ≥1 foci neighbor). The niche is not scattered noise.

## 2. Single methods vs consensus (recovery of the foci)

Best single domain that overlaps the TLS foci (max F1 over that method's domains):

| method | domains | best-domain F1 | Jaccard | domains for 80% of foci |
|---|---|---|---|---|
| tenx_graphclust (10x reference) | 8 | **0.416** | 0.263 | 2 |
| kmeans | 8 | 0.393 | 0.245 | 1 |
| spectral | 8 | 0.354 | 0.215 | 2 |
| gmm | 8 | 0.267 | 0.154 | 2 |
| agglomerative | 8 | 0.106 | 0.056 | 2 |
| **CONSENSUS (co-association)** | 8 | **0.130** | 0.070 | 1 |

The consensus (co-association matrix → average-linkage agglomerative at k = median domains = 8) is **worse than every real single method**. Naive consensus averages away the small niche rather than sharpening it. We deliberately did **not** tune the consensus until it "won" — that would be p-hacking.

## 3. Why single methods (and consensus) fail: a resolution mismatch

Sweeping the number of clusters k and measuring best-domain recovery of the 99-spot niche:

| k | F1 (expression) | precision | recall | recovered-domain size |
|---|---|---|---|---|
| 7 | 0.422 | 0.280 | 0.859 | 304 |
| **8** | **0.434** | **0.296** | **0.818** | **274** |
| 15 | 0.466 | 0.335 | 0.768 | 227 |
| 25 | 0.482 | 0.343 | 0.808 | 233 |
| 30 | 0.496 | 0.386 | 0.697 | 179 |
| 40 | 0.511 | 0.553 | 0.475 | 85 |

At the standard granularity (k = 7–8 used by essentially all spatial-domain methods), **recall is high (~0.82) but precision is only ~0.30** — the niche is *embedded inside* a domain ~3× its size, not isolated. Precision (and F1) climb only as k rises toward ~40, where the recovered domain finally shrinks to ~99 spots (≈ niche size). No resolution cleanly isolates the niche (F1 max ≈ 0.51).

**Spatial weighting makes it worse.** Concatenating spatially-weighted coordinates (sw = 0.3) collapses F1 to 0.11–0.14 for k ≤ 15 (recovered "domain" balloons to 1,000–1,600 spots) — smoothing fuses the small immune niche into large tissue blocks. It only recovers at k ≥ 30.

## 4. Biological validation of the recovered niche

Within the best-overlapping recovered domain vs the rest of the tissue:

| panel | fold enrichment (inside/outside) |
|---|---|
| B-cell markers | **3.13×** |
| T-cell markers | **2.60×** |
| chemokines | **2.33×** |
| independent immune signature (14 genes) | **1.69×** |

TLS score inside vs outside the recovered domain: Mann–Whitney U, one-sided, **p = 8.2 × 10⁻¹³⁹**. The recovered region is unambiguously an immune/TLS niche — the problem is purely one of *resolution*, not signal.

## 5. Honesty statement & limitations

- **Single sample, exploratory.** One public 10x breast-cancer section. No claim of generalization or clinical utility.
- **Reference labels are transcriptomic, not pathologist-annotated.** The intended upgrade is expert TLS annotations (Zenodo record 18917100, Leij & Swarbrick / Garvan); those file downloads were throttled from this environment, so validation here is marker/spatial-statistics based. If/when annotations are retrieved, the same pipeline re-runs against pathologist ground truth.
- **The consensus result is a genuine negative.** We report that naive consensus underperforms; we did not search for a consensus variant that beats single methods.
- **The positive finding is methodological:** fine niches need finer-than-architecture resolution (or a targeted score-based detector), and spatial smoothing is counterproductive for them.

## 6. Second independent dataset: negative transport result

The locked per-unit endpoint was applied without retuning to the official 10x
Xenium Prime reactive human lymph-node dataset. It did **not** replicate:
Moran's I fell from 0.665 to 0.190, 29 co-high cells had contiguity 0, none
overlapped the 50 retained pathology germinal-center cells (F1 0), and a fixed
k=20 B/T neighbourhood co-localisation sensitivity had AUROC 0.364. The
cell-resolved assay separates B and T signal across observations, while the GC
reference is sparse after the documented stratified subsample. This negative
result does not disprove the within-sample breast observation, but it prevents
a general TLS claim and makes an assay-aware endpoint a prerequisite for
further validation. See
`second_dataset_xenium_lymph/REPORT_tls_second_dataset.md`.

## Reproducibility

- Method labels: `research/phaseB_tls_consensus/run_bc_method_labels.py`
- Discovery core: `research/phaseB_tls_consensus/analyze_tls_consensus.py`
- Resolution sweep: recorded in the execution notebook; outputs `resolution_sensitivity.csv`.
- Outputs: `recovery_metrics.csv`, `discovery_summary.json`, `resolution_sensitivity.csv`, `per_spot.parquet`.
- Figures: `figures/fig1_tls_niche_spatial.svg`, `figures/fig2_resolution_sensitivity.svg`, `figures/fig3_marker_validation.svg`.
