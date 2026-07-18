# External validation (5 datasets × 15 methods)

A **cross-study generalization** benchmark for HistoWeave's method-recommendation
engine. The within-study 5×10 / 5×15 DLPFC benchmarks showed the recommender
cannot beat the trivial "always pick kmeans" global baseline (top-1 = 0, mean
selection regret 0.075 vs global-best 0.055) because all 5 slices come from one
study with near-identical feature profiles. This benchmark adds **5 external
validation datasets** spanning 4 platforms, 2 organisms, 4 tissues, and 4
independent studies, all with **strict region ground truth** (anatomical /
pathology / manual — never cell-type predictions), then re-tests the
recommender's leave-one-dataset-out generalization.

## Datasets

| Dataset | Platform | Organism / Tissue | Ground truth | Source |
|---|---|---|---|---|
| `visium_hd_crc` | Visium HD | Human colorectal cancer (FFPE) | Pathologist regions (Neoplasm / Non-neoplastic Epithelium / Connective Tissue / Smooth Muscle) | 10x Visium HD CRC + [Zenodo 11077886](https://zenodo.org/records/11077886) (CC0) |
| `xenium_lung_cancer` | Xenium (IO panel) | Human lung adenocarcinoma (FFPE) | Pathology GeoJSON polygons | [10x FFPE Human Lung Cancer](https://www.10xgenomics.com/datasets/ffpe-human-lung-cancer-data-with-human-immuno-oncology-profiling-panel-and-custom-add-on-1-standard) |
| `xenium_ovarian_cancer` | Xenium Prime 5K | Human ovarian cancer (FF) | Pathology GeoJSON (tumor / necrosis / smooth muscle / fallopian tube / ovary) | [10x Xenium Prime Ovarian Cancer](https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-ovarian-cancer) |
| `visium_mouse_brain` | Visium v2 | Mouse brain (coronal, H&E) | 15 Allen-reference anatomical regions | `squidpy.datasets.visium_hne_adata()` |
| `allen_merfish_brain_section` | MERFISH (ABC Atlas) | Mouse whole-brain (single coronal section) | Allen CCFv3 `parcellation_division` | [Allen Brain Cell Atlas](https://alleninstitute.github.io/abc_atlas_access/) (AWS S3 public) |

**Diversity axes:** 4 platforms (Visium HD, Xenium, Visium v2, MERFISH), 2
organisms (human/mouse), 4 tissues (colon, lung, ovary, brain ×2), disease +
normal, 4 independent studies, pathology + anatomical ground truth. None
overlap with the existing repo datasets (DLPFC, breast Xenium, lymph-node
Xenium, hypothalamus MERFISH, hippocampus Slide-seqV2).

### Ground-truth policy

Consistent with the existing `benchmark_cross_tissue` policy, **only
anatomical / pathology / manual region annotations count as `domain_truth`**.
Predicted cell-type clusters are never used as primary spatial-domain truth.
The Xenium preparers exclude cells outside pathology polygons and cells in
conflicting polygon overlaps; the Allen preparer requires an anatomical CCF
column (cell-class fallback is a sensitivity-only flag, never the default).

## Data preparation

### 1. Visium HD CRC (downloads automatically)

```bash
python benchmark_external_validation/prepare_visium_hd_crc.py
```

Downloads the 10x Visium HD CRC binned-outputs tarball + the Zenodo pathologist
annotation CSV, joins by barcode, attaches bin x/y as `obsm['spatial']`. Set
`--matrix-dir` / `--annotation` to use pre-downloaded files.

### 2. Xenium lung + ovarian (supply official 10x bundle files)

Download the Xenium Output Bundle for each dataset from the 10x dataset page,
then:

```bash
python benchmark_external_validation/prepare_xenium_lung_cancer.py \
    --matrix cell_feature_matrix.h5 \
    --metadata cells.csv.gz \
    --pathology-geojson pathology_annotations.geojson

python benchmark_external_validation/prepare_xenium_ovarian_cancer.py \
    --matrix cell_feature_matrix.h5 \
    --metadata cells.csv.gz \
    --pathology-geojson pathology_annotations.geojson
```

### 3. Visium mouse brain (auto-downloads via squidpy)

```bash
pip install "histoweave[scanpy,spatial]"
python benchmark_external_validation/prepare_visium_mouse_brain.py
```

### 4. Allen MERFISH brain section (auto-downloads via AbcProjectCache)

```bash
pip install abc-atlas-access
python benchmark_external_validation/prepare_allen_merfish_brain_section.py
```

Downloads one coronal section from the Allen Brain Cell Atlas MERFISH
whole-brain dataset (AWS S3 public bucket, no account needed). Override the
section with `--section-label` or `--section-index`.

## Run the benchmark + recommender validation

```bash
# 5 datasets × 15 methods × 3 seeds, ARI vs domain_truth, bootstrap CIs
python benchmark_external_validation/experiment_5x_external.py

# Leave-one-dataset-out recommender generalization test
python benchmark_external_validation/recommender_loocv_external.py

# Figures (SVG + PNG)
python benchmark_external_validation/make_figures.py
```

Outputs (in this directory by default, override with `HISTOWEAVE_EXT_OUT`):
`benchmark_long.csv`, `performance_matrix_mean.csv` / `_std.csv`,
`bootstrap_ci.csv`, `benchmark_5x_external.json`, `dataset_manifest.json`,
`recommendation_loocv.csv` / `.json`, `manifest.json`, and `figures/`
(4 figures × SVG + PNG).

## Method list (shared with 7×15)

**10 sklearn baselines:** `agglomerative, birch, bisecting_kmeans, dbscan,
gaussian_mixture, kmeans, mean_shift, minibatch_kmeans, optics, spectral`.

**5 spatial-aware:** `banksy_py, spatialde_kmeans, nnsvg_kmeans,
harmony_kmeans, moran_spectral`.

7 partitional/hierarchical methods receive each dataset's true domain count;
3 density/mode-seeking methods (`dbscan`, `optics`, `mean_shift`) auto-determine
cluster count. Datasets above 15 000 cells are stratified-subsampled per
(dataset, seed) so every method sees the same slice; bootstrap CIs are
refit-free (100 × 80% cell resamples per cell).

## See also

- `5x10_dlpfc_benchmark/` — within-study DLPFC baseline (the benchmark this
  external set is designed to complement).
- `benchmark_crossplatform/` — 7×15 cross-platform landscape (5 DLPFC + Xenium
  breast + MERFISH hypothalamus + Slide-seqV2 hippocampus).
- `benchmark_cross_tissue/` — 7×19 cross-tissue landscape (5 DLPFC + lymph
  node + Allen mouse brain).
- `report_external_validation.md` — narrative report with the
  recommender-generalization claim and honest limitations.
