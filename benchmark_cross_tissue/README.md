# Cross-tissue real-data validation (7 datasets × 19 methods)

This benchmark adds two non-DLPFC tissues to the five-slice DLPFC benchmark:

| Dataset | Platform | Independent `domain_truth` | Evaluation bundle |
|---|---|---|---|
| Human lymph node | Xenium Prime | 10x pathology annotation polygons | ≤15,000 pathology-labelled cells |
| Mouse brain (Allen Brain Atlas) | MERFISH | Allen CCF anatomical division/region | ≤60,000 cells per input section; benchmark samples ≤15,000 |

Cell-type predictions are not accepted as primary spatial-domain truth. The lymph-node
preparer excludes cells outside pathology polygons and conflicting overlaps. The Allen
preparer requires an anatomical column by default; its older cell-class collapse is
available only through the explicit `--allow-cell-class-fallback` sensitivity option.

## Data preparation

Download the official Xenium Prime Human Lymph Node matrix, cells table, and pathology
annotation GeoJSON from the [10x preview dataset](https://www.10xgenomics.com/datasets/preview-data-xenium-prime-gene-expression), then run:

```powershell
python benchmark_cross_tissue/prepare_human_lymph_node.py `
  --matrix cell_feature_matrix.h5 `
  --metadata cells.csv.gz `
  --pathology-geojson pathology_annotations.geojson
```

Obtain Allen Brain Cell Atlas MERFISH section h5ad files containing `obsm['spatial']`
and an anatomical CCF column. The preparer auto-detects common anatomical columns, or
one can be named explicitly:

```powershell
python benchmark_cross_tissue/prepare_allen_mouse_brain.py `
  --section section_1.h5ad --section section_2.h5ad --section section_3.h5ad `
  --region-column parcellation_division
```

Allen sources: [ABC Atlas guide](https://alleninstitute.org/education-resources/database-guide-abc-atlas)
and [Allen data and technology](https://alleninstitute.org/brain-science/data-technology).

## Run the comparison

```powershell
python benchmark_cross_tissue/experiment_7x19.py
```

Outputs include the long-form results, 7×19 mean/std ARI matrices, dataset truth
manifest, and checksummed run manifest. BayesSpace is recorded as `unsupported` for
Xenium/MERFISH because these assays do not provide Visium array row/column coordinates.
