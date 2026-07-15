#!/usr/bin/env Rscript
# --------------------------------------------------------------------------
# nnSVG driver for the histoweave 5x15 spatial-aware benchmark.
#
# Reads an h5ad written by the Python adapter, ranks genes with nnSVG, and
# writes the ranked list to CSV.  Uses `zellkonverter` to avoid re-encoding
# assumptions about h5ad layouts across scanpy releases.
#
# Usage:
#   Rscript run_nnsvg.R <in.h5ad> <out.csv> <n_threads> <r_lib_path>
# --------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: run_nnsvg.R <in.h5ad> <out.csv> <n_threads> <r_lib_path>")
}
in_h5    <- args[[1]]
out_csv  <- args[[2]]
n_thread <- as.integer(args[[3]])
r_lib    <- args[[4]]

.libPaths(c(r_lib, .libPaths()))

suppressPackageStartupMessages({
  library(SpatialExperiment)
  library(SingleCellExperiment)
  library(nnSVG)
  library(BiocParallel)
  library(zellkonverter)
})

message(sprintf("[nnSVG] loading %s", in_h5))
sce <- zellkonverter::readH5AD(in_h5, X_name = "X", reader = "python")
# Ensure counts assay exists under the expected name
if (!"counts" %in% assayNames(sce)) {
  assay(sce, "counts") <- assay(sce, 1)
}

coords <- reducedDim(sce, "spatial")
if (is.null(coords)) {
  # zellkonverter puts obsm["spatial"] under reducedDims by default; sanity-check
  if ("spatial" %in% names(reducedDims(sce))) {
    coords <- reducedDim(sce, "spatial")
  } else {
    stop("spatial coords not found in reducedDim(sce, 'spatial')")
  }
}

# Promote to SpatialExperiment (needed by nnSVG)
spe <- SpatialExperiment(
  assays = list(counts = assay(sce, "counts")),
  colData = colData(sce),
  spatialCoords = as.matrix(coords)
)

# Basic filtering; nnSVG is expensive so keep it lean
keep_rows <- rowSums(counts(spe)) > 0
spe <- spe[keep_rows, ]
message(sprintf("[nnSVG] genes after filter: %d / cells: %d",
                nrow(spe), ncol(spe)))

# Log-normalise counts as required by nnSVG
suppressPackageStartupMessages(library(scran))
qclust <- scran::quickCluster(spe)
spe <- scran::computeSumFactors(spe, clusters = qclust)
suppressPackageStartupMessages(library(scuttle))
spe <- scuttle::logNormCounts(spe)

BPPARAM <- MulticoreParam(workers = max(1L, n_thread))
message(sprintf("[nnSVG] running nnSVG with %d workers", n_thread))

set.seed(42)
spe <- nnSVG(spe, X = NULL, assay_name = "logcounts",
             n_neighbors = 10, BPPARAM = BPPARAM,
             verbose = FALSE)

res <- as.data.frame(rowData(spe))
# nnSVG stores results in rowData; primary rank field is `rank`
if (!"rank" %in% colnames(res)) {
  # older nnSVG versions use LR_stat / padj
  if ("LR_stat" %in% colnames(res)) {
    res$rank <- rank(-res$LR_stat, ties.method = "first")
  } else if ("padj" %in% colnames(res)) {
    res$rank <- rank(res$padj, ties.method = "first")
  } else {
    stop("nnSVG rowData missing rank / LR_stat / padj columns")
  }
}
res$gene <- rownames(res)
out <- res[order(res$rank), c("gene", "rank")]

write.csv(out, out_csv, row.names = FALSE)
message(sprintf("[nnSVG] wrote %d ranks -> %s", nrow(out), out_csv))
