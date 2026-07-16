#!/usr/bin/env Rscript
# Official BayesSpace bridge for the DLPFC benchmark.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("usage: run_bayesspace.R <in.h5ad> <out.csv> <seed> <q> <nrep> <r_lib>")
}
in_h5 <- args[[1]]
out_csv <- args[[2]]
seed <- as.integer(args[[3]])
q <- as.integer(args[[4]])
nrep <- as.integer(args[[5]])
r_lib <- args[[6]]
.libPaths(c(r_lib, .libPaths()))

suppressPackageStartupMessages({
  library(BayesSpace)
  library(SingleCellExperiment)
  library(zellkonverter)
})

sce <- zellkonverter::readH5AD(in_h5, X_name = "X", reader = "python")
if (!"counts" %in% assayNames(sce)) {
  assay(sce, "counts") <- assay(sce, 1)
}
required <- c("array_row", "array_col")
if (!all(required %in% colnames(colData(sce)))) {
  stop("input h5ad is missing array_row/array_col")
}

n_pcs <- min(15L, q + 3L, nrow(sce) - 1L, ncol(sce) - 1L)
if (n_pcs < 2L) {
  stop("BayesSpace requires at least two principal components")
}
set.seed(seed)
sce <- spatialPreprocess(
  sce,
  platform = "Visium",
  n.PCs = n_pcs,
  n.HVGs = min(2000L, nrow(sce)),
  log.normalize = TRUE
)
sce <- spatialCluster(
  sce,
  q = q,
  platform = "Visium",
  d = n_pcs,
  init.method = "mclust",
  model = "t",
  gamma = 2,
  nrep = nrep,
  burn.in = max(100L, as.integer(nrep * 0.1)),
  save.chain = FALSE
)
labels <- data.frame(
  spot_id = colnames(sce),
  label = as.integer(colData(sce)$spatial.cluster)
)
write.csv(labels, out_csv, row.names = FALSE, quote = FALSE)
