#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(anndata)
  library(Banksy)
  library(SpatialExperiment)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(paste(
    "Usage: histoweave-banksy.R <input.h5ad> <output.h5ad>",
    "[layer=] [lambda=0.8] [k_geom=15] [k_neighbors=50] [npcs=20] [algorithm=leiden]",
    "[resolution=0.8] [n_domains=5] [seed=0]"
  ))
}

input_path <- args[[1]]
output_path <- args[[2]]
options <- list(
  layer = "", lambda = 0.8, k_geom = 15L, k_neighbors = 50L,
  npcs = 20L, algorithm = "leiden", resolution = 0.8,
  n_domains = 5L, seed = 0L
)
if (length(args) > 2) {
  for (arg in args[3:length(args)]) {
    parts <- strsplit(arg, "=", fixed = TRUE)[[1]]
    if (length(parts) == 2 && parts[[1]] %in% names(options)) {
      options[[parts[[1]]]] <- parts[[2]]
    }
  }
}
options$lambda <- as.numeric(options$lambda)
options$k_geom <- as.integer(options$k_geom)
options$k_neighbors <- as.integer(options$k_neighbors)
options$npcs <- as.integer(options$npcs)
options$resolution <- as.numeric(options$resolution)
options$n_domains <- as.integer(options$n_domains)
options$seed <- as.integer(options$seed)

adata <- read_h5ad(input_path)
if (!("spatial" %in% names(adata$obsm))) {
  stop("AnnData input is missing obsm['spatial']")
}
coordinates <- as.matrix(adata$obsm[["spatial"]])
if (ncol(coordinates) != 2) {
  stop("BANKSY currently requires exactly two spatial coordinate columns")
}
colnames(coordinates) <- c("x", "y")

# AnnData is observations x genes; SpatialExperiment assays are genes x observations.
source <- if (nzchar(options$layer)) adata$layers[[options$layer]] else adata$X
if (is.null(source)) {
  stop(sprintf("AnnData layer '%s' does not exist", options$layer))
}
counts <- Matrix::t(source)
if (any(!is.finite(counts)) || any(counts < 0)) {
  stop("BANKSY input counts must be finite and non-negative")
}
stored_counts <- if (inherits(counts, "sparseMatrix")) counts@x else as.vector(counts)
if (length(stored_counts) > 0 && any(abs(stored_counts - round(stored_counts)) > 1e-6)) {
  stop("BANKSY requires integer-like raw counts")
}
rownames(counts) <- as.character(adata$var_names)
colnames(counts) <- as.character(adata$obs_names)
rownames(coordinates) <- as.character(adata$obs_names)
spe <- SpatialExperiment(
  assays = list(counts = counts),
  spatialCoords = coordinates
)

# BANKSY expects normalized expression, not raw library-size-confounded counts.
spe <- scuttle::computeLibraryFactors(spe)
spe <- scuttle::logNormCounts(spe)

effective_k_geom <- min(options$k_geom, ncol(spe) - 1L)
spe <- Banksy::computeBanksy(
  spe, assay_name = "logcounts", M = 1L, k_geom = effective_k_geom
)
spe <- Banksy::runBanksyPCA(
  spe, assay_name = "logcounts", M = 1L, lambda = options$lambda,
  npcs = min(options$npcs, nrow(spe) - 1L, ncol(spe) - 1L)
)
effective_k_neighbors <- min(options$k_neighbors, ncol(spe) - 1L)
spe <- Banksy::clusterBanksy(
  spe, assay_name = "logcounts", M = 1L, lambda = options$lambda,
  algo = options$algorithm, k_neighbors = effective_k_neighbors,
  resolution = options$resolution,
  kmeans.centers = options$n_domains, mclust.G = options$n_domains,
  seed = options$seed
)

cluster_names <- Banksy::clusterNames(spe)
if (length(cluster_names) == 0) {
  stop("BANKSY did not produce a cluster-label column")
}
domain <- as.character(SummarizedExperiment::colData(spe)[[tail(cluster_names, 1)]])
adata$obs[["domain"]] <- domain
adata$uns[["banksy_lambda"]] <- options$lambda
adata$uns[["banksy_algorithm"]] <- options$algorithm
adata$uns[["banksy_k_geom"]] <- effective_k_geom
adata$uns[["banksy_k_neighbors"]] <- effective_k_neighbors
write_h5ad(adata, output_path)
