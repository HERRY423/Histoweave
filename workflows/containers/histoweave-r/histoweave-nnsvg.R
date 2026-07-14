#!/usr/bin/env Rscript

# HistoWeave nnSVG wrapper.
#
# Reads an AnnData .h5ad written by the HistoWeave R bridge, fits nnSVG's
# nearest-neighbour Gaussian process per gene, and writes the per-gene ranking
# (rank / LR statistic / adjusted p-value) back into var, plus the top-N gene
# names into uns['nnsvg_top_genes'].

suppressPackageStartupMessages({
  library(anndata)
  library(nnSVG)
  library(SpatialExperiment)
  library(scuttle)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(paste(
    "Usage: histoweave-nnsvg.R <input.h5ad> <output.h5ad>",
    "[n_top=50] [n_neighbors=10] [order=AMMD] [n_threads=1]",
    "[assay_name=logcounts] [seed=0]"
  ))
}

input_path <- args[[1]]
output_path <- args[[2]]
options <- list(
  n_top = 50L, n_neighbors = 10L, order = "AMMD",
  n_threads = 1L, assay_name = "logcounts", seed = 0L
)
if (length(args) > 2) {
  for (arg in args[3:length(args)]) {
    parts <- strsplit(arg, "=", fixed = TRUE)[[1]]
    if (length(parts) == 2 && parts[[1]] %in% names(options)) {
      options[[parts[[1]]]] <- parts[[2]]
    }
  }
}
options$n_top <- as.integer(options$n_top)
options$n_neighbors <- as.integer(options$n_neighbors)
options$n_threads <- as.integer(options$n_threads)
options$seed <- as.integer(options$seed)

set.seed(options$seed)

adata <- read_h5ad(input_path)
if (!("spatial" %in% names(adata$obsm))) {
  stop("AnnData input is missing obsm['spatial']")
}
coordinates <- as.matrix(adata$obsm[["spatial"]])
if (ncol(coordinates) != 2) {
  stop("nnSVG requires exactly two spatial coordinate columns")
}
colnames(coordinates) <- c("x", "y")

# AnnData is observations x genes; SpatialExperiment assays are genes x observations.
mat <- Matrix::t(adata$X)
rownames(mat) <- as.character(adata$var_names)
colnames(mat) <- as.character(adata$obs_names)
rownames(coordinates) <- as.character(adata$obs_names)

# nnSVG expects a logcounts assay. If the requested assay is 'logcounts' but the
# incoming matrix looks like raw counts, normalise; otherwise treat X as already
# transformed and store it under the requested assay name.
assay_name <- options$assay_name
is_countlike <- all(mat@x == floor(mat@x)) && min(mat) >= 0
spe <- SpatialExperiment(
  assays = setNames(list(mat), if (is_countlike) "counts" else assay_name),
  spatialCoords = coordinates
)
if (is_countlike && assay_name == "logcounts") {
  spe <- scuttle::logNormCounts(spe)
} else if (!(assay_name %in% assayNames(spe))) {
  # ensure the requested assay exists
  assay(spe, assay_name) <- mat
}

# Filter out genes with zero expression (nnSVG requires non-degenerate genes).
keep <- rowSums(as.matrix(assay(spe, assay_name))) > 0
spe <- spe[keep, ]

spe <- nnSVG::nnSVG(
  spe,
  assay_name = assay_name,
  n_neighbors = options$n_neighbors,
  order = options$order,
  n_threads = options$n_threads
)

rd <- SummarizedExperiment::rowData(spe)
res <- data.frame(
  gene = rownames(spe),
  nnsvg_rank = as.integer(rd$rank),
  nnsvg_LR_stat = as.numeric(rd$LR_stat),
  nnsvg_padj = as.numeric(rd$padj),
  stringsAsFactors = FALSE
)

# Align results back to the full var index (genes filtered out get NA).
var_df <- adata$var
all_genes <- as.character(adata$var_names)
idx <- match(all_genes, res$gene)
var_df[["nnsvg_rank"]] <- res$nnsvg_rank[idx]
var_df[["nnsvg_LR_stat"]] <- res$nnsvg_LR_stat[idx]
var_df[["nnsvg_padj"]] <- res$nnsvg_padj[idx]
adata$var <- var_df

ranked <- res[order(res$nnsvg_rank), ]
top_genes <- head(ranked$gene, options$n_top)
adata$uns[["nnsvg_top_genes"]] <- top_genes
adata$uns[["nnsvg_assay"]] <- assay_name

write_h5ad(adata, output_path)
