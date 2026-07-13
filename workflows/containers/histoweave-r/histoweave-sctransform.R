#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(anndata)
  library(Matrix)
  library(sctransform)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop(paste(
    "Usage: histoweave-sctransform.R <input.h5ad> <output.h5ad>",
    "[layer=] [vst_flavor=v2] [residual_type=pearson] [min_cells=5]"
  ))
}

input_path <- args[[1]]
output_path <- args[[2]]
options <- list(layer = "", vst_flavor = "v2", residual_type = "pearson", min_cells = 5L)
if (length(args) > 2) {
  for (arg in args[3:length(args)]) {
    parts <- strsplit(arg, "=", fixed = TRUE)[[1]]
    if (length(parts) == 1 && startsWith(arg, "layer=")) {
      options$layer <- ""
    } else if (length(parts) == 2 && parts[[1]] %in% names(options)) {
      options[[parts[[1]]]] <- parts[[2]]
    }
  }
}
options$min_cells <- as.integer(options$min_cells)

adata <- read_h5ad(input_path)
source <- if (nzchar(options$layer)) adata$layers[[options$layer]] else adata$X
if (is.null(source)) {
  stop(sprintf("AnnData layer '%s' does not exist", options$layer))
}
counts <- Matrix::t(source)
rownames(counts) <- as.character(adata$var_names)
colnames(counts) <- as.character(adata$obs_names)
if (any(counts < 0) || any(!is.finite(counts))) {
  stop("SCTransform input counts must be finite and non-negative")
}

fit <- sctransform::vst(
  umi = counts,
  vst.flavor = options$vst_flavor,
  residual_type = options$residual_type,
  min_cells = options$min_cells,
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  return_only_var_genes = FALSE,
  verbosity = 0
)

residuals <- matrix(
  0,
  nrow = nrow(counts),
  ncol = ncol(counts),
  dimnames = dimnames(counts)
)
modeled_genes <- intersect(rownames(counts), rownames(fit$y))
residuals[modeled_genes, ] <- as.matrix(fit$y[modeled_genes, , drop = FALSE])
if (!("counts" %in% names(adata$layers))) {
  adata$layers[["counts"]] <- adata$X
}
adata$X <- Matrix::t(residuals)
adata$var[["sctransform_modeled"]] <- rownames(counts) %in% modeled_genes
if ("residual_variance" %in% colnames(fit$gene_attr)) {
  residual_variance <- rep(NA_real_, nrow(counts))
  names(residual_variance) <- rownames(counts)
  residual_variance[rownames(fit$gene_attr)] <- fit$gene_attr$residual_variance
  adata$var[["sctransform_residual_variance"]] <- unname(residual_variance)
}
adata$uns[["sctransform_normalized"]] <- TRUE
adata$uns[["sctransform_vst_flavor"]] <- options$vst_flavor
adata$uns[["sctransform_residual_type"]] <- options$residual_type
write_h5ad(adata, output_path)
