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
    "[layer=] [vst_flavor=v2] [residual_type=pearson] [min_cells=5] [n_cells=]"
  ))
}

input_path <- args[[1]]
output_path <- args[[2]]
options <- list(
  layer = "", vst_flavor = "v2", residual_type = "pearson",
  min_cells = 5L, n_cells = ""
)
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
options$n_cells <- if (nzchar(options$n_cells)) as.integer(options$n_cells) else NULL

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
stored_counts <- if (inherits(counts, "sparseMatrix")) counts@x else as.vector(counts)
if (length(stored_counts) == 0 || !any(stored_counts > 0)) {
  stop("SCTransform requires at least one positive count")
}
if (any(abs(stored_counts - round(stored_counts)) > 1e-6)) {
  stop("SCTransform requires integer-like raw UMI counts")
}

vst_args <- list(
  umi = counts,
  vst.flavor = options$vst_flavor,
  residual_type = options$residual_type,
  min_cells = options$min_cells,
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  verbosity = 0
)
if (!is.null(options$n_cells)) {
  vst_args$n_cells <- options$n_cells
}
fit <- do.call(sctransform::vst, vst_args)

residuals <- matrix(
  0,
  nrow = nrow(counts),
  ncol = ncol(counts),
  dimnames = dimnames(counts)
)
modeled_genes <- intersect(rownames(counts), rownames(fit$y))
residuals[modeled_genes, ] <- as.matrix(fit$y[modeled_genes, , drop = FALSE])
# Always preserve the exact matrix used for fitting, even when it came from a named layer.
adata$layers[["counts"]] <- source
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
adata$uns[["sctransform_n_modeled_genes"]] <- length(modeled_genes)
adata$uns[["sctransform_min_cells"]] <- options$min_cells
if (!is.null(options$n_cells)) {
  adata$uns[["sctransform_n_cells"]] <- options$n_cells
}
write_h5ad(adata, output_path)
