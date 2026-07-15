#!/usr/bin/env Rscript

# Dependency-light SCTransform runner. Matrix Market input avoids the optional
# R `anndata` package; the model is sctransform::vst v2 on exact UMI counts.

suppressPackageStartupMessages({
  library(Matrix)
  library(sctransform)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8) {
  stop(paste(
    "Usage: run_sct_scores.R <matrix.mtx> <genes.txt> <cells.txt>",
    "<module_spec.tsv> <scores.csv> <coverage.csv> <n_cells> <seed>"
  ))
}

matrix_path <- args[[1]]
genes_path <- args[[2]]
cells_path <- args[[3]]
module_path <- args[[4]]
scores_path <- args[[5]]
coverage_path <- args[[6]]
n_cells <- as.integer(args[[7]])
seed <- as.integer(args[[8]])

umi <- Matrix::readMM(matrix_path)
genes <- readLines(genes_path, warn = FALSE)
cells <- readLines(cells_path, warn = FALSE)
if (nrow(umi) != length(genes) || ncol(umi) != length(cells)) {
  stop("Matrix dimensions do not match gene/cell names")
}
rownames(umi) <- genes
colnames(umi) <- cells
umi <- as(umi, "CsparseMatrix")
stored <- umi@x
if (length(stored) == 0 || any(!is.finite(stored)) || any(stored < 0)) {
  stop("SCTransform input must contain finite, non-negative counts")
}
if (any(abs(stored - round(stored)) > 1e-6)) {
  stop("SCTransform input must be integer-like raw UMI counts")
}

set.seed(seed)
fit <- sctransform::vst(
  umi = umi,
  n_genes = min(2000L, nrow(umi)),
  n_cells = min(n_cells, ncol(umi)),
  vst.flavor = "v2",
  residual_type = "pearson",
  min_cells = 5L,
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  verbosity = 0
)
residuals <- fit$y
spec <- read.delim(module_path, stringsAsFactors = FALSE)
scores <- data.frame(spot_id = cells, stringsAsFactors = FALSE, check.names = FALSE)
coverage <- list()

standardize_columns <- function(values) {
  means <- colMeans(values)
  sds <- apply(values, 2, sd)
  sds[!is.finite(sds) | sds < 1e-10] <- 1
  sweep(sweep(values, 2, means, "-"), 2, sds, "/")
}

standardize_vector <- function(values) {
  value_sd <- sd(values)
  if (!is.finite(value_sd) || value_sd < 1e-10) {
    return(rep(0, length(values)))
  }
  (values - mean(values)) / value_sd
}

for (module in unique(spec$module)) {
  requested <- spec$gene[spec$module == module]
  present <- intersect(requested, rownames(umi))
  used <- intersect(requested, rownames(residuals))
  if (length(used) < 2) {
    scores[[module]] <- rep(NA_real_, length(cells))
  } else {
    values <- t(as.matrix(residuals[used, , drop = FALSE]))
    scores[[module]] <- rowMeans(standardize_columns(values))
  }
  coverage[[length(coverage) + 1]] <- data.frame(
    branch = "SCT",
    section = sub("^([^:]+):.*$", "\\1", cells[[1]]),
    module = module,
    n_requested = length(requested),
    n_present = length(present),
    n_used = length(used),
    genes_used = paste(used, collapse = ";"),
    stringsAsFactors = FALSE
  )
}

gei_components <- c("astro_ion", "oligo_myelin", "vascular_barrier")
component_values <- sapply(gei_components, function(module) {
  standardize_vector(scores[[module]])
})
scores[["GEI"]] <- rowMeans(component_values)
coverage[[length(coverage) + 1]] <- data.frame(
  branch = "SCT",
  section = sub("^([^:]+):.*$", "\\1", cells[[1]]),
  module = "GEI",
  n_requested = sum(spec$module %in% gei_components),
  n_present = sum(spec$gene[spec$module %in% gei_components] %in% rownames(umi)),
  n_used = sum(spec$gene[spec$module %in% gei_components] %in% rownames(residuals)),
  genes_used = "component module z-scores",
  stringsAsFactors = FALSE
)

write.csv(scores, scores_path, row.names = FALSE, quote = TRUE)
write.csv(do.call(rbind, coverage), coverage_path, row.names = FALSE, quote = TRUE)
cat(sprintf(
  "sctransform=%s; cells=%d; genes=%d; modeled=%d; seed=%d\n",
  as.character(packageVersion("sctransform")), ncol(umi), nrow(umi),
  nrow(residuals), seed
))
