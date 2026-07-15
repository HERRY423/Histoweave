#!/usr/bin/env Rscript

# Run a deliberately small but genuine sctransform::vst validation on raw
# integer counts.  Residuals are written as float32 column-major binary so the
# Python orchestrator can attach them to an H5AD without an R HDF5 dependency.

suppressPackageStartupMessages({
  library(Matrix)
  library(R.utils)
  library(sctransform)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: run_sct_pilot.R <pilot_directory> <output_directory>")
}

pilot_dir <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

matrix_path <- file.path(pilot_dir, "pilot_counts_genes_by_beads.mtx.gz")
gene_path <- file.path(pilot_dir, "pilot_genes.txt")
barcode_path <- file.path(pilot_dir, "pilot_barcodes.txt")
if (!file.exists(matrix_path) || !file.exists(gene_path) || !file.exists(barcode_path)) {
  stop("Pilot directory is missing matrix, gene, or barcode files")
}

counts <- Matrix::readMM(gzfile(matrix_path, open = "rt"))
counts <- as(counts, "dgCMatrix")
genes <- readLines(gene_path, warn = FALSE, encoding = "UTF-8")
barcodes <- readLines(barcode_path, warn = FALSE, encoding = "UTF-8")
if (!identical(dim(counts), c(length(genes), length(barcodes)))) {
  stop("Pilot matrix dimensions disagree with gene/barcode files")
}
rownames(counts) <- genes
colnames(counts) <- barcodes

stored <- counts@x
if (!length(stored) || any(!is.finite(stored)) || any(stored < 0) ||
    any(abs(stored - round(stored)) > 1e-8)) {
  stop("SCTransform pilot input is not finite non-negative integer-like raw counts")
}

fit_n_cells <- min(3000L, ncol(counts))
set.seed(20011508L)
fit <- sctransform::vst(
  umi = counts,
  vst.flavor = "v2",
  residual_type = "pearson",
  min_cells = 5L,
  n_cells = fit_n_cells,
  return_cell_attr = TRUE,
  return_gene_attr = TRUE,
  verbosity = 0
)

modeled_genes <- rownames(fit$y)
if (is.null(modeled_genes) || !length(modeled_genes)) {
  stop("sctransform returned no modeled genes")
}
residuals <- as.matrix(fit$y)
if (!all(is.finite(residuals))) {
  stop("sctransform returned non-finite residuals")
}

# Binary layout is genes x beads in R/Fortran column-major order.
binary_path <- file.path(output_dir, "sct_residuals_float32.bin")
connection <- file(binary_path, open = "wb")
writeBin(as.numeric(residuals), connection, size = 4L, endian = "little")
close(connection)
writeLines(modeled_genes, file.path(output_dir, "sct_modeled_genes.txt"), useBytes = TRUE)
writeLines(colnames(residuals), file.path(output_dir, "sct_barcodes.txt"), useBytes = TRUE)

gene_attr <- data.frame(gene = rownames(fit$gene_attr), fit$gene_attr, check.names = FALSE)
write.csv(gene_attr, file.path(output_dir, "sct_gene_attributes.csv"), row.names = FALSE)
cell_attr <- data.frame(barcode = rownames(fit$cell_attr), fit$cell_attr, check.names = FALSE)
write.csv(cell_attr, file.path(output_dir, "sct_cell_attributes.csv"), row.names = FALSE)

metadata <- c(
  sprintf("n_input_genes=%d", nrow(counts)),
  sprintf("n_input_beads=%d", ncol(counts)),
  sprintf("n_modeled_genes=%d", nrow(residuals)),
  sprintf("n_output_beads=%d", ncol(residuals)),
  sprintf("fit_n_cells=%d", fit_n_cells),
  "vst_flavor=v2",
  "residual_type=pearson",
  "min_cells=5",
  sprintf("sctransform_version=%s", as.character(packageVersion("sctransform")))
)
writeLines(metadata, file.path(output_dir, "sct_metadata.txt"), useBytes = TRUE)

cat(sprintf("SCT_OK genes=%d beads=%d fit_n_cells=%d\n", nrow(residuals), ncol(residuals), fit_n_cells))
cat(sprintf("finite=true residual_min=%.6g residual_max=%.6g\n", min(residuals), max(residuals)))
