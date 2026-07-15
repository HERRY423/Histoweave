#!/usr/bin/env Rscript

# Export the original Slide-seqV2 RData without ever touching the normalized
# Squidpy expression matrix.  The exported matrix stays genes x beads until
# the Python builder transposes it into AnnData's beads x genes convention.

suppressPackageStartupMessages({
  library(Matrix)
  library(R.utils)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("Usage: export_raw_rdata.R <input.RData> <output_directory>")
}

input_path <- normalizePath(args[[1]], mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

loaded <- new.env(parent = emptyenv())
load(input_path, envir = loaded)
if (!exists("countmat", envir = loaded, inherits = FALSE) ||
    !exists("location", envir = loaded, inherits = FALSE)) {
  stop("RData must contain objects named 'countmat' and 'location'")
}

counts <- get("countmat", envir = loaded, inherits = FALSE)
location <- get("location", envir = loaded, inherits = FALSE)
if (!inherits(counts, "sparseMatrix")) {
  stop("countmat must be a Matrix sparseMatrix")
}
counts <- as(counts, "dgCMatrix")
if (!is.matrix(location) || ncol(location) != 2L) {
  stop("location must be a two-column matrix")
}
if (is.null(rownames(counts)) || is.null(colnames(counts))) {
  stop("countmat must have gene row names and bead barcode column names")
}
if (anyDuplicated(rownames(counts)) || anyDuplicated(colnames(counts))) {
  stop("Gene names and bead barcodes must be unique")
}
if (is.null(rownames(location)) || anyDuplicated(rownames(location))) {
  stop("location must have unique barcode row names")
}
if (ncol(counts) != nrow(location)) {
  stop("countmat columns and location rows have different lengths")
}

location_index <- match(colnames(counts), rownames(location))
if (anyNA(location_index)) {
  stop("At least one countmat barcode is absent from location")
}
location <- location[location_index, , drop = FALSE]
if (!identical(colnames(counts), rownames(location))) {
  stop("Internal error while ordering location rows to countmat barcodes")
}
colnames(location) <- c("x", "y")

stored <- counts@x
if (!length(stored) || !any(stored > 0)) {
  stop("countmat contains no positive counts")
}
if (any(!is.finite(stored)) || any(stored < 0)) {
  stop("countmat must contain finite, non-negative counts")
}
if (any(abs(stored - round(stored)) > 1e-8)) {
  stop("countmat violates the integer-like raw UMI count contract")
}
if (max(stored) > .Machine$integer.max) {
  stop("countmat contains values larger than signed int32")
}

counts@x <- as.numeric(round(counts@x))
matrix_path <- file.path(output_dir, "counts_genes_by_beads.mtx")
matrix_gz_path <- paste0(matrix_path, ".gz")
Matrix::writeMM(counts, matrix_path)
R.utils::gzip(
  matrix_path,
  destname = matrix_gz_path,
  overwrite = TRUE,
  remove = TRUE
)

writeLines(rownames(counts), file.path(output_dir, "genes.txt"), useBytes = TRUE)
writeLines(colnames(counts), file.path(output_dir, "barcodes.txt"), useBytes = TRUE)
coordinate_table <- data.frame(
  barcode = colnames(counts),
  x = as.numeric(location[, 1]),
  y = as.numeric(location[, 2]),
  check.names = FALSE
)
write.table(
  coordinate_table,
  file.path(output_dir, "coordinates.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

cat(sprintf("EXPORT_OK input=%s\n", input_path))
cat(sprintf("shape_genes_by_beads=%dx%d\n", nrow(counts), ncol(counts)))
cat(sprintf("nnz=%d\n", length(counts@x)))
cat(sprintf("stored_min=%.0f stored_max=%.0f\n", min(counts@x), max(counts@x)))
cat(sprintf("integer_contract=true barcode_coordinate_contract=true\n"))
