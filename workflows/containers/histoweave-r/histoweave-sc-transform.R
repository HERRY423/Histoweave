#!/usr/bin/env Rscript
#
# histoweave-sc-transform.R — R-side log-normalisation for the Python ↔ R bridge.
#
# This script is the reference implementation that proves the bridge works.
# Production plugins follow the same pattern: read .h5ad, transform, write .h5ad.
#
# Usage:
#   Rscript histoweave-sc-transform.R input.h5ad output.h5ad [target_sum=1e4]
#
# The caller (histoweave.plugins.builtin.r_demo) handles the container orchestration;
# this script runs inside the histoweave-r container image.

library(anndata)

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript histoweave-sc-transform.R <input.h5ad> <output.h5ad> [target_sum=1e4]")
}

input_path  <- args[1]
output_path <- args[2]
target_sum  <- 1e4  # default

if (length(args) >= 3) {
  # Parse "key=value" style arguments.
  for (arg in args[3:length(args)]) {
    parts <- strsplit(arg, "=")[[1]]
    if (length(parts) == 2 && parts[1] == "target_sum") {
      target_sum <- as.numeric(parts[2])
    }
  }
}

cat(sprintf("R normalisation: input=%s  target_sum=%.0f\n", input_path, target_sum))

# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------
adata <- read_h5ad(input_path)

# Work with the count matrix (dense or sparse).
counts <- adata$X
if (inherits(counts, "dgCMatrix") || inherits(counts, "dgRMatrix")) {
  counts <- as(counts, "CsparseMatrix")
}

# Library-size normalise to target_sum, then log1p.
lib_sizes <- Matrix::rowSums(counts)
lib_sizes[lib_sizes == 0] <- 1
normed <- sweep(counts, 1, lib_sizes, "/") * target_sum
adata$X <- log1p(normed)

# Store normalisation metadata in the AnnData object so the Python caller can
# verify that the round-trip was lossless.
adata$uns[["r_normalized"]] <- TRUE
adata$uns[["r_target_sum"]] <- target_sum

cat(sprintf("  wrote %d obs x %d vars -> %s\n", nrow(adata$X), ncol(adata$X), output_path))
write_h5ad(adata, output_path)
cat("R normalisation done.\n")
