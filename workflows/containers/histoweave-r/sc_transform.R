#!/usr/bin/env Rscript
# R-side reference method: library-size normalisation + log1p.
#
# Called as:
#   Rscript sc_transform.R <input.h5ad> <output.h5ad> [target_sum=1e4]
#
# This is the minimal "does the R bridge work?" proof — it reads a .h5ad,
# normalises the count matrix, and writes the result back. The Python-side
# plugin (r_lognorm) shells out here after converting .ttab → .h5ad.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript sc_transform.R <input.h5ad> <output.h5ad> [target_sum=1e4]")
}

input_path  <- args[1]
output_path <- args[2]
target_sum  <- if (length(args) >= 3) as.numeric(sub("target_sum=", "", args[3])) else 1e4

library(anndata)

# ---- load ---------------------------------------------------------------
cat(sprintf("[R] Reading %s\n", input_path))
adata <- read_h5ad(input_path)

# ---- normalise -----------------------------------------------------------
counts <- Matrix::t(adata$X)   # AnnData X is (obs x var); R anndata stores (var x obs)
target_sum <- as.numeric(target_sum)
libsize <- Matrix::colSums(counts)
libsize[libsize == 0] <- 1
normed <- sweep(counts, 2, libsize / target_sum, "/")
adata$X <- Matrix::t(log1p(normed))

adata$uns[["r_normalized"]] <- TRUE
adata$uns[["r_target_sum"]]  <- target_sum

cat(sprintf("[R] Normalised %d genes x %d cells -> %s\n",
            nrow(adata$X), ncol(adata$X), output_path))

# ---- write ---------------------------------------------------------------
write_h5ad(adata, output_path)
cat("[R] Done.\n")
