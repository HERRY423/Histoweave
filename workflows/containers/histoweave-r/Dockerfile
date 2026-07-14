ARG R_VERSION=4.4.3
FROM rocker/r-ver:${R_VERSION}

ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/HERRY423/Histoweave"
LABEL org.opencontainers.image.description="HistoWeave R method container (Bioconductor spatial methods)"
LABEL org.opencontainers.image.authors="HistoWeave contributors"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${REVISION}"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

ENV DEBIAN_FRONTEND=noninteractive
ENV R_REMOTES_NO_ERRORS_FROM_WARNINGS=true

RUN set -eux; for attempt in 1 2 3 4 5; do apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=60 update && apt-get -o Acquire::Retries=10 -o Acquire::http::Timeout=60 install -y --no-install-recommends libcairo2-dev libcurl4-openssl-dev libfontconfig1-dev libfreetype6-dev libfribidi-dev libglpk-dev libgmp-dev libgsl-dev libharfbuzz-dev libhdf5-dev libjpeg-dev libmagick++-dev libmpfr-dev libpng-dev python3 python3-venv libssl-dev libtiff-dev libxml2-dev && break; if [ "$attempt" = 5 ]; then exit 1; fi; sleep 10; done; rm -rf /var/lib/apt/lists/*

RUN echo 'local({ r <- getOption("repos"); r["CRAN"] <- "https://packagemanager.posit.co/cran/__linux__/noble/latest"; r["CRAN2"] <- "https://cloud.r-project.org"; options(repos = r, timeout = 600) })' >> /usr/local/lib/R/etc/Rprofile.site

RUN --mount=type=cache,target=/root/.cache/R Rscript -e 'repos <- c("https://packagemanager.posit.co/cran/__linux__/noble/latest", "https://cloud.r-project.org"); dependencies <- c("Depends", "Imports", "LinkingTo"); loadable <- function(pkg) tryCatch({ loadNamespace(pkg); TRUE }, error = function(e) { message("Namespace check for ", pkg, " failed: ", conditionMessage(e)); FALSE }); install_one <- function(pkg) { if (loadable(pkg)) return(invisible(TRUE)); for (attempt in seq_len(5)) { for (repo in repos) { tryCatch(install.packages(pkg, repos = repo, dependencies = dependencies), error = function(e) message("Attempt ", attempt, " for ", pkg, " via ", repo, " failed: ", conditionMessage(e))); if (loadable(pkg)) return(invisible(TRUE)) }; Sys.sleep(10) }; stop("Failed to install ", pkg, " from all CRAN mirrors") }; invisible(lapply(c("jsonlite", "anndata", "Matrix", "SeuratObject", "sctransform", "BiocManager"), install_one))' && Rscript -e 'pkgs <- c("scran", "scater", "scuttle", "SpatialExperiment", "DropletUtils", "Banksy", "nnSVG", "glmGamPoi"); for (attempt in seq_len(5)) { missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (!length(missing)) break; tryCatch(BiocManager::install(missing, ask = FALSE, update = FALSE), error = function(e) message("Bioconductor attempt ", attempt, " failed: ", conditionMessage(e))); if (attempt < 5) Sys.sleep(10) }; missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]; if (length(missing)) stop("Missing Bioconductor packages: ", paste(missing, collapse = ", "))' && rm -rf /tmp/downloaded_packages

ARG HISTOWEAVE_EXTRAS=io,spatial
RUN python3 -m venv /opt/histoweave-venv
ENV PATH="/opt/histoweave-venv/bin:${PATH}"
ENV RETICULATE_PYTHON=/opt/histoweave-venv/bin/python

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip setuptools wheel && pip install ".[${HISTOWEAVE_EXTRAS}]" && python -m compileall -q /opt/histoweave-venv
COPY workflows/containers/histoweave-r/histoweave-sc-transform.R /usr/local/bin/histoweave-sc-transform.R
COPY workflows/containers/histoweave-r/histoweave-banksy.R /usr/local/bin/histoweave-banksy.R
COPY workflows/containers/histoweave-r/histoweave-sctransform.R /usr/local/bin/histoweave-sctransform.R
COPY workflows/containers/histoweave-r/histoweave-nnsvg.R /usr/local/bin/histoweave-nnsvg.R

RUN chmod 0755 /usr/local/bin/histoweave-sc-transform.R /usr/local/bin/histoweave-banksy.R /usr/local/bin/histoweave-sctransform.R /usr/local/bin/histoweave-nnsvg.R && useradd --system --create-home --shell /usr/sbin/nologin histoweave

USER histoweave
WORKDIR /home/histoweave

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD histoweave list >/dev/null && python -c "import anndata, histoweave" && Rscript -e 'stopifnot(all(vapply(c("anndata", "Matrix", "Banksy", "nnSVG", "sctransform", "glmGamPoi"), requireNamespace, logical(1), quietly = TRUE)))' || exit 1

CMD ["Rscript", "-e", "cat('HistoWeave R container ready.\\n'); library(anndata); library(Matrix); library(Banksy); library(nnSVG); sessionInfo()"]
