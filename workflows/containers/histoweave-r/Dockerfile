# HistoWeave R container - wraps Bioconductor spatial methods as HistoWeave plugins.
FROM rocker/r-ver:4.4

LABEL org.opencontainers.image.source="https://github.com/histoweave-spatial/histoweave"
LABEL org.opencontainers.image.description="HistoWeave R method container (Bioconductor spatial methods)"
LABEL org.opencontainers.image.authors="HistoWeave contributors"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libgsl-dev \
    libgmp-dev \
    libmpfr-dev \
    && rm -rf /var/lib/apt/lists/*

RUN install2.r --error --skipinstalled \
    anndata \
    Matrix \
    SeuratObject \
    sctransform

RUN Rscript -e 'if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")' \
    && Rscript -e 'BiocManager::install(c("scran","scater","scuttle","SpatialExperiment","DropletUtils","Banksy"), ask=FALSE, update=FALSE)'

COPY histoweave-sc-transform.R /usr/local/bin/histoweave-sc-transform.R
COPY histoweave-banksy.R /usr/local/bin/histoweave-banksy.R
COPY histoweave-sctransform.R /usr/local/bin/histoweave-sctransform.R
RUN chmod +x /usr/local/bin/histoweave-sc-transform.R /usr/local/bin/histoweave-banksy.R /usr/local/bin/histoweave-sctransform.R

CMD ["Rscript", "-e", "cat('HistoWeave R container ready.\\n'); library(anndata); library(Matrix); library(Banksy); sessionInfo()"]
