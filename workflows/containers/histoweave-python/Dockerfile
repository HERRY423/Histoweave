# HistoWeave Python container — runs the core platform and non-R plugins.
#
# Build:
#   docker build -t ghcr.io/histoweave-spatial/histoweave-python:latest .
#   docker push ghcr.io/histoweave-spatial/histoweave-python:latest

FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/histoweave-spatial/histoweave"
LABEL org.opencontainers.image.description="HistoWeave Python method container"
LABEL org.opencontainers.image.authors="HistoWeave contributors"

# System deps for h5py (HDF5 reads), pyarrow (Parquet), and spatial libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the histoweave package with all extras baked in.
COPY . /opt/histoweave
WORKDIR /opt/histoweave
RUN pip install --no-cache-dir -e ".[io,spatial,scanpy,dev]" \
    && pip cache purge

# Pre-compile .pyc so first invocation is fast.
RUN python -c "import histoweave; print('HistoWeave', histoweave.__version__, 'ready.')"

ENTRYPOINT ["histoweave"]
CMD ["--help"]
