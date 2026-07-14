ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG HISTOWEAVE_EXTRAS=io,spatial,scanpy,harmony,spatialde,cell2location,scanvi,cellpose2

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/histoweave-venv
ENV PATH="/opt/histoweave-venv/bin:${PATH}"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/pip pip install --upgrade pip setuptools wheel && pip install ".[${HISTOWEAVE_EXTRAS}]" && python -m compileall -q /opt/histoweave-venv

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG VERSION=dev
ARG REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/HERRY423/Histoweave"
LABEL org.opencontainers.image.description="HistoWeave Python method container"
LABEL org.opencontainers.image.authors="HistoWeave contributors"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${REVISION}"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

ENV PATH="/opt/histoweave-venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp/numba

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/* && groupadd --system histoweave && useradd --system --gid histoweave --create-home histoweave

COPY --from=builder /opt/histoweave-venv /opt/histoweave-venv

USER histoweave
WORKDIR /home/histoweave

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD python -c "import histoweave, anndata, scanpy, SpatialDE, cell2location, cellpose, scvi; print(histoweave.__version__)" || exit 1

ENTRYPOINT ["histoweave"]
CMD ["--help"]
