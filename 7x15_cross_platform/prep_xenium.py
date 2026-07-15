"""Prepare a real 10x Xenium dataset (Human Breast Cancer Rep1, Xenium v1.0.1).

Source: 10x Genomics public Xenium bundle
  https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/
We download the full outs.zip (~9.9 GB), extract only the cell_feature_matrix (MEX) and
cells.csv.gz (centroids), build an AnnData, then delete the zip.

Xenium ships no manual spatial-domain annotation, so we derive a proxy domain label by
Leiden clustering the expression embedding (documented in the report). This mirrors the
'proxy label fallback' the user approved for platforms lacking expert domain truth.

If the 10x download is unavailable, set XENIUM_FALLBACK_SEQFISH=1 to substitute the
squidpy seqFISH mouse-embryo dataset (imaging-based, single-cell resolution) instead.
"""

from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import scanpy as sc
from _prep_common import finalize

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


BUNDLE = (
    "https://cf.10xgenomics.com/samples/xenium/1.0.1/"
    "Xenium_FFPE_Human_Breast_Cancer_Rep1/"
    "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip"
)
DL_DIR = Path(os.environ.get("HISTOWEAVE_XENIUM_DL", "/workspace/xenium_dl"))
DL_DIR.mkdir(parents=True, exist_ok=True)


def _leiden_proxy(a: sc.AnnData, resolution: float = 1.0) -> sc.AnnData:
    """Cluster expression to produce a proxy domain label (Xenium has no manual truth)."""
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.pca(tmp, n_comps=min(50, tmp.n_vars - 1), random_state=0)
    sc.pp.neighbors(tmp, n_neighbors=15, random_state=0)
    sc.tl.leiden(
        tmp, resolution=resolution, random_state=0, flavor="igraph", n_iterations=2, directed=False
    )
    a.obs["proxy_cluster"] = tmp.obs["leiden"].values
    return a


def load_xenium_10x() -> sc.AnnData:
    zpath = DL_DIR / "xenium_bc_rep1_outs.zip"
    if not (zpath.exists() and zpath.stat().st_size > 9_000_000_000):
        _log("[xenium] downloading 10x bundle (~9.9 GB)...")
        # 10x CDN rejects the default urllib user-agent (403); use a browser UA
        # and stream to disk in chunks.
        req = urllib.request.Request(
            BUNDLE, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as fh:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                chunk = r.read(8 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if total:
                    _log(f"[xenium]   {done / 1e9:.2f}/{total / 1e9:.2f} GB")
    _log(f"[xenium] zip ready: {zpath.stat().st_size / 1e9:.2f} GB")

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        mtx = [n for n in names if n.endswith("cell_feature_matrix.h5")]
        cells = [n for n in names if n.endswith("cells.csv.gz")]
        if mtx:
            with z.open(mtx[0]) as fh:
                data = fh.read()
            h5tmp = DL_DIR / "cell_feature_matrix.h5"
            h5tmp.write_bytes(data)
            a = sc.read_10x_h5(str(h5tmp))
        else:
            # fall back to MEX folder
            mex = [n for n in names if "cell_feature_matrix/" in n]
            for n in mex:
                z.extract(n, DL_DIR)
            mexdir = DL_DIR / os.path.dirname(mex[0])
            a = sc.read_10x_mtx(str(mexdir))
        a.var_names_make_unique()
        with z.open(cells[0]) as fh:
            cdf = pd.read_csv(io.BytesIO(fh.read()), compression="gzip")

    cdf["cell_id"] = cdf["cell_id"].astype(str)
    cdf = cdf.drop_duplicates("cell_id").set_index("cell_id")
    a = a[a.obs_names.isin(cdf.index)].copy()
    coords = cdf.reindex(a.obs_names)
    a.obsm["spatial"] = coords[["x_centroid", "y_centroid"]].to_numpy(dtype=float)

    # keep panel genes only (drop control/blank probes)
    keep = ~a.var_names.str.contains("BLANK|NegControl|antisense|Control", case=False, regex=True)
    a = a[:, keep.to_numpy()].copy()

    a = _leiden_proxy(a)
    return a


def load_seqfish_fallback() -> sc.AnnData:
    import squidpy as sq

    a = sq.datasets.seqfish()
    return a


if __name__ == "__main__":
    if os.environ.get("XENIUM_FALLBACK_SEQFISH") == "1":
        a = load_seqfish_fallback()
        finalize(
            a,
            dataset_id="xenium",
            platform="seqFISH(fallback)",
            label_col="celltype_mapped_refined",
        )
    else:
        a = load_xenium_10x()
        finalize(a, dataset_id="xenium", platform="Xenium", label_col="proxy_cluster")
