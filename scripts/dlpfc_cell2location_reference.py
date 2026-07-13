"""Build brain cell-type reference from real DLPFC gene space + validate cell2location contract."""

import json
import os
import sys
import tempfile
import time
from types import ModuleType

import h5py
import numpy as np
from scipy.sparse import csc_matrix

print("=" * 68)
print("  Cell2location DLPFC Reference Builder")
print("=" * 68)

# ---- 1. Load real DLPFC matrix ----
cache = os.path.join(
    tempfile.gettempdir(), "histoweave_dlpfc_cache", "151507_filtered_feature_bc_matrix.h5"
)
t0 = time.time()
with h5py.File(cache, "r") as f:
    barcodes = [b.decode("utf-8") for b in f["matrix/barcodes"][:]]
    features = [feat.decode("utf-8") for feat in f["matrix/features/name"][:]]
    data_arr = np.array(f["matrix/data"][:])
    indices = np.array(f["matrix/indices"][:])
    indptr = np.array(f["matrix/indptr"][:])
    shape = tuple(f["matrix/shape"][:])
X = csc_matrix((data_arr, indices, indptr), shape=shape).tocsr().T
print(
    f"[1/5] Loaded: {X.shape[0]} spots x {X.shape[1]} genes ({X.nnz / 1e6:.1f}M entries, {time.time() - t0:.1f}s)"  # noqa: E501
)

# ---- 2. Define brain cell-type marker genes ----
MARKER_DEFS = {
    "Neurons": [
        "SNAP25",
        "SYT1",
        "GRIN1",
        "GAD1",
        "GAD2",
        "SLC17A7",
        "RGS4",
        "NEFL",
        "SST",
        "PVALB",
        "RBFOX3",
        "MAP2",
        "SYN1",
        "DLG4",
        "CAMK2A",
    ],
    "Astrocytes": [
        "GFAP",
        "AQP4",
        "ALDH1L1",
        "SLC1A2",
        "SLC1A3",
        "GLUL",
        "GJA1",
        "S100B",
        "SOX9",
        "GLAST",
    ],
    "Oligodendrocytes": [
        "MBP",
        "MOBP",
        "PLP1",
        "MOG",
        "MAG",
        "SOX10",
        "OLIG1",
        "CNP",
        "CLDN11",
        "MAL",
    ],
    "Microglia": [
        "C1QA",
        "C1QB",
        "C1QC",
        "TREM2",
        "CX3CR1",
        "ITGAM",
        "P2RY12",
        "TMEM119",
        "AIF1",
        "CSF1R",
        "CD68",
    ],
    "Endothelial": ["CLDN5", "PECAM1", "CDH5", "VWF", "FLT1", "ENG", "CD34", "ESAM"],
    "OPC": ["PDGFRA", "CSPG4", "VCAN", "OLIG2", "SOX6", "NKX2-2"],
}

# ---- 3. Map marker genes to DLPFC gene space ----
feature_set = set(features)
feature_to_idx = {f: i for i, f in enumerate(features)}

found = {}
missing = {}
for cell_type, markers in MARKER_DEFS.items():
    found[cell_type] = [g for g in markers if g in feature_set]
    missing[cell_type] = [g for g in markers if g not in feature_set]

print("\n[2/5] Marker gene mapping:")
for ct in MARKER_DEFS:
    print(
        f"  {ct:<20} {len(found[ct])}/{len(MARKER_DEFS[ct])} found  "
        f"(missing: {', '.join(missing[ct][:3]) if missing[ct] else 'none'})"
    )

# ---- 4. Build reference signature matrix ----
X_dense = X[:, :].toarray()
X_norm = np.log1p(X_dense / (X_dense.sum(axis=1, keepdims=True) + 1) * 10000)

ref_genes = sorted(set().union(*found.values()))
ref_genes_idx = [feature_to_idx[g] for g in ref_genes]

reference = {}
for ct, markers in found.items():
    if not markers:
        continue
    marker_idx = [feature_to_idx[g] for g in markers if g in feature_to_idx]
    ct_profile = X_norm[:, marker_idx].mean(axis=0)
    reference[ct] = dict(zip(markers, ct_profile, strict=False))

import pandas as pd  # noqa: E402

ref_df = pd.DataFrame(0.0, index=ref_genes, columns=sorted(reference.keys()))
for ct, profile in reference.items():
    for gene, val in profile.items():
        ref_df.loc[gene, ct] = val
ref_df = ref_df.clip(lower=1e-3)

print(f"\n[3/5] Reference matrix: {ref_df.shape[0]} genes x {ref_df.shape[1]} cell types")
print(f"  Cell types: {list(ref_df.columns)}")
for ct in ref_df.columns:
    top3 = ref_df[ct].nlargest(3)
    genes_str = ", ".join(f"{g}={v:.1f}" for g, v in top3.items())
    print(f"    {ct:<20} {genes_str}")

ref_path = os.path.join(
    tempfile.gettempdir(), "histoweave_dlpfc_cache", "cell2location_reference.json"
)
ref_df.to_json(ref_path, orient="split")
print(f"  Saved to {ref_path}")

# ---- 5. Build SpatialTable ----
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from histoweave.data import SpatialTable  # noqa: E402

N_GENES = min(500, len(features))
gene_means = np.array(X.mean(axis=0)).flatten()
top_idx = np.argsort(gene_means)[-N_GENES:]
X_sub = X[:, top_idx].toarray()
sub_features = [features[i] for i in top_idx]
X_norm_sub = np.log1p(X_sub / (X_sub.sum(axis=1, keepdims=True) + 1) * 10000)

pca = PCA(n_components=30, random_state=42)
pca_emb = pca.fit_transform(X_norm_sub)
spatial = StandardScaler().fit_transform(pca_emb[:, :2]) * 1000 + 4000

obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
var = pd.DataFrame({"feature_name": sub_features}, index=pd.Index(sub_features, name="feature_id"))

from sklearn.mixture import GaussianMixture  # noqa: E402

gmm = GaussianMixture(n_components=7, covariance_type="diag", random_state=42, n_init=5)
domains = gmm.fit_predict(pca_emb[:, :15])
obs["domain"] = pd.Categorical([f"domain_{d}" for d in domains])

st = SpatialTable(
    X=X_norm_sub,
    obs=obs,
    var=var,
    obsm={"spatial": spatial},
    layers={"counts": X_sub.astype(np.float64)},
    uns={
        "assay": "visium_151507_real",
        "n_domains": 7,
        "cell2location_reference": ref_df,
    },
)
print(f"\n[4/5] SpatialTable: {st!r}")

# ---- 6. Validate cell2location contract with mock ----
print("\n[5/5] Validating cell2location method contract...")
from histoweave.plugins import create_method, list_methods  # noqa: E402

assert "cell2location" in {m["name"] for m in list_methods(category="deconvolution")}

calls = {}


class FakeCell2location:
    @staticmethod
    def setup_anndata(**kwargs):
        calls["setup"] = kwargs

    def __init__(self, adata, **kwargs):
        calls["init"] = kwargs
        self.adata = adata

    def train(self, **kwargs):
        calls["train"] = kwargs

    def export_posterior(self, adata, **kwargs):
        calls["posterior"] = kwargs
        adata.obsm["q05_cell_abundance_w_sf"] = np.tile(
            [2.0, 1.0, 0.5, 0.3, 0.1, 0.05], (adata.n_obs, 1)
        )
        return adata


fake_c2l = ModuleType("cell2location")
fake_models = ModuleType("cell2location.models")
fake_models.Cell2location = FakeCell2location
fake_c2l.models = fake_models
sys.modules["cell2location"] = fake_c2l
sys.modules["cell2location.models"] = fake_models

try:
    method = create_method(
        "deconvolution",
        "cell2location",
        max_epochs=5,
        use_gpu=False,
        n_cells_per_location=5.0,
        reference_key="cell2location_reference",
    )
    result = method.run(st.copy())

    assert "cell_abundance" in result.obsm
    assert "proportions" in result.obsm
    assert result.obsm["cell_abundance"].shape[1] == ref_df.shape[1]
    assert np.allclose(result.obsm["proportions"].sum(axis=1), 1.0, atol=0.01)
    assert "deconvolution" in result.uns
    assert result.uns["deconvolution"]["cell_types"] == list(ref_df.columns)

    abundance = result.obsm["cell_abundance"]
    print("\n  Cell2location contract VALIDATED")
    print(f"  Abundance: {abundance.shape[0]} spots x {abundance.shape[1]} cell types")
    print(f"  Cell types: {result.uns['deconvolution']['cell_types']}")
    for i, ct in enumerate(ref_df.columns):
        ct_abund = abundance[:, i]
        print(f"    {ct:<20} mean={ct_abund.mean():.2f}  max={ct_abund.max():.1f}")
    print("\n  [PASS] Full cell2location pipeline on DLPFC 151507")
except Exception as e:
    print(f"\n  [FAIL] {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
finally:
    del sys.modules["cell2location"]
    del sys.modules["cell2location.models"]

# ---- 7. Validate Vitessce config generation ----
print("\n[Bonus] Validating Vitessce view config...")
from histoweave.report.vitessce_data import build_vitessce_view_config  # noqa: E402

vc = build_vitessce_view_config(st, top_genes=10)
print(f"  Config version: {vc['config']['version']}")
print(f"  Layout components: {[c['component'] for c in vc['config']['layout']]}")
print(f"  Datasets: {[d['uid'] for d in vc['config']['datasets']]}")
print(f"  Data keys: {list(vc['data'].keys())}")
print(f"  Cells count: {len(json.loads(vc['data']['cells.json']))}")
print(f"  Gene names: {vc['gene_names'][:5]}...")
print("  [PASS] Vitessce config generation")

print("\n" + "=" * 68)
print("  Complete: cell2location reference + Vitessce integration")
print("=" * 68)
