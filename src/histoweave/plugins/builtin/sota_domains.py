"""First-class SOTA spatial-domain methods (SpaGCN, GraphST, STAGATE, BayesSpace).

These plugins wrap the official upstream implementations.  Heavy dependencies are
imported inside ``run`` so a bare HistoWeave install can still list and select the
methods; missing backends raise explicit ``ModuleNotFoundError`` / ``RuntimeError``
rather than silently substituting a toy algorithm.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...data import SpatialTable
from ..interfaces import (
    BackendRequirement,
    Method,
    MethodCategory,
    MethodImplementation,
    MethodMaturity,
    MethodSpec,
    ParamSpec,
)
from ..registry import register
from ._sota_common import (
    cluster_embedding,
    make_adata,
    resolve_n_domains,
    torch_device,
)
from ._validation import validate_count_matrix, validate_spatial_coordinates


def _expression_matrix(data: SpatialTable, layer: str | None):
    if layer is None:
        return data.X
    if layer not in data.layers:
        raise KeyError(f"count layer {layer!r} does not exist")
    return data.layers[layer]


def _finalize_domains(
    method: Method,
    data: SpatialTable,
    labels: np.ndarray,
    *,
    key: str,
    extras: dict | None = None,
) -> SpatialTable:
    result = data.copy()
    result.obs[key] = pd.Categorical([f"domain_{int(lab)}" for lab in labels])
    payload = {
        "method": method.spec.name,
        "n_domains": int(pd.Series(labels).nunique()),
        "wraps": method.spec.wraps,
    }
    if extras:
        payload.update(extras)
    result.uns["domain_detection"] = payload
    return method.finalize(result, step="domain_detection")


@register
class SpaGCNDomains(Method):
    """Official SpaGCN spatial-domain detection (Hu et al., 2021)."""

    spec = MethodSpec(
        name="spagcn",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="SpaGCN graph-convolutional spatial domains (official package).",
        params=(
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec(
                "n_domains",
                "int|None",
                None,
                "Target domain count; falls back to uns['n_domains'].",
                minimum=2,
            ),
            ParamSpec(
                "p", "float", 0.5, "SpaGCN target neighbour percentage.", minimum=0.0, maximum=1.0
            ),
            ParamSpec("max_epochs", "int", 200, "Training epochs.", minimum=1),
            ParamSpec("random_state", "int", 0, "Random seed.", minimum=0),
            ParamSpec("key_added", "str", "domain", "obs column for labels."),
            ParamSpec(
                "refine_with_array",
                "bool",
                True,
                "Apply hexagon refine when array_row/array_col exist.",
            ),
        ),
        assumptions=(
            "obsm['spatial'] contains two-dimensional coordinates.",
            "Raw non-negative counts are available.",
        ),
        assays=("visium", "slideseq", "stereoseq"),
        maturity=MethodMaturity.BETA,
        wraps="SpaGCN (Hu et al., 2021)",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("SpaGCN", ">=1.2.7", "scanpy"),),
        metadata={"track": "sota", "task": "spatial_domain"},
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        try:
            import scanpy as sc
            import SpaGCN as spg
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SpaGCN (and scanpy/torch) are required for the spagcn method. "
                "Install SpaGCN==1.2.7 in a compatible environment."
            ) from exc

        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for SpaGCN")
        coords = validate_spatial_coordinates(data.spatial, method="SpaGCN", exact_dimensions=2)
        matrix = _expression_matrix(data, self.params["layer"])
        validate_count_matrix(matrix, method="SpaGCN")
        n_domains = resolve_n_domains(data, self.params["n_domains"])
        array_coords = None
        if (
            self.params["refine_with_array"]
            and "array_row" in data.obs.columns
            and "array_col" in data.obs.columns
        ):
            array_coords = np.column_stack(
                [
                    data.obs["array_row"].to_numpy(dtype=float),
                    data.obs["array_col"].to_numpy(dtype=float),
                ]
            )

        adata = make_adata(
            matrix,
            coords,
            list(data.var_names),
            obs_names=list(data.obs_names),
            n_genes=3000,
            array_coords=array_coords,
        )
        spg.prefilter_genes(adata, min_cells=3)
        spg.prefilter_specialgenes(adata)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        # SpaGCN<=1.2.7 still uses matrix.A (removed in SciPy 1.14+). Densify once
        # after preprocessing so official search_res / train paths keep working.
        if hasattr(adata.X, "toarray"):
            adata.X = np.asarray(adata.X.toarray(), dtype=np.float32)
        if "counts" in adata.layers and hasattr(adata.layers["counts"], "toarray"):
            adata.layers["counts"] = np.asarray(adata.layers["counts"].toarray(), dtype=np.float32)

        adj = spg.calculate_adj_matrix(x=coords[:, 0], y=coords[:, 1], histology=False)
        seed = int(self.params["random_state"])
        length_scale = spg.search_l(
            float(self.params["p"]), adj, start=0.01, end=1000, tol=0.01, max_run=100
        )
        if length_scale is None:
            raise RuntimeError("SpaGCN could not find a graph length scale")
        resolution = spg.search_res(
            adata,
            adj,
            length_scale,
            int(n_domains),
            start=0.7,
            step=0.1,
            tol=5e-3,
            lr=0.05,
            max_epochs=20,
            r_seed=seed,
            t_seed=seed,
            n_seed=seed,
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = spg.SpaGCN()
        model.set_l(length_scale)
        model.train(
            adata,
            adj,
            init_spa=True,
            init="louvain",
            res=resolution,
            tol=5e-3,
            lr=0.05,
            max_epochs=int(self.params["max_epochs"]),
        )
        labels, _ = model.predict()
        labels = np.asarray(labels)
        if array_coords is not None:
            grid_adj = spg.calculate_adj_matrix(
                x=array_coords[:, 0], y=array_coords[:, 1], histology=False
            )
            labels = spg.refine(
                sample_id=adata.obs_names.tolist(),
                pred=np.asarray(labels).tolist(),
                dis=grid_adj,
                shape="hexagon",
            )
        return _finalize_domains(
            self,
            data,
            np.asarray(labels, dtype=int),
            key=self.params["key_added"],
            extras={"length_scale": float(length_scale), "resolution": float(resolution)},
        )


@register
class GraphSTDomains(Method):
    """Official GraphST graph-contrastive spatial domains."""

    spec = MethodSpec(
        name="graphst",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="GraphST contrastive graph representation + fixed-q clustering.",
        params=(
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec(
                "n_domains",
                "int|None",
                None,
                "Target domain count; falls back to uns['n_domains'].",
                minimum=2,
            ),
            ParamSpec("epochs", "int", 600, "GraphST training epochs.", minimum=1),
            ParamSpec("random_state", "int", 0, "Random seed.", minimum=0),
            ParamSpec("key_added", "str", "domain", "obs column for labels."),
        ),
        assumptions=(
            "obsm['spatial'] contains two-dimensional coordinates.",
            "Raw non-negative counts are available.",
        ),
        assays=("visium", "xenium", "slideseq"),
        maturity=MethodMaturity.BETA,
        wraps="GraphST (Long et al. / JinmiaoChenLab)",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("GraphST", "official package", "deep-learning"),),
        metadata={"track": "sota", "task": "spatial_domain"},
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        try:
            import torch

            # Official layout: class lives in GraphST.GraphST (not re-exported).
            try:
                from GraphST.GraphST import GraphST as model_cls
            except ImportError:  # pragma: no cover - alternate packaging
                package = __import__("GraphST")
                model_cls = getattr(package, "GraphST", None)
                if model_cls is not None and not callable(model_cls):
                    model_cls = getattr(model_cls, "GraphST", None)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GraphST is required for the graphst method. "
                "Install JinmiaoChenLab/GraphST in a compatible environment."
            ) from exc

        if not callable(model_cls):
            raise ImportError(
                "GraphST package does not expose the GraphST model class "
                "(expected GraphST.GraphST.GraphST)"
            )

        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for GraphST")
        coords = validate_spatial_coordinates(data.spatial, method="GraphST", exact_dimensions=2)
        matrix = _expression_matrix(data, self.params["layer"])
        validate_count_matrix(matrix, method="GraphST")
        n_domains = resolve_n_domains(data, self.params["n_domains"])
        seed = int(self.params["random_state"])

        adata = make_adata(
            matrix,
            coords,
            list(data.var_names),
            obs_names=list(data.obs_names),
            n_genes=3000,
        )
        device = torch_device(torch)
        if isinstance(device, str):
            device = torch.device(device)
        model = model_cls(
            adata,
            device=device,
            epochs=int(self.params["epochs"]),
            random_seed=seed,
            datatype="10X",
        )
        result = model.train()
        if result is None:
            result = model.adata
        if "emb" not in result.obsm:
            raise RuntimeError("GraphST completed without adata.obsm['emb']")
        labels = cluster_embedding(result.obsm["emb"], n_domains=n_domains, seed=seed)
        out = data.copy()
        out.obsm["X_graphst"] = np.asarray(result.obsm["emb"], dtype=float)
        return _finalize_domains(
            self,
            out,
            labels,
            key=self.params["key_added"],
            extras={"epochs": int(self.params["epochs"])},
        )


@register
class STAGATEDomains(Method):
    """Official STAGATE_pyG graph-attention spatial domains."""

    spec = MethodSpec(
        name="stagate",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="STAGATE graph-attention autoencoder + fixed-q clustering.",
        params=(
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec(
                "n_domains",
                "int|None",
                None,
                "Target domain count; falls back to uns['n_domains'].",
                minimum=2,
            ),
            ParamSpec("n_epochs", "int", 1000, "Training epochs.", minimum=1),
            ParamSpec("random_state", "int", 0, "Random seed.", minimum=0),
            ParamSpec("key_added", "str", "domain", "obs column for labels."),
        ),
        assumptions=(
            "obsm['spatial'] contains two-dimensional coordinates.",
            "Raw non-negative counts are available.",
        ),
        assays=("visium", "xenium", "slideseq", "stereoseq"),
        maturity=MethodMaturity.BETA,
        wraps="STAGATE_pyG (Dong & Zhang / QIFEIDKN)",
        language="python",
        implementation=MethodImplementation.EXTERNAL,
        backends=(BackendRequirement("STAGATE_pyG", "official package", "deep-learning"),),
        metadata={"track": "sota", "task": "spatial_domain"},
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        try:
            stagate = __import__("STAGATE_pyG")
            import scanpy as sc
            import torch
            from sklearn.neighbors import NearestNeighbors
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "STAGATE_pyG (and scanpy/torch/sklearn) are required for the "
                "stagate method. Install QIFEIDKN/STAGATE_pyG in a compatible environment."
            ) from exc

        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for STAGATE")
        coords = validate_spatial_coordinates(data.spatial, method="STAGATE", exact_dimensions=2)
        matrix = _expression_matrix(data, self.params["layer"])
        validate_count_matrix(matrix, method="STAGATE")
        n_domains = resolve_n_domains(data, self.params["n_domains"])
        seed = int(self.params["random_state"])

        adata = make_adata(
            matrix,
            coords,
            list(data.var_names),
            obs_names=list(data.obs_names),
            n_genes=3000,
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(3000, adata.n_vars))

        n_neighbors = min(7, data.n_obs)
        distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(coords).kneighbors(coords)
        radius = float(np.median(distances[:, -1]) * 1.05)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("could not derive a positive STAGATE spatial radius")
        stagate.Cal_Spatial_Net(adata, rad_cutoff=radius)
        result = stagate.train_STAGATE(
            adata,
            n_epochs=int(self.params["n_epochs"]),
            random_seed=seed,
            device=torch_device(torch),
            verbose=False,
        )
        if "STAGATE" not in result.obsm:
            raise RuntimeError("STAGATE completed without adata.obsm['STAGATE']")
        labels = cluster_embedding(result.obsm["STAGATE"], n_domains=n_domains, seed=seed)
        out = data.copy()
        out.obsm["X_stagate"] = np.asarray(result.obsm["STAGATE"], dtype=float)
        return _finalize_domains(
            self,
            out,
            labels,
            key=self.params["key_added"],
            extras={"radius": radius, "n_epochs": int(self.params["n_epochs"])},
        )


@register
class BayesSpaceDomains(Method):
    """Bioconductor BayesSpace spatial clustering via Rscript."""

    spec = MethodSpec(
        name="bayesspace",
        category=MethodCategory.DOMAIN_DETECTION,
        version="0.1.0",
        summary="BayesSpace Bayesian spatial clustering (Bioconductor).",
        params=(
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec(
                "n_domains",
                "int|None",
                None,
                "Target domain count q; falls back to uns['n_domains'].",
                minimum=2,
            ),
            ParamSpec("nrep", "int", 10000, "MCMC iterations.", minimum=100),
            ParamSpec("random_state", "int", 0, "Random seed.", minimum=0),
            ParamSpec("key_added", "str", "domain", "obs column for labels."),
            ParamSpec(
                "r_script",
                "str|None",
                None,
                "Optional path to a BayesSpace R driver script.",
            ),
        ),
        assumptions=(
            "obsm['spatial'] present.",
            "Visium array_row/array_col preferred for full BayesSpace geometry.",
            "R + Bioconductor BayesSpace + zellkonverter available via Rscript.",
        ),
        assays=("visium",),
        maturity=MethodMaturity.BETA,
        wraps="Bioconductor::BayesSpace",
        language="r",
        implementation=MethodImplementation.EXTERNAL,
        backends=(
            BackendRequirement("Bioconductor::BayesSpace", ">=1.0", runtime="r"),
            BackendRequirement("Rscript", "on PATH", runtime="r"),
        ),
        metadata={"track": "sota", "task": "spatial_domain"},
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for BayesSpace")
        if "array_row" not in data.obs.columns or "array_col" not in data.obs.columns:
            raise ValueError(
                "BayesSpace requires obs['array_row'] and obs['array_col'] "
                "(Visium array coordinates)"
            )
        coords = validate_spatial_coordinates(data.spatial, method="BayesSpace", exact_dimensions=2)
        matrix = _expression_matrix(data, self.params["layer"])
        validate_count_matrix(matrix, method="BayesSpace")
        n_domains = resolve_n_domains(data, self.params["n_domains"])
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError(
                "BayesSpace requires Rscript on PATH (or the histoweave-r container)"
            )

        script = self.params["r_script"]
        if script is None:
            candidates = [
                Path(__file__).resolve().parents[4] / "5x15_spatial_aware" / "run_bayesspace.R",
                Path(os.environ.get("HISTOWEAVE_BAYESSPACE_SCRIPT", "")),
            ]
            script_path = next((path for path in candidates if path and path.exists()), None)
        else:
            script_path = Path(script)
        if script_path is None or not script_path.exists():
            raise FileNotFoundError(
                "BayesSpace R driver not found. Set params['r_script'] or "
                "HISTOWEAVE_BAYESSPACE_SCRIPT to run_bayesspace.R"
            )

        array_coords = np.column_stack(
            [
                data.obs["array_row"].to_numpy(dtype=float),
                data.obs["array_col"].to_numpy(dtype=float),
            ]
        )
        adata = make_adata(
            matrix,
            coords,
            list(data.var_names),
            obs_names=list(data.obs_names),
            n_genes=2000,
            array_coords=array_coords,
        )
        tmp = Path(tempfile.mkdtemp(prefix="histoweave_bayesspace_"))
        try:
            in_h5 = tmp / "in.h5ad"
            out_csv = tmp / "labels.csv"
            adata.write_h5ad(in_h5)
            r_lib = os.environ.get("HISTOWEAVE_R_LIB", "")
            cmd = [
                rscript,
                "--vanilla",
                str(script_path),
                str(in_h5),
                str(out_csv),
                str(int(self.params["random_state"])),
                str(int(n_domains)),
                str(int(self.params["nrep"])),
            ]
            if r_lib:
                cmd.append(r_lib)
            env = os.environ.copy()
            if r_lib:
                env["R_LIBS_USER"] = r_lib
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env, timeout=7200, check=False
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"BayesSpace failed (exit {proc.returncode})\n"
                    f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
                )
            table = pd.read_csv(out_csv)
            if "label" not in table.columns:
                raise RuntimeError(f"unexpected BayesSpace columns: {list(table.columns)}")
            labels = table["label"].to_numpy(dtype=int)
            if labels.shape[0] != data.n_obs:
                raise RuntimeError("BayesSpace label count does not match observations")
            return _finalize_domains(
                self,
                data,
                labels,
                key=self.params["key_added"],
                extras={"nrep": int(self.params["nrep"])},
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@register
class RCTDDeconvolution(Method):
    """spacexr RCTD cell-type deconvolution via Rscript (first-class plugin)."""

    spec = MethodSpec(
        name="rctd",
        category=MethodCategory.DECONVOLUTION,
        version="0.1.0",
        summary="RCTD robust cell-type decomposition (spacexr) via R bridge.",
        params=(
            ParamSpec("layer", "str|None", "counts", "Raw-count layer; None uses X."),
            ParamSpec(
                "reference_key",
                "str",
                "rctd_reference",
                "uns key with genes x cell-types reference signature matrix.",
            ),
            ParamSpec(
                "doublet_mode",
                "str",
                "full",
                "RCTD doublet mode.",
                choices=("full", "doublet", "multi"),
            ),
            ParamSpec("max_cores", "int", 1, "R parallel cores.", minimum=1),
            ParamSpec("random_state", "int", 0, "Random seed.", minimum=0),
            ParamSpec("abundance_key", "str", "rctd_weights", "obsm key for weights."),
            ParamSpec(
                "r_script",
                "str|None",
                None,
                "Optional path to an RCTD R driver script.",
            ),
        ),
        assumptions=(
            "Raw integer-like counts available.",
            "uns[reference_key] is a genes x cell-types non-negative matrix.",
            "R + spacexr available via Rscript or container.",
        ),
        assays=("visium", "slideseq", "stereoseq"),
        maturity=MethodMaturity.BETA,
        wraps="spacexr::RCTD (Cable et al., 2022)",
        language="r",
        implementation=MethodImplementation.EXTERNAL,
        backends=(
            BackendRequirement("spacexr", ">=2.0", runtime="r"),
            BackendRequirement("Rscript", "on PATH", runtime="r"),
        ),
        metadata={"track": "sota", "task": "deconvolution"},
    )

    def run(self, data: SpatialTable) -> SpatialTable:
        reference_key = self.params["reference_key"]
        if reference_key not in data.uns:
            raise KeyError(
                f"rctd reference {reference_key!r} is missing from uns; "
                "provide a genes x cell-types signature matrix"
            )
        reference = pd.DataFrame(data.uns[reference_key]).copy()
        if reference.empty:
            raise ValueError("rctd reference signature matrix is empty")
        try:
            values = reference.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("rctd reference signatures must be numeric") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("rctd reference signatures must be finite and non-negative")

        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError(
                "RCTD requires Rscript on PATH (or the histoweave-r container). "
                "Install spacexr and provide an R driver via params['r_script']."
            )
        script = self.params["r_script"]
        script_path = Path(script) if script else Path(os.environ.get("HISTOWEAVE_RCTD_SCRIPT", ""))
        if not script_path or not script_path.exists():
            # Explicit, non-silent failure: first-class plugin is registered, but
            # the R driver must be provisioned for production runs.
            raise FileNotFoundError(
                "RCTD R driver not found. Ship a spacexr driver script and set "
                "params['r_script'] or HISTOWEAVE_RCTD_SCRIPT. "
                "HistoWeave refuses to substitute a marker-score fallback."
            )

        if data.spatial is None:
            raise ValueError("obsm['spatial'] is required for RCTD")
        matrix = _expression_matrix(data, self.params["layer"])
        validate_count_matrix(matrix, method="RCTD")
        coords = validate_spatial_coordinates(data.spatial, method="RCTD", minimum_dimensions=2)

        tmp = Path(tempfile.mkdtemp(prefix="histoweave_rctd_"))
        try:
            counts_path = tmp / "counts.csv"
            coords_path = tmp / "coords.csv"
            ref_path = tmp / "reference.csv"
            out_path = tmp / "weights.csv"
            # Dense export is intentional for the R bridge contract; large assays
            # should use the containerised Nextflow path instead.
            dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
            pd.DataFrame(dense, index=data.obs_names, columns=data.var_names).T.to_csv(counts_path)
            pd.DataFrame(coords[:, :2], index=data.obs_names, columns=["x", "y"]).to_csv(
                coords_path
            )
            reference.to_csv(ref_path)
            proc = subprocess.run(
                [
                    rscript,
                    "--vanilla",
                    str(script_path),
                    str(counts_path),
                    str(coords_path),
                    str(ref_path),
                    str(out_path),
                    str(self.params["doublet_mode"]),
                    str(int(self.params["max_cores"])),
                    str(int(self.params["random_state"])),
                ],
                capture_output=True,
                text=True,
                timeout=7200,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"RCTD failed (exit {proc.returncode})\n"
                    f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
                )
            weights = pd.read_csv(out_path, index_col=0)
            weights = weights.reindex(data.obs_names)
            if weights.isna().any().any():
                raise RuntimeError("RCTD weights missing for some observations")
            result = data.copy()
            result.obsm[self.params["abundance_key"]] = weights.to_numpy(dtype=float)
            result.uns["deconvolution"] = {
                "method": "rctd",
                "cell_types": list(weights.columns.astype(str)),
                "doublet_mode": self.params["doublet_mode"],
            }
            return self.finalize(result, step="deconvolution")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
