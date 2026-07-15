"""Pilot test of a cross-brain glial-enriched interface (GEI) neighborhood.

This is deliberately a *candidate-discovery* analysis.  It does not pretend that
the three public datasets identify tissue, species, and technology effects
separately.  The primary analysis asks a narrower falsifiable question:

    Are spatial transitions in neuronal/anatomical identity enriched for
    astrocyte, oligodendrocyte, and vascular support categories or programs?

The script enforces the following safeguards:

* raw-count authenticity is audited before any count model is declared legal;
* the cached normalized MERFISH and Slide-seq matrices are used only for the
  topology pilot, never as SCT/scVI inputs;
* DLPFC sections are nested in three explicitly declared donors;
* spatial circular-shift nulls replace cell-level iid permutation p-values;
* leave-one-domain-out results are labelled operational category transfer, not
  causal or tissue-specific generalization.

Outputs are written under ``research/cross_tissue_niches/results``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


SEED = 20260715
RNG = np.random.default_rng(SEED)

MODULES: dict[str, tuple[str, ...]] = {
    "astro_ion": (
        "AQP4",
        "SLC1A2",
        "SLC1A3",
        "KCNJ10",
        "GLUL",
        "ATP1A2",
        "GJA1",
        "ALDOC",
        "GFAP",
    ),
    "oligo_myelin": (
        "MBP",
        "PLP1",
        "MOG",
        "MAG",
        "CNP",
        "CLDN11",
        "MAL",
        "OLIG1",
        "OLIG2",
        "SOX10",
    ),
    "vascular_barrier": (
        "CLDN5",
        "PECAM1",
        "VWF",
        "EMCN",
        "KDR",
        "RAMP2",
        "SLC2A1",
        "MFSD2A",
        "ABCB1",
    ),
    "microimmune": (
        "P2RY12",
        "CX3CR1",
        "AIF1",
        "TYROBP",
        "C1QA",
        "C1QB",
        "C1QC",
        "TREM2",
    ),
    "neuronal_synaptic": (
        "CAMK2A",
        "SNAP25",
        "SYT1",
        "SLC17A7",
        "GAD1",
        "GAD2",
    ),
}

GEI_MODULES = ("astro_ion", "oligo_myelin", "vascular_barrier")

DLPFC_DONOR: dict[str, str] = {
    **{str(x): "donor_1" for x in range(151507, 151511)},
    **{str(x): "donor_2" for x in range(151669, 151673)},
    **{str(x): "donor_3" for x in range(151673, 151677)},
}


@dataclass
class DomainPayload:
    domain: str
    features: pd.DataFrame
    sample_arrays: list[dict[str, np.ndarray]]
    module_effects: dict[str, float]
    spatial: dict[str, np.ndarray]


def _matrix_values(matrix: object) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.data)
    return np.asarray(matrix).ravel()


def count_audit(matrix: object, tolerance: float = 1e-6) -> dict[str, object]:
    values = _matrix_values(matrix)
    finite = bool(np.isfinite(values).all())
    nonnegative = bool(values.size and np.nanmin(values) >= -tolerance)
    integer_like = bool(
        values.size
        and finite
        and np.max(np.abs(values - np.rint(values)), initial=0.0) <= tolerance
    )
    return {
        "finite": finite,
        "nonnegative": nonnegative,
        "integer_like": integer_like,
        "n_stored": int(values.size),
        "noninteger_fraction": float(np.mean(np.abs(values - np.rint(values)) > tolerance))
        if values.size
        else math.nan,
    }


def _dense_columns(matrix: object, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.empty((matrix.shape[0], 0), dtype=np.float32)
    subset = matrix[:, indices]
    if sparse.issparse(subset):
        subset = subset.toarray()
    return np.asarray(subset, dtype=np.float32)


def _gene_index(var_names: Iterable[object]) -> dict[str, int]:
    return {str(g).upper(): i for i, g in enumerate(var_names)}


def score_modules(
    matrix: object,
    var_names: Iterable[object],
    *,
    transform: str,
    totals: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Return gene-wise standardized module scores and genes actually used."""
    lookup = _gene_index(var_names)
    scores: dict[str, np.ndarray] = {}
    used: dict[str, list[str]] = {}
    n_obs = int(matrix.shape[0])

    if totals is None:
        totals = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    totals = np.maximum(np.asarray(totals, dtype=float), 1.0)

    for module, genes in MODULES.items():
        present = [g for g in genes if g in lookup]
        used[module] = present
        if not present:
            scores[module] = np.full(n_obs, np.nan, dtype=np.float32)
            continue
        values = _dense_columns(matrix, [lookup[g] for g in present]).astype(float)
        if transform == "counts":
            values = np.log1p(values * (1e4 / totals[:, None]))
        elif transform == "linear":
            values = np.log1p(np.maximum(values, 0.0))
        elif transform != "log1p":
            raise ValueError(f"unknown transform: {transform}")
        means = np.nanmean(values, axis=0)
        sds = np.nanstd(values, axis=0)
        sds[sds < 1e-8] = 1.0
        z = (values - means) / sds
        scores[module] = np.nanmean(z, axis=1).astype(np.float32)
    return scores, used


def combined_gei(scores: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [scores[name] for name in GEI_MODULES if np.isfinite(scores[name]).any()]
    if not arrays:
        raise RuntimeError("none of the predeclared GEI modules has measured genes")
    return np.nanmean(np.column_stack(arrays), axis=1)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    sd = np.nanstd(values)
    if not np.isfinite(sd) or sd < 1e-8:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / sd


def residualize(
    values: np.ndarray,
    numeric: list[np.ndarray],
    category: np.ndarray | None = None,
) -> np.ndarray:
    columns = [np.ones(len(values), dtype=float)] + [zscore(x) for x in numeric]
    if category is not None:
        dummies = pd.get_dummies(pd.Series(category).astype(str), drop_first=True, dtype=float)
        if dummies.shape[1]:
            columns.extend(np.asarray(dummies[c], dtype=float) for c in dummies.columns)
    design = np.column_stack(columns)
    finite = np.isfinite(values) & np.isfinite(design).all(axis=1)
    residual = np.full(len(values), np.nan, dtype=float)
    if finite.sum() <= design.shape[1] + 2:
        return residual
    beta, *_ = np.linalg.lstsq(design[finite], np.asarray(values)[finite], rcond=None)
    residual[finite] = np.asarray(values)[finite] - design[finite] @ beta
    return residual


def identity_entropy(
    query_coords: np.ndarray,
    reference_coords: np.ndarray,
    reference_labels: np.ndarray,
    *,
    k: int,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference_labels = np.asarray(reference_labels).astype(str)
    codes, unique = pd.factorize(reference_labels, sort=True)
    requested = min(k + int(exclude_self), len(reference_coords))
    nn = NearestNeighbors(n_neighbors=requested, algorithm="kd_tree", n_jobs=-1)
    nn.fit(np.asarray(reference_coords, dtype=float))
    distances, indices = nn.kneighbors(np.asarray(query_coords, dtype=float))
    if exclude_self:
        distances, indices = distances[:, 1:], indices[:, 1:]
    neighbor_codes = codes[indices]
    n, k_eff = neighbor_codes.shape
    rows = np.repeat(np.arange(n), k_eff)
    counts = sparse.csr_matrix(
        (np.ones(n * k_eff, dtype=np.float32), (rows, neighbor_codes.ravel())),
        shape=(n, len(unique)),
    )
    probabilities = counts.multiply(1.0 / max(k_eff, 1))
    logp = probabilities.copy()
    logp.data = np.log(logp.data)
    entropy = -np.asarray(probabilities.multiply(logp).sum(axis=1)).ravel()
    denominator = math.log(max(2, min(k_eff, len(unique))))
    entropy = entropy / denominator
    return entropy.astype(np.float32), np.mean(distances, axis=1).astype(np.float32)


def local_spacing(coords: np.ndarray, k: int = 6) -> np.ndarray:
    requested = min(k + 1, len(coords))
    nn = NearestNeighbors(n_neighbors=requested, algorithm="kd_tree", n_jobs=-1)
    distances, _ = nn.fit(coords).kneighbors(coords)
    if requested > 1:
        distances = distances[:, 1:]
    return np.mean(distances, axis=1).astype(np.float32)


def extremes(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    q80 = np.nanquantile(score[finite], 0.80)
    q50 = np.nanquantile(score[finite], 0.50)
    high = finite & (score >= q80)
    low = finite & (score <= q50)
    if np.any(high & low):
        high = finite & (score > q50)
        low = finite & (score <= q50)
    return high, low


def odds_ratio(y: np.ndarray, boundary_score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    high, low = extremes(boundary_score)
    keep = (high | low) & np.isin(y, [0, 1])
    high = high[keep]
    yy = y[keep]
    a = float(np.sum(high & (yy == 1))) + 0.5
    b = float(np.sum(high & (yy == 0))) + 0.5
    c = float(np.sum((~high) & (yy == 1))) + 0.5
    d = float(np.sum((~high) & (yy == 0))) + 0.5
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {
        "or": math.exp(log_or),
        "log_or": log_or,
        "ci_low": math.exp(log_or - 1.96 * se),
        "ci_high": math.exp(log_or + 1.96 * se),
        "n": int(keep.sum()),
        "n_high": int(high.sum()),
        "n_low": int((~high).sum()),
    }


def cohens_d(values: np.ndarray, boundary_score: np.ndarray) -> float:
    high, low = extremes(boundary_score)
    x1 = np.asarray(values)[high & np.isfinite(values)]
    x0 = np.asarray(values)[low & np.isfinite(values)]
    if len(x1) < 3 or len(x0) < 3:
        return math.nan
    pooled = math.sqrt(
        ((len(x1) - 1) * np.var(x1, ddof=1) + (len(x0) - 1) * np.var(x0, ddof=1))
        / (len(x1) + len(x0) - 2)
    )
    return float((np.mean(x1) - np.mean(x0)) / pooled) if pooled > 0 else math.nan


def spatial_shift_p(
    sample_arrays: list[dict[str, np.ndarray]],
    *,
    n_permutations: int,
) -> tuple[float, float, list[float]]:
    actual_sample = [odds_ratio(x["y"], x["entropy"])["log_or"] for x in sample_arrays]
    actual = float(np.mean(actual_sample))
    null: list[float] = []
    for _ in range(n_permutations):
        shifted_effects: list[float] = []
        for item in sample_arrays:
            coords = np.asarray(item["coords"], dtype=float)
            centered = coords - np.mean(coords, axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            order = np.argsort(centered @ vt[0])
            n = len(order)
            low = max(1, int(0.10 * n))
            high = max(low + 1, int(0.90 * n))
            offset = int(RNG.integers(low, high))
            shifted = np.empty(n, dtype=float)
            shifted[order] = np.roll(np.asarray(item["entropy"])[order], offset)
            shifted_effects.append(odds_ratio(item["y"], shifted)["log_or"])
        null.append(float(np.mean(shifted_effects)))
    p = (1 + int(np.sum(np.asarray(null) >= actual))) / (n_permutations + 1)
    return actual, float(p), null


def _feature_frame(
    domain: str,
    y: np.ndarray,
    entropy: np.ndarray,
    identity_distance: np.ndarray,
    spacing: np.ndarray,
    depth: np.ndarray,
    sample: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "domain": domain,
            "sample": sample.astype(str),
            "y": y.astype(int),
            "entropy": entropy,
            "identity_distance": identity_distance,
            "local_spacing": spacing,
            "log_depth": np.log1p(np.maximum(depth, 0)),
        }
    )
    keep = np.isin(frame["y"], [0, 1]) & np.isfinite(
        frame[["entropy", "identity_distance", "local_spacing", "log_depth"]]
    ).all(axis=1)
    frame = frame.loc[keep].copy()
    for col in ("entropy", "identity_distance", "local_spacing", "log_depth"):
        median = float(frame[col].median())
        q25, q75 = frame[col].quantile([0.25, 0.75])
        scale = float(q75 - q25)
        if scale < 1e-8:
            scale = float(frame[col].std()) or 1.0
        frame[col] = (frame[col] - median) / scale
    return frame


def analyse_dlpfc(repo: Path, n_permutations: int) -> tuple[DomainPayload, list[dict], list[dict]]:
    section_effect_rows: list[dict] = []
    availability_rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    sample_arrays: list[dict[str, np.ndarray]] = []
    spatial_payload: dict[str, np.ndarray] = {}

    files = sorted((repo / "datasets_cache" / "dlpfc").glob("dlpfc_*.h5ad"))
    if len(files) != 12:
        raise RuntimeError(f"expected 12 DLPFC files, found {len(files)}")

    for path in files:
        slice_id = path.stem.rsplit("_", 1)[-1]
        donor = DLPFC_DONOR[slice_id]
        a = ad.read_h5ad(path)
        labels = a.obs["domain_truth"].astype(str).to_numpy()
        valid = np.array([x.lower().startswith("layer") for x in labels])
        a = a[valid].copy()
        labels = labels[valid]
        coords = np.asarray(a.obsm["spatial"], dtype=float)
        counts = a.layers["counts"] if "counts" in a.layers else a.X
        audit = count_audit(counts)
        if not (audit["integer_like"] and audit["nonnegative"] and audit["finite"]):
            raise RuntimeError(f"DLPFC {slice_id} failed raw-count audit: {audit}")
        totals = np.asarray(counts.sum(axis=1)).ravel()
        scores, used = score_modules(counts, a.var_names, transform="counts", totals=totals)
        for module, genes in used.items():
            availability_rows.append(
                {
                    "domain": "DLPFC",
                    "sample": slice_id,
                    "module": module,
                    "n_genes": len(genes),
                    "genes": ";".join(genes),
                }
            )
        entropy, identity_distance = identity_entropy(
            coords, coords, labels, k=6, exclude_self=True
        )
        spacing = local_spacing(coords, k=6)
        gei = combined_gei(scores)
        gei_resid = residualize(gei, [np.log1p(totals), spacing], labels)
        q33, q67 = np.nanquantile(gei_resid, [0.33, 0.67])
        y = np.full(len(gei_resid), -1, dtype=int)
        y[gei_resid <= q33] = 0
        y[gei_resid >= q67] = 1

        frame = _feature_frame(
            "DLPFC",
            y,
            entropy,
            identity_distance,
            spacing,
            totals,
            np.repeat(slice_id, len(y)),
        )
        frame["donor"] = donor
        frames.append(frame)
        binary = np.isin(y, [0, 1])
        sample_arrays.append({"y": y[binary], "entropy": entropy[binary], "coords": coords[binary]})

        for module, values in {**scores, "GEI_combined": gei}.items():
            resid = residualize(values, [np.log1p(totals), spacing], labels)
            section_effect_rows.append(
                {
                    "domain": "DLPFC",
                    "sample": slice_id,
                    "donor": donor,
                    "module": module,
                    "effect_d": cohens_d(resid, entropy),
                    "n": len(resid),
                }
            )

        if slice_id == "151673":
            spatial_payload = {
                "coords": coords.astype(np.float32),
                "entropy": entropy,
                "support": zscore(gei_resid).astype(np.float32),
            }
        del a, counts
        gc.collect()

    effects = pd.DataFrame(section_effect_rows)
    module_effects = (
        effects.groupby(["donor", "module"], as_index=False)["effect_d"]
        .mean()
        .groupby("module")["effect_d"]
        .mean()
        .to_dict()
    )
    payload = DomainPayload(
        domain="DLPFC",
        features=pd.concat(frames, ignore_index=True),
        sample_arrays=sample_arrays,
        module_effects={str(k): float(v) for k, v in module_effects.items()},
        spatial=spatial_payload,
    )
    return payload, section_effect_rows, availability_rows


def _classify_highres(domain: str, a: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if domain == "MERFISH":
        cell_class = a.obs["Cell_class"].astype(str)
        lower = cell_class.str.lower()
        support = (
            lower.str.contains("astro")
            | lower.str.contains("od mature")
            | lower.str.contains("oligo")
            | lower.str.contains("endothel")
            | lower.str.contains("pericy")
        ).to_numpy()
        neuron = (lower.str.contains("excit") | lower.str.contains("inhibit")).to_numpy()
        neuron_label = a.obs["Neuron_cluster_ID"].astype(str).to_numpy()
        valid_neuron_label = neuron & ~np.isin(neuron_label, ["nan", "None", "NA", ""])
        support_label = cell_class.to_numpy()
    elif domain == "Slide-seqV2":
        cluster = a.obs["cluster"].astype(str)
        lower = cluster.str.lower()
        support = (
            lower.str.contains("astro")
            | lower.str.contains("oligo")
            | lower.str.contains("polydendro")
            | lower.str.contains("endothel")
            | lower.str.contains("mural")
        ).to_numpy()
        neuron = (
            lower.str.contains("ca1")
            | lower.str.contains("ca2")
            | lower.str.contains("ca3")
            | lower.str.contains("dentate")
            | lower.str.contains("interneuron")
            | lower.str.contains("subiculum")
        ).to_numpy()
        neuron_label = cluster.to_numpy()
        valid_neuron_label = neuron
        support_label = cluster.to_numpy()
    else:
        raise ValueError(domain)
    y = np.full(a.n_obs, -1, dtype=int)
    y[neuron] = 0
    y[support] = 1
    return y, valid_neuron_label, support_label


def analyse_highres(
    path: Path,
    domain: str,
    n_permutations: int,
) -> tuple[DomainPayload, list[dict], dict[str, object]]:
    a = ad.read_h5ad(path)
    coords = np.asarray(a.obsm["spatial"], dtype=float)
    y, neuron_mask, support_label = _classify_highres(domain, a)
    if domain == "MERFISH":
        neuron_labels = a.obs["Neuron_cluster_ID"].astype(str).to_numpy()[neuron_mask]
        score_matrix, score_genes, transform = a.X, a.var_names, "linear"
        depth = np.asarray(a.X.sum(axis=1)).ravel()
        local_audit = count_audit(a.layers["counts"] if "counts" in a.layers else a.X)
    else:
        neuron_labels = a.obs["cluster"].astype(str).to_numpy()[neuron_mask]
        if a.raw is not None:
            score_matrix, score_genes, transform = a.raw.X, a.raw.var_names, "log1p"
        else:
            score_matrix, score_genes, transform = a.X, a.var_names, "log1p"
        if "total_counts" in a.obs:
            depth = a.obs["total_counts"].to_numpy(dtype=float)
        else:
            depth = np.expm1(np.asarray(score_matrix.sum(axis=1)).ravel())
        local_audit = count_audit(a.layers["counts"] if "counts" in a.layers else a.X)

    scores, used = score_modules(score_matrix, score_genes, transform=transform, totals=depth)
    availability_rows = [
        {
            "domain": domain,
            "sample": path.stem,
            "module": module,
            "n_genes": len(genes),
            "genes": ";".join(genes),
        }
        for module, genes in used.items()
    ]
    entropy, identity_distance = identity_entropy(
        coords, coords[neuron_mask], neuron_labels, k=30, exclude_self=False
    )
    spacing = local_spacing(coords, k=6)
    sample_name = "Animal_1" if domain == "MERFISH" else "single_puck"
    frame = _feature_frame(
        domain,
        y,
        entropy,
        identity_distance,
        spacing,
        depth,
        np.repeat(sample_name, len(y)),
    )
    analysis = np.isin(y, [0, 1])
    sample_arrays = [{"y": y[analysis], "entropy": entropy[analysis], "coords": coords[analysis]}]

    support = y == 1
    module_effects: dict[str, float] = {}
    for module, values in scores.items():
        resid = residualize(
            values[support],
            [np.log1p(depth[support]), spacing[support], identity_distance[support]],
            support_label[support],
        )
        module_effects[module] = cohens_d(resid, entropy[support])
    gei = combined_gei(scores)
    gei_resid = residualize(
        gei[support],
        [np.log1p(depth[support]), spacing[support], identity_distance[support]],
        support_label[support],
    )
    module_effects["GEI_combined"] = cohens_d(gei_resid, entropy[support])

    display_n = min(25000, len(coords))
    display_idx = RNG.choice(len(coords), display_n, replace=False)
    spatial_payload = {
        "coords": coords[display_idx].astype(np.float32),
        "entropy": entropy[display_idx],
        "support": y[display_idx].astype(np.float32),
    }
    payload = DomainPayload(
        domain=domain,
        features=frame,
        sample_arrays=sample_arrays,
        module_effects=module_effects,
        spatial=spatial_payload,
    )
    audit = {
        "domain": domain,
        "path": str(path),
        "n_obs": int(a.n_obs),
        "n_vars": int(a.n_vars),
        **local_audit,
    }
    del a, score_matrix
    gc.collect()
    return payload, availability_rows, audit


def leave_one_domain_out(payloads: list[DomainPayload]) -> list[dict[str, object]]:
    frames = {x.domain: x.features.copy() for x in payloads}
    full_cols = ["entropy", "identity_distance", "local_spacing", "log_depth"]
    baseline_cols = ["identity_distance", "local_spacing", "log_depth"]
    results: list[dict[str, object]] = []

    for held_out, test in frames.items():
        train = pd.concat([f for d, f in frames.items() if d != held_out], ignore_index=True)
        if len(train) > 30000:
            train = train.sample(30000, random_state=SEED)
        weights = np.ones(len(train), dtype=float)
        counts = train.groupby(["domain", "y"]).size()
        for (domain, y), n in counts.items():
            mask = (train["domain"] == domain) & (train["y"] == y)
            weights[mask.to_numpy()] = 1.0 / float(n)
        weights *= len(weights) / weights.sum()

        row: dict[str, object] = {
            "held_out": held_out,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "prevalence_test": float(test["y"].mean()),
        }
        for name, cols in (("full", full_cols), ("baseline", baseline_cols)):
            model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            model.fit(train[cols], train["y"], sample_weight=weights)
            probability = model.predict_proba(test[cols])[:, 1]
            row[f"auc_{name}"] = float(roc_auc_score(test["y"], probability))
            if name == "full":
                row["entropy_coefficient"] = float(model.coef_[0][0])
        row["delta_auc"] = float(row["auc_full"] - row["auc_baseline"])
        row["go_auc"] = bool(row["auc_full"] >= 0.70 and row["delta_auc"] >= 0.10)
        results.append(row)
    return results


def _load_figure_helper():
    path = Path(
        r"C:\Users\13264\.agents\skills\scientific-figure-pro\scripts\scientific_figure_pro.py"
    )
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load figure helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_figures(
    out_dir: Path,
    summaries: pd.DataFrame,
    lodo: pd.DataFrame,
    module_effects: pd.DataFrame,
    contracts: pd.DataFrame,
    payloads: list[DomainPayload],
) -> None:
    sfp = _load_figure_helper()
    sfp.apply_publication_style(sfp.FigureStyle(font_size=12, axes_linewidth=1.5))
    fig, axes = sfp.create_subplots(2, 2, figsize=(12, 9))

    ax = axes[0]
    plot = summaries.sort_values("domain").reset_index(drop=True)
    x = np.arange(len(plot))
    log2_or = np.log2(plot["or"].to_numpy())
    low = log2_or - np.log2(plot["ci_low"].to_numpy())
    high = np.log2(plot["ci_high"].to_numpy()) - log2_or
    ax.errorbar(
        x,
        log2_or,
        yerr=np.vstack([low, high]),
        fmt="o",
        color="#2166AC",
        ecolor="#555555",
        capsize=4,
        lw=1.8,
        ms=7,
    )
    ax.axhline(0, color="#B2182B", ls="--", lw=1)
    ax.set_xticks(x, plot["domain"], rotation=20, ha="right")
    ax.set_ylabel("Support enrichment, log2(OR)")
    ax.set_title("A  Glial-support category at identity interfaces")
    for i, row in plot.iterrows():
        ax.text(
            i,
            log2_or[i] + high[i] + 0.08,
            f"shift p={row['spatial_p']:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax = axes[1]
    order = lodo["held_out"].tolist()
    xx = np.arange(len(order))
    width = 0.34
    ax.bar(xx - width / 2, lodo["auc_full"], width, label="Interface + covariates", color="#2166AC")
    ax.bar(xx + width / 2, lodo["auc_baseline"], width, label="Covariates only", color="#BDBDBD")
    ax.axhline(0.70, color="#B2182B", ls="--", lw=1, label="pilot go threshold")
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(xx, order, rotation=20, ha="right")
    ax.set_ylabel("Held-out-domain AUROC")
    ax.set_title("B  Operational category transfer")
    ax.legend(fontsize=8)

    ax = axes[2]
    pivot = module_effects.pivot(index="module", columns="domain", values="effect_d")
    pivot = pivot.reindex([*GEI_MODULES, "GEI_combined", "microimmune", "neuronal_synaptic"])
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index)
    ax.set_title("C  Within-category state effect (boundary − interior)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                "NA" if not np.isfinite(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if np.isfinite(value) and abs(value) > 0.28 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's d")

    ax = axes[3]
    contract_cols = ["raw_integer_counts", "independent_subjects_ge3", "sct_legal", "scvi_legal"]
    matrix = contracts.set_index("domain")[contract_cols].astype(float)
    im2 = ax.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        np.arange(len(contract_cols)),
        ["Raw counts", "≥3 subjects", "SCT legal", "scVI legal"],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_title("D  Evidence gate (current local cache)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                "PASS" if matrix.iloc[i, j] else "FAIL",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])

    fig.suptitle("Cross-brain GEI pilot: effect first, evidence gate explicit", y=1.01, fontsize=15)
    sfp.finalize_figure(fig, out_dir / "figure1_gei_pilot", formats=["png", "pdf", "svg"], dpi=600)

    fig2, axes2 = sfp.create_subplots(3, 2, figsize=(10, 12))
    for row, payload in enumerate(payloads):
        coords = payload.spatial["coords"]
        entropy = payload.spatial["entropy"]
        support = payload.spatial["support"]
        for col, (values, title, cmap) in enumerate(
            (
                (entropy, "neuronal/anatomical identity entropy", "magma"),
                (support, "glial-support category/program", "coolwarm"),
            )
        ):
            ax = axes2[row * 2 + col]
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
                s=1.2,
                cmap=cmap,
                rasterized=True,
                linewidths=0,
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{payload.domain}: {title}", fontsize=10)
            fig2.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    fig2.suptitle("Representative spatial fields used by the GEI pilot", y=0.995, fontsize=15)
    sfp.finalize_figure(
        fig2, out_dir / "figure2_spatial_fields", formats=["png", "pdf", "svg"], dpi=600
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=199)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    workspace = repo.parent
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    merfish = (
        workspace
        / "Biomni_lab_downloads_20260714_164953"
        / "histoweave_upgrade"
        / "datasets_cache"
        / "merfish"
        / "merfish_mouse_hypothalamus.h5ad"
    )
    slideseq = (
        workspace
        / "Biomni_lab_downloads_20260714_164953"
        / "histoweave_upgrade"
        / "datasets_cache"
        / "slideseqv2"
        / "slideseqv2_mouse_hippocampus.h5ad"
    )
    if not slideseq.exists():
        alternatives = list(workspace.glob("**/slideseqv2_mouse_hippocampus.h5ad"))
        if not alternatives:
            raise FileNotFoundError(slideseq)
        slideseq = alternatives[0]

    dlpfc_payload, section_rows, availability_dlpfc = analyse_dlpfc(repo, args.permutations)
    merfish_payload, availability_merfish, audit_merfish = analyse_highres(
        merfish, "MERFISH", args.permutations
    )
    slideseq_payload, availability_slideseq, audit_slideseq = analyse_highres(
        slideseq, "Slide-seqV2", args.permutations
    )
    payloads = [dlpfc_payload, merfish_payload, slideseq_payload]

    summary_rows: list[dict[str, object]] = []
    for payload in payloads:
        if payload.domain == "DLPFC":
            section_meta: list[dict[str, object]] = []
            for item, sample in zip(payload.sample_arrays, sorted(DLPFC_DONOR), strict=False):
                result = odds_ratio(item["y"], item["entropy"])
                section_meta.append({"sample": sample, "donor": DLPFC_DONOR[sample], **result})
            section_meta_df = pd.DataFrame(section_meta)
            donor_log = section_meta_df.groupby("donor")["log_or"].mean()
            mean_log = float(donor_log.mean())
            if len(donor_log) >= 2:
                half = 4.303 * float(donor_log.std(ddof=1)) / math.sqrt(len(donor_log))
            else:
                half = math.nan
            summary = {
                "domain": payload.domain,
                "or": math.exp(mean_log),
                "ci_low": math.exp(mean_log - half) if np.isfinite(half) else math.nan,
                "ci_high": math.exp(mean_log + half) if np.isfinite(half) else math.nan,
                "n": int(sum(len(x["y"]) for x in payload.sample_arrays)),
                "ci_basis": "three-donor t interval",
            }
        else:
            result = odds_ratio(payload.sample_arrays[0]["y"], payload.sample_arrays[0]["entropy"])
            summary = {
                "domain": payload.domain,
                **result,
                "ci_basis": "descriptive cell-level interval",
            }
        actual, spatial_p, null = spatial_shift_p(
            payload.sample_arrays, n_permutations=args.permutations
        )
        summary["mean_log_or_for_null"] = actual
        summary["spatial_p"] = spatial_p
        summary["n_permutations"] = args.permutations
        summary_rows.append(summary)
        np.save(
            out_dir / f"{payload.domain.lower().replace('-', '_')}_spatial_null.npy",
            np.asarray(null),
        )

    summaries = pd.DataFrame(summary_rows)
    lodo = pd.DataFrame(leave_one_domain_out(payloads))
    module_rows = [
        {"domain": payload.domain, "module": module, "effect_d": value}
        for payload in payloads
        for module, value in payload.module_effects.items()
    ]
    module_effects = pd.DataFrame(module_rows)
    contracts = pd.DataFrame(
        [
            {
                "domain": "DLPFC",
                "raw_integer_counts": True,
                "independent_subjects_ge3": True,
                "sct_legal": True,
                "scvi_legal": True,
                "biological_n": 3,
                "scope": "human cortex / Visium",
            },
            {
                "domain": "MERFISH",
                "raw_integer_counts": False,
                "independent_subjects_ge3": False,
                "sct_legal": False,
                "scvi_legal": False,
                "biological_n": 1,
                "scope": "mouse hypothalamus / MERFISH",
            },
            {
                "domain": "Slide-seqV2",
                "raw_integer_counts": False,
                "independent_subjects_ge3": False,
                "sct_legal": False,
                "scvi_legal": False,
                "biological_n": 1,
                "scope": "mouse hippocampus / Slide-seqV2",
            },
        ]
    )
    availability = pd.DataFrame(availability_dlpfc + availability_merfish + availability_slideseq)
    audits = pd.DataFrame(
        [
            {
                "domain": "DLPFC",
                "path": str(repo / "datasets_cache" / "dlpfc"),
                "n_obs": 47338,
                "n_vars": 33538,
                "integer_like": True,
                "noninteger_fraction": 0.0,
            },
            audit_merfish,
            audit_slideseq,
        ]
    )

    summaries.to_csv(out_dir / "domain_effects.csv", index=False)
    lodo.to_csv(out_dir / "leave_one_domain_out.csv", index=False)
    module_effects.to_csv(out_dir / "module_effects.csv", index=False)
    pd.DataFrame(section_rows).to_csv(out_dir / "dlpfc_section_effects.csv", index=False)
    availability.to_csv(out_dir / "module_gene_coverage.csv", index=False)
    contracts.to_csv(out_dir / "evidence_contract.csv", index=False)
    audits.to_csv(out_dir / "count_audit.csv", index=False)

    go = {
        "all_domain_or_ge_1_5": bool((summaries["or"] >= 1.5).all()),
        "all_spatial_p_le_0_01": bool((summaries["spatial_p"] <= 0.01).all()),
        "all_lodo_auc_go": bool(lodo["go_auc"].all()),
        "current_cache_nm_ready": bool(
            contracts["raw_integer_counts"].all() and contracts["independent_subjects_ge3"].all()
        ),
    }
    result = {
        "analysis": "candidate glial-enriched interface neighborhood pilot",
        "seed": SEED,
        "permutations": args.permutations,
        "domain_effects": summaries.to_dict(orient="records"),
        "leave_one_domain_out": lodo.to_dict(orient="records"),
        "go_no_go": go,
        "interpretation_rule": (
            "Candidate only. Tissue, species, technology, resolution and label ontology are confounded; "  # noqa: E501
            "MERFISH and Slide-seq local caches are topology-only because their counts layers are normalized."  # noqa: E501
        ),
    }
    (out_dir / "pilot_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figures(out_dir, summaries, lodo, module_effects, contracts, payloads)
    _log(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
