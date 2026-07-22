#!/usr/bin/env python
"""Expand the real independent-study panel to ≥15 units and re-score personalisation.

Sources of *real* independent units
-----------------------------------
Already in ``protocol_endpoints_results/multisource_landscape.json`` (11):
  * 3 DLPFC biological donors
  * 5 external multi-platform studies
  * 3 cross-platform studies (MERFISH Moffitt / Slide-seqV2 / Xenium)

Additional real units prepared here (target +4…):
  * ``xenium_human_lymph_node`` — local pathology-labelled bundle
  * ``slideseq_puck_200115_08`` — Stickels-lineage annotated puck (research cache)
  * Squidpy public corpora with published annotations (seqFISH, MIBI-TOF, 4i, IMC,
    Visium H&E, Visium fluorescence) when importable

Each new unit is scored with the shared sklearn domain method set (oracle-K from
``domain_truth`` cardinality for fair ARI), then merged into the independent
personalisation panel **without synthetic labs** (unless ``--include-synthetic``).

Usage
-----
python scripts/expand_real_independent_studies.py
python scripts/expand_real_independent_studies.py
  --min-real 15 --out-dir independent_personalisation_results
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.features import (  # noqa: E402
    RECOMMENDATION_FEATURE_ORDER,
    extract_features,
    feature_vector,
)
from histoweave.benchmark.independent_personalisation import (  # noqa: E402
    DEFAULT_METHODS,
    IndependentStudyUnit,
    aggregate_units_to_landscape,
    cross_lab_reproducibility_report,
    default_independent_units_from_multisource,
    evaluate_personalisation_policies,
    merge_unit_landscapes,
    summarise_policies,
    synthetic_lab_units,
    write_independent_personalisation_bundle,
)
from histoweave.benchmark.landscape import LandscapeResult, run_task_landscape  # noqa: E402
from histoweave.benchmark.task_contract import AnalysisTask, GroundTruthKind  # noqa: E402
from histoweave.data import SpatialTable  # noqa: E402
from histoweave.plugins import MethodCategory  # noqa: E402

LOG = logging.getLogger("histoweave.expand_real_independent_studies")

MAX_CELLS = 6_000
METHODS = list(DEFAULT_METHODS)


def _load_multisource(path: Path) -> LandscapeResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LandscapeResult(
        performance={str(k): dict(v) for k, v in payload["performance"].items()},
        features={
            str(k): np.asarray(v, dtype=float) for k, v in (payload.get("features") or {}).items()
        },
        embedding={str(k): (0.0, 0.0) for k in payload["performance"]},
        best_method={str(k): v for k, v in (payload.get("best_method") or {}).items()},
        niches={},
        timings={str(k): dict(v) for k, v in (payload.get("timings") or {}).items()},
        feature_order=list(payload.get("feature_order") or RECOMMENDATION_FEATURE_ORDER),
        method_count=int(payload.get("method_count") or 0),
        dataset_count=len(payload["performance"]),
        task=str(payload.get("task") or AnalysisTask.SPATIAL_DOMAIN.value),
        metric=str(payload.get("metric") or "ARI"),
        higher_is_better=bool(payload.get("higher_is_better", True)),
        dataset_meta={str(k): dict(v) for k, v in (payload.get("dataset_meta") or {}).items()},
    )


def _subsample_adata(adata, *, max_cells: int, seed: int, label_col: str):

    if adata.n_obs <= max_cells:
        return adata
    labels = adata.obs[label_col].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        quota = max(1, int(round(len(idx) / len(labels) * max_cells)))
        take = min(len(idx), quota)
        keep.append(rng.choice(idx, size=take, replace=False))
    merged = np.unique(np.concatenate(keep))
    if len(merged) > max_cells:
        merged = np.sort(rng.choice(merged, size=max_cells, replace=False))
    elif len(merged) < max_cells:
        rest = np.setdiff1d(np.arange(adata.n_obs), merged)
        if len(rest):
            extra = rng.choice(rest, size=min(max_cells - len(merged), len(rest)), replace=False)
            merged = np.sort(np.concatenate([merged, extra]))
    return adata[merged].copy()


def _adata_to_table(adata, *, unit_id: str, platform: str) -> tuple[SpatialTable, int]:
    if "domain_truth" not in adata.obs:
        raise ValueError(f"{unit_id}: missing domain_truth")
    truth = adata.obs["domain_truth"].astype(str)
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
    else:
        counts = adata.X
    X = np.asarray(counts.todense()) if hasattr(counts, "todense") else np.asarray(counts)
    X = np.asarray(X, dtype=np.float32)
    if "spatial" not in adata.obsm:
        raise ValueError(f"{unit_id}: missing obsm['spatial']")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if spatial.ndim != 2 or spatial.shape[1] < 2:
        raise ValueError(f"{unit_id}: spatial must be (n, ≥2)")
    spatial = spatial[:, :2]
    obs = pd.DataFrame(
        {"domain_truth": pd.Categorical(truth.to_numpy())},
        index=adata.obs_names.astype(str),
    )
    var = pd.DataFrame(index=adata.var_names.astype(str))
    tab = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm={"spatial": spatial},
        uns={
            "slice_id": unit_id,
            "platform": platform,
            "assay": platform,
            "study_group": unit_id,
        },
        layers={"counts": X},
    )
    n_domains = int(pd.Categorical(truth).categories.size)
    return tab, n_domains


def _hvg_subset(adata, n_top: int = 2000, seed: int = 0):
    import scanpy as sc

    if adata.n_vars <= n_top:
        return adata
    tmp = adata.copy()
    # Prefer counts for HVG when available.
    if "counts" in tmp.layers:
        tmp.X = tmp.layers["counts"]
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(
        tmp, n_top_genes=min(n_top, tmp.n_vars), flavor="seurat", subset=False
    )
    if "highly_variable" in tmp.var:
        genes = tmp.var_names[tmp.var["highly_variable"].to_numpy()].tolist()
    else:
        # Fallback: high mean genes
        means = np.asarray(tmp.X.mean(axis=0)).ravel()
        genes = tmp.var_names[np.argsort(-means)[:n_top]].tolist()
    return adata[:, genes].copy()


def load_local_lymph_node(repo: Path) -> tuple[str, SpatialTable, int, dict[str, Any]] | None:
    path = repo / "datasets_cache" / "xenium" / "xenium_human_lymph_node.h5ad"
    if not path.is_file():
        path = repo / "datasets_cache" / "xenium_human_lymph_node" / "xenium_human_lymph_node.h5ad"
    if not path.is_file():
        return None
    import anndata as ad

    adata = ad.read_h5ad(path)
    adata = _subsample_adata(adata, max_cells=MAX_CELLS, seed=0, label_col="domain_truth")
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X
    tab, k = _adata_to_table(adata, unit_id="xenium_human_lymph_node", platform="xenium")
    meta = {
        "platform": "xenium",
        "independence_class": "external_study",
        "study": "10x_xenium_prime_lymph_node",
        "paper": "10x Xenium Prime Human Lymph Node pathology polygons",
        "source_path": str(path),
    }
    return "xenium_human_lymph_node", tab, k, meta


def load_local_slideseq_puck(repo: Path) -> tuple[str, SpatialTable, int, dict[str, Any]] | None:
    path = (
        repo
        / "research"
        / "cross_tissue_niches"
        / "slideseq_raw"
        / "results"
        / "Puck_200115_08_raw_counts_annotated.h5ad"
    )
    if not path.is_file():
        return None
    import anndata as ad

    adata = ad.read_h5ad(path)
    # Drop unannotated for cleaner domain recovery signal.
    keep = ~adata.obs["domain_truth"].astype(str).isin({"unannotated", "nan", ""})
    adata = adata[keep.to_numpy()].copy()
    adata.obsm["spatial"] = np.column_stack(
        [adata.obs["x"].to_numpy(dtype=float), adata.obs["y"].to_numpy(dtype=float)]
    )
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X
    adata = _subsample_adata(adata, max_cells=MAX_CELLS, seed=1, label_col="domain_truth")
    adata = _hvg_subset(adata, n_top=2000)
    tab, k = _adata_to_table(adata, unit_id="slideseq_puck_200115_08", platform="slideseqv2")
    meta = {
        "platform": "slideseqv2",
        "independence_class": "external_study",
        "study": "Stickels2021_SlideSeqV2_Puck_200115_08",
        "paper": "Stickels et al. Slide-seqV2 hippocampus puck 200115_08",
        "source_path": str(path),
        "notes": "Distinct puck from 7x15 squidpy slideseqv2 cache when both present",
    }
    return "slideseq_puck_200115_08", tab, k, meta


def _prepare_squidpy_unit(
    name: str,
    loader,
    *,
    label_col: str,
    platform: str,
    study: str,
    drop_labels: tuple[str, ...] = (),
    seed: int = 0,
) -> tuple[str, SpatialTable, int, dict[str, Any]] | None:
    try:
        adata = loader()
    except Exception as exc:  # network / missing dep
        LOG.warning("squidpy dataset %s unavailable: %s", name, exc)
        return None
    adata.var_names = pd.Index(adata.var_names.astype(str))
    if not adata.var_names.is_unique:
        adata.var_names = pd.Index([f"{n}__{i}" for i, n in enumerate(adata.var_names)])
    if label_col not in adata.obs.columns:
        LOG.warning(
            "squidpy dataset %s missing label %s; cols=%s", name, label_col, list(adata.obs.columns)
        )
        return None
    lab = adata.obs[label_col].astype(str)
    mask = ~lab.isin(set(drop_labels) | {"nan", "NA", "None", ""})
    adata = adata[mask.to_numpy()].copy()
    adata.obs["domain_truth"] = adata.obs[label_col].astype(str).values
    if "spatial" not in adata.obsm:
        # some use X_spatial
        for key in ("spatial", "X_spatial"):
            if key in adata.obsm:
                adata.obsm["spatial"] = np.asarray(adata.obsm[key])
                break
        else:
            LOG.warning("squidpy dataset %s has no spatial coordinates", name)
            return None
    # counts layer
    X = adata.X
    dense = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    int_frac = float(np.mean(np.isclose(dense, np.round(dense)))) if dense.size else 0.0
    if int_frac >= 0.5 or float(np.nanmax(dense)) > 50:
        counts = np.clip(np.rint(dense), 0, None)
    else:
        counts = np.clip(np.expm1(dense), 0, None)
    adata.layers["counts"] = counts.astype(np.float32)
    adata = _subsample_adata(adata, max_cells=MAX_CELLS, seed=seed, label_col="domain_truth")
    if adata.n_vars > 2000:
        try:
            adata = _hvg_subset(adata, n_top=2000)
        except Exception:
            # protein panels etc. — keep all features
            pass
    unit_id = f"squidpy_{name}"
    tab, k = _adata_to_table(adata, unit_id=unit_id, platform=platform)
    if k < 2:
        LOG.warning("squidpy dataset %s has <2 domains after filtering", name)
        return None
    meta = {
        "platform": platform,
        "independence_class": "external_study",
        "study": study,
        "paper": f"squidpy.datasets.{name}",
        "label_col": label_col,
        "notes": "Published annotation used as domain proxy for study-level personalisation panel",
    }
    return unit_id, tab, k, meta


def _load_cached_squidpy_h5ad(name: str):
    """Prefer on-disk squidpy/scverse cache to avoid re-downloading."""
    import anndata as ad

    candidates = [
        ROOT / "data" / "anndata" / f"{name}.h5ad",
        ROOT / "data" / "anndata" / f"{name}_adata.h5ad",
        Path.home() / ".cache" / "squidpy" / f"{name}.h5ad",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1000:
            LOG.info("loading cached squidpy h5ad %s", path)
            return ad.read_h5ad(path)
    return None


def load_squidpy_units() -> list[tuple[str, SpatialTable, int, dict[str, Any]]]:
    if importlib.util.find_spec("squidpy") is None:
        LOG.warning("squidpy not installed; will use cached h5ad only if present")

    # name -> (label_col, platform, study, drops, seed, loader_attr)
    # Skip merfish/slideseqv2 by default — already represented as platform_* units.
    specs = [
        # Prefer fully-cached public corpora. Large Visium demos (≈300MB+) are
        # optional — only loaded when a complete local h5ad already exists.
        (
            "seqfish",
            "celltype_mapped_refined",
            "seqfish",
            "Lohoff2022_seqFISH_embryo",
            (),
            2,
            "seqfish",
        ),
        ("mibitof", "Cluster", "mibi", "Hartmann2020_MIBI_TOF", (), 3, "mibitof"),
        ("four_i", "cluster", "four_i", "Gut2018_4i", (), 4, "four_i"),
        ("imc", "cell type", "imc", "Jackson2020_IMC", (), 5, "imc"),
    ]
    out: list[tuple[str, SpatialTable, int, dict[str, Any]]] = []
    for cache_name, label, platform, study, drops, seed, attr in specs:

        def _make_loader(_cache=cache_name, _attr=attr):
            def _load():
                cached = _load_cached_squidpy_h5ad(_cache)
                if cached is not None:
                    return cached
                # Do not auto-download multi-hundred-MB demos during CI / eval.
                raise FileNotFoundError(
                    f"no local cache for {_cache}; download via squidpy.datasets.{_attr}() first"
                )

            return _load

        short = cache_name.replace("_adata", "")
        item = _prepare_squidpy_unit(
            short,
            _make_loader(),
            label_col=label,
            platform=platform,
            study=study,
            drop_labels=drops,
            seed=seed,
        )
        if item is None:
            for alt in ("cluster", "Cluster", "leiden", "cell_type", "cell type", "library_id"):
                if alt == label:
                    continue
                item = _prepare_squidpy_unit(
                    short,
                    _make_loader(),
                    label_col=alt,
                    platform=platform,
                    study=study,
                    drop_labels=drops,
                    seed=seed,
                )
                if item is not None:
                    break
        if item is not None:
            LOG.info("prepared squidpy unit %s k=%s n_obs=%s", item[0], item[2], item[1].n_obs)
            out.append(item)
    return out


def score_units(
    units: list[tuple[str, SpatialTable, int, dict[str, Any]]],
    *,
    methods: list[str],
    seed: int = 42,
) -> LandscapeResult:
    if not units:
        raise ValueError("no units to score")
    datasets = {uid: tab for uid, tab, _k, _m in units}
    n_map = {uid: k for uid, _t, k, _m in units}
    _meta_in = {uid: meta for uid, _t, _k, meta in units}

    def factory(data: SpatialTable) -> dict[str, Any]:
        sid = str(data.uns.get("slice_id") or "")
        return {"n_domains": int(n_map.get(sid, 3)), "random_state": seed}

    LOG.info("Scoring %d new real units × %d methods ...", len(datasets), len(methods))
    landscape = run_task_landscape(
        datasets,
        category=MethodCategory.DOMAIN_DETECTION,
        methods=methods,
        extra_params_factory=factory,
    )
    # Ensure target-free features + meta
    for uid, tab, _k, meta in units:
        feats = extract_features(tab, include_domain=False)
        landscape.features[uid] = feature_vector(feats, order=RECOMMENDATION_FEATURE_ORDER)
        row = {
            "platform": meta.get("platform"),
            "task": AnalysisTask.SPATIAL_DOMAIN.value,
            "ground_truth_kind": GroundTruthKind.SPATIAL_DOMAIN.value,
            "study_group": uid,
            "independence_class": meta.get("independence_class", "external_study"),
            "study": meta.get("study", uid),
            "member_datasets": [uid],
            "notes": meta.get("notes", ""),
            "paper": meta.get("paper", ""),
        }
        landscape.dataset_meta[uid] = row
    landscape.feature_order = list(RECOMMENDATION_FEATURE_ORDER)
    landscape.task = AnalysisTask.SPATIAL_DOMAIN.value
    return landscape


def landscape_to_units(landscape: LandscapeResult) -> list[IndependentStudyUnit]:
    units: list[IndependentStudyUnit] = []
    for name in landscape.dataset_order():
        meta = landscape.dataset_meta.get(name, {})
        units.append(
            IndependentStudyUnit(
                unit_id=name,
                independence_class=str(meta.get("independence_class") or "external_study"),  # type: ignore[arg-type]
                member_datasets=list(meta.get("member_datasets") or [name]),
                platform=meta.get("platform"),
                notes=str(meta.get("notes") or meta.get("study") or ""),
            )
        )
    return units


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multisource",
        type=Path,
        default=ROOT / "protocol_endpoints_results" / "multisource_landscape.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "independent_personalisation_results",
    )
    parser.add_argument("--min-real", type=int, default=15)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-floor", type=float, default=0.20)
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also append synthetic labs (not counted toward min-real)",
    )
    parser.add_argument(
        "--skip-squidpy",
        action="store_true",
        help="Only use local lymph-node + slideseq puck + existing multisource",
    )
    args = parser.parse_args(argv)

    if not args.multisource.is_file():
        LOG.error(
            "Missing multisource landscape at %s — run run_protocol_endpoints.py first",
            args.multisource,
        )
        return 1

    multi = _load_multisource(args.multisource)
    for _name, row in multi.performance.items():
        for m in METHODS:
            row.setdefault(m, float("nan"))

    base_units = default_independent_units_from_multisource(list(multi.performance))
    base_land = aggregate_units_to_landscape(multi, base_units, methods=METHODS)
    for u in base_units:
        base_land.dataset_meta.setdefault(u.unit_id, {})
        base_land.dataset_meta[u.unit_id]["independence_class"] = u.independence_class
        base_land.dataset_meta[u.unit_id]["member_datasets"] = u.member_datasets
        base_land.dataset_meta[u.unit_id]["platform"] = u.platform
    LOG.info("Base real independent units: %d", base_land.dataset_count)

    new_specs: list[tuple[str, SpatialTable, int, dict[str, Any]]] = []
    for loader in (load_local_lymph_node, load_local_slideseq_puck):
        item = loader(ROOT)
        if item is not None:
            LOG.info("local unit ready: %s", item[0])
            new_specs.append(item)

    if not args.skip_squidpy:
        new_specs.extend(load_squidpy_units())

    # Drop units already present under the same id
    new_specs = [s for s in new_specs if s[0] not in base_land.performance]
    # Avoid double-counting exact study when platform_* already covers squidpy merfish/slideseq
    existing_studies = {
        str(base_land.dataset_meta.get(u, {}).get("study") or u) for u in base_land.dataset_order()
    }
    filtered: list[tuple[str, SpatialTable, int, dict[str, Any]]] = []
    for spec in new_specs:
        study = str(spec[3].get("study") or spec[0])
        # Always keep lymph node and annotated puck even if related platforms exist
        if study in existing_studies and not spec[0].startswith(
            ("xenium_human_lymph", "slideseq_puck")
        ):
            LOG.info("skip %s — study %s already in panel", spec[0], study)
            continue
        filtered.append(spec)
    new_specs = filtered

    landscapes = [base_land]
    all_units = list(base_units)
    if new_specs:
        new_land = score_units(new_specs, methods=METHODS, seed=args.seed)
        landscapes.append(new_land)
        all_units.extend(landscape_to_units(new_land))
        LOG.info("Added %d newly scored real units", new_land.dataset_count)

    panel = merge_unit_landscapes(*landscapes) if len(landscapes) > 1 else landscapes[0]
    for name in panel.performance:
        for m in METHODS:
            panel.performance[name].setdefault(m, float("nan"))
        panel.dataset_meta.setdefault(name, {})
        panel.dataset_meta[name].setdefault("independence_class", "external_study")
        panel.dataset_meta[name].setdefault("task", AnalysisTask.SPATIAL_DOMAIN.value)
        panel.dataset_meta[name].setdefault(
            "ground_truth_kind", GroundTruthKind.SPATIAL_DOMAIN.value
        )

    real_ids = [
        u
        for u in panel.dataset_order()
        if str(panel.dataset_meta.get(u, {}).get("independence_class")) != "synthetic_lab"
    ]
    n_real = len(real_ids)
    LOG.info("Real independent units now: %d (target ≥%d)", n_real, args.min_real)
    LOG.info("Real unit ids: %s", ", ".join(sorted(real_ids)))

    if args.include_synthetic:
        synth_land, synth_units = synthetic_lab_units(seed=args.seed, methods=METHODS)
        panel = merge_unit_landscapes(panel, synth_land)
        all_units.extend(synth_units)

    # Persist expanded real landscape snapshot
    args.out_dir.mkdir(parents=True, exist_ok=True)
    expanded_path = args.out_dir / "real_independent_unit_landscape.json"
    expanded_path.write_text(
        json.dumps(
            {
                "n_real_units": n_real,
                "real_unit_ids": sorted(real_ids),
                "performance": panel.performance,
                "features": {k: np.asarray(v).tolist() for k, v in panel.features.items()},
                "dataset_meta": panel.dataset_meta,
                "methods": METHODS,
                "feature_order": panel.feature_order,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    if n_real < args.min_real:
        LOG.error(
            "Only %d real independent units (< %d). Install squidpy and re-run, "
            "or add more labelled public h5ad bundles.",
            n_real,
            args.min_real,
        )
        # Still write partial artefacts for debugging
        policy_rows = evaluate_personalisation_policies(
            panel,
            methods=METHODS,
            k_neighbours=3,
            confidence_floor=args.confidence_floor,
            proxy_advantage=0.02,
        )
        summary = summarise_policies(
            policy_rows,
            noninferior_margin=args.margin,
            min_queries=args.min_real,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        cross = cross_lab_reproducibility_report(
            panel, policy_rows, methods=METHODS, n_boot=args.n_boot, seed=args.seed
        )
        write_independent_personalisation_bundle(
            args.out_dir,
            landscape=panel,
            units=all_units if all_units else landscape_to_units(panel),
            policy_rows=policy_rows,
            summary=summary,
            cross_lab=cross,
        )
        return 2

    policy_rows = evaluate_personalisation_policies(
        panel,
        methods=METHODS,
        k_neighbours=3,
        confidence_floor=args.confidence_floor,
        proxy_advantage=0.02,
    )
    summary = summarise_policies(
        policy_rows,
        noninferior_margin=args.margin,
        min_queries=args.min_real,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    # Annotate real-only count
    summary["n_real_independent_units"] = n_real
    summary["real_unit_ids"] = sorted(real_ids)
    summary["meets_real_query_target"] = n_real >= args.min_real

    cross = cross_lab_reproducibility_report(
        panel, policy_rows, methods=METHODS, n_boot=args.n_boot, seed=args.seed
    )
    paths = write_independent_personalisation_bundle(
        args.out_dir,
        landscape=panel,
        units=all_units if all_units else landscape_to_units(panel),
        policy_rows=policy_rows,
        summary=summary,
        cross_lab=cross,
    )
    # Rewrite summary with real counts explicit
    summary_path = args.out_dir / "independent_personalisation_summary.json"
    master = json.loads(summary_path.read_text(encoding="utf-8"))
    master["n_real_independent_units"] = n_real
    master["meets_real_query_target"] = n_real >= args.min_real
    master["real_unit_ids"] = sorted(real_ids)
    summary_path.write_text(json.dumps(master, indent=2), encoding="utf-8")

    LOG.info(
        "DONE real=%d gated=%.4f global=%.4f primary_NI=%s superior=%s",
        n_real,
        summary.get("mean_gated_regret"),
        summary.get("mean_global_best_regret"),
        summary.get("primary_noninferior"),
        summary.get("primary_superior"),
    )
    LOG.info("Artifacts: %s", {k: str(v) for k, v in paths.items()})
    return 0 if summary.get("primary_noninferior") else 3


if __name__ == "__main__":
    raise SystemExit(main())
