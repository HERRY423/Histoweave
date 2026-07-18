"""P1 tests: landscape merge, dataset contracts, uncertainty report, leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from histoweave.benchmark import (
    AnalysisTask,
    GroundTruthKind,
    MethodRecommender,
    attach_dataset_meta,
    landscape_from_long_csv,
    merge_landscapes,
    meta_from_registry,
    validate_landscape_contracts,
    write_landscape_json,
)
from histoweave.data import SpatialTable
from histoweave.datasets import get_dataset, list_datasets, registry_summary
from histoweave.datasets.real import DatasetEntry
from histoweave.report.report import _build_context, _collect_method_predictions


def test_registry_covers_multiple_platforms_and_tissues():
    summary = registry_summary()
    assert summary["n_datasets"] >= 15
    assert summary["n_assays"] >= 3
    assert summary["n_tissues"] >= 3
    assert "spatial_domain" in summary["tasks"]
    # Xenium breast is cell_type task, not spatial_domain.
    breast = get_dataset("xenium_breast_cancer")
    assert breast.analysis_task == "cell_type"
    assert breast.ground_truth_kind == "cell_type"
    breast.task_contract()  # must validate
    dlpfc = get_dataset("dlpfc_151673")
    assert dlpfc.analysis_task == "spatial_domain"
    assert dlpfc.study == "Maynard2021_spatialLIBD"
    meta = dlpfc.to_dataset_meta()
    assert meta["platform"] == "visium"
    assert meta["ground_truth_kind"] == "spatial_domain"


def test_list_datasets_filters_by_task():
    domain = list_datasets(task="spatial_domain", has_ground_truth=True)
    cell = list_datasets(task="cell_type")
    assert any(row["name"].startswith("dlpfc_") for row in domain)
    assert any(row["name"] == "xenium_breast_cancer" for row in cell)
    assert all(row["analysis_task"] == "spatial_domain" for row in domain)


def test_landscape_from_long_csv_and_merge(tmp_path):
    baseline = tmp_path / "base.csv"
    baseline.write_text(
        "dataset,config,method,seed,ari,seconds\n"
        "151673,kmeans@sw0.0,kmeans,42,0.20,1.0\n"
        "151673,kmeans@sw0.8,kmeans,42,0.30,1.1\n"
        "151674,kmeans@sw0.0,kmeans,42,0.22,1.0\n"
        "151674,kmeans@sw0.8,kmeans,42,0.28,1.0\n",
        encoding="utf-8",
    )
    sota = tmp_path / "sota.csv"
    sota.write_text(
        "dataset,method,seed,ari,seconds,status\n"
        "151673,spagcn,42,0.40,12.0,success\n"
        "151674,spagcn,42,0.35,11.0,success\n"
        "151673,graphst,42,,9.0,failed\n",
        encoding="utf-8",
    )
    base_land = landscape_from_long_csv(baseline, prefer_config_as_method=True)
    sota_land = landscape_from_long_csv(sota, prefer_config_as_method=False)
    merged = merge_landscapes(base_land, sota_land, task=AnalysisTask.SPATIAL_DOMAIN.value)
    assert "kmeans@sw0.8" in merged.performance["151673"]
    assert "spagcn" in merged.performance["151673"]
    assert merged.performance["151673"]["spagcn"] == pytest.approx(0.40)
    # failed rows must not invent a finite score
    assert "graphst" not in merged.performance["151673"] or not np.isfinite(
        merged.performance["151673"].get("graphst", np.nan)
    )

    attach_dataset_meta(
        merged,
        meta_from_registry(
            merged.performance.keys(), name_map={"151673": "dlpfc_151673", "151674": "dlpfc_151674"}
        ),
        overwrite=True,
    )
    for meta in merged.dataset_meta.values():
        meta["task"] = AnalysisTask.SPATIAL_DOMAIN.value
        meta["ground_truth_kind"] = GroundTruthKind.SPATIAL_DOMAIN.value
        meta["label_key"] = "domain_truth"
    problems = validate_landscape_contracts(merged)
    assert problems == []

    out = write_landscape_json(merged, tmp_path / "kb.json")
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 3
    assert "dataset_meta" in raw
    rec = MethodRecommender(out, k_neighbours=1)
    # Knowledge base loads with platform priors available.
    assert rec._global_best_method is not None


def test_build_dlpfc_merged_landscape_from_repo():
    from histoweave.benchmark.landscape_io import build_dlpfc_merged_landscape

    root = Path(__file__).resolve().parents[1]
    csv_path = root / "5x15_spatial_aware" / "benchmark_long.csv"
    if not csv_path.exists():
        pytest.skip("5x15 benchmark CSV not present")
    landscape = build_dlpfc_merged_landscape(repo_root=root)
    assert landscape.dataset_count >= 5
    assert landscape.method_count >= 5
    assert landscape.task == "spatial_domain"
    assert all(
        landscape.dataset_meta[ds].get("ground_truth_kind") == "spatial_domain"
        for ds in landscape.performance
    )
    problems = validate_landscape_contracts(landscape)
    assert problems == []


def test_report_includes_boundary_uncertainty():
    rng = np.random.default_rng(0)
    n = 40
    coords = rng.normal(size=(n, 2))
    labels_a = np.array(["a"] * 20 + ["b"] * 20)
    labels_b = labels_a.copy()
    labels_b[15:25] = "b"  # disagree near the cut
    data = SpatialTable(
        X=np.abs(rng.normal(size=(n, 5))),
        obs=pd.DataFrame(
            {"domain": labels_a, "domain_alt": labels_b},
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
        obsm={"spatial": coords},
        uns={
            "method_predictions": {
                "method_a": labels_a,
                "method_b": labels_b,
            },
            "assay": "visium",
        },
    )
    preds = _collect_method_predictions(data)
    assert len(preds) == 2
    ctx = _build_context(data)
    assert ctx["uncertainty"] is not None
    assert ctx["uncertainty"]["summary"]["n_obs"] == n
    assert "boundary_uncertainty" in data.obs.columns


def test_xenium_breast_not_valid_as_spatial_domain_contract_when_kind_wrong():
    entry = DatasetEntry(
        name="bad",
        description="x",
        url="local://x",
        sha256="",
        assay="xenium",
        analysis_task="spatial_domain",
        ground_truth_kind="self_supervised",
        ground_truth={"domain_truth": "obs['leiden']"},
        label_key="leiden",
    )
    with pytest.raises(ValueError):
        entry.task_contract()
