"""Build the reproducible 7 x 15 cross-platform performance topography.

This formal experiment consumes the archived ``dataset_features.csv`` and
``performance_matrix_mean.csv`` artifacts. It therefore rebuilds without private h5ad
caches or Scanpy and records the PCA transform, method-selection margins, platform
response curves, and file checksums.

Usage
-----
python benchmark_crossplatform/build_topography.py --results-dir 7x15_cross_platform
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.scientific_figure_pro import (  # noqa: E402
    PALETTE,
    FigureStyle,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    make_trend,
)

_LOGGER = logging.getLogger(__name__)
_WEIGHT_PATTERN = re.compile(r"@sw(?P<weight>0(?:\.\d+)?|1(?:\.0+)?)$")
_PROHIBITED_FEATURE_TOKENS = {
    "truth",
    "target",
    "label",
    "domain",
    "prediction",
    "method",
    "score",
    "ari",
}
_WEIGHT_COLORS = {
    0.0: PALETTE["red_strong"],
    0.3: PALETTE["blue_main"],
    0.8: PALETTE["green_3"],
}
_PLATFORM_MARKERS = {"Visium": "o", "MERFISH": "s", "Slide-seqV2": "^", "Xenium": "D"}
_PLATFORM_COLORS = {
    "Visium": PALETTE["blue_main"],
    "MERFISH": PALETTE["red_strong"],
    "Slide-seqV2": PALETTE["green_3"],
    "Xenium": PALETTE["violet"],
}
_LABEL_OFFSETS = {
    "151507": (5, 6),
    "151669": (-20, 11),
    "151674": (-22, -13),
    "151673": (7, -13),
    "151670": (7, 8),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _load_inputs(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_path = results_dir / "dataset_features.csv"
    performance_path = results_dir / "performance_matrix_mean.csv"
    landscape_path = results_dir / "landscape.json"
    for path in (feature_path, performance_path):
        if not path.is_file():
            raise FileNotFoundError(f"required topography input is missing: {path}")

    features = pd.read_csv(feature_path)
    performance = pd.read_csv(performance_path)
    if "dataset" not in features or "dataset" not in performance:
        raise ValueError("both input tables must contain a 'dataset' column")
    if features["dataset"].duplicated().any() or performance["dataset"].duplicated().any():
        raise ValueError("dataset identifiers must be unique")
    if set(features["dataset"].astype(str)) != set(performance["dataset"].astype(str)):
        raise ValueError("feature and performance tables contain different datasets")
    if "platform" not in features:
        raise ValueError("dataset_features.csv must contain a platform column")

    metadata = (
        json.loads(landscape_path.read_text(encoding="utf-8")) if landscape_path.exists() else {}
    )
    feature_order = metadata.get("feature_order")
    if not isinstance(feature_order, list) or not feature_order:
        feature_order = [column for column in features if column not in {"dataset", "platform"}]
    missing = [column for column in feature_order if column not in features]
    if missing:
        raise ValueError(f"landscape feature_order columns are missing: {missing}")
    leaked = [
        column
        for column in feature_order
        if any(token in column.casefold().split("_") for token in _PROHIBITED_FEATURE_TOKENS)
    ]
    if leaked:
        raise ValueError(f"target-derived columns are forbidden in the topography: {leaked}")
    metadata["feature_order"] = feature_order
    return features, performance, metadata


def _pca_embedding(
    frame: pd.DataFrame, feature_order: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = frame[feature_order].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    medians = np.nanmedian(values, axis=0)
    missing_rows, missing_columns = np.where(~np.isfinite(values))
    if len(missing_rows):
        values[missing_rows, missing_columns] = medians[missing_columns]
    means = values.mean(axis=0)
    scales = values.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (values - means) / scales
    u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    coordinates = u[:, :2] * singular_values[:2]
    loadings = vt[:2].copy()
    for component in range(loadings.shape[0]):
        anchor = int(np.argmax(np.abs(loadings[component])))
        if loadings[component, anchor] < 0:
            loadings[component] *= -1
            coordinates[:, component] *= -1
    variance = singular_values**2
    explained = variance[:2] / variance.sum() if variance.sum() else np.zeros(2)
    return coordinates, means, scales, loadings, explained


def _parse_weight(method: str) -> float:
    match = _WEIGHT_PATTERN.search(method)
    if match is None:
        raise ValueError(f"method column does not encode a spatial weight: {method!r}")
    return float(match.group("weight"))


def _selection_table(
    features: pd.DataFrame, performance: pd.DataFrame, coordinates: np.ndarray
) -> pd.DataFrame:
    method_columns = [column for column in performance if column != "dataset"]
    if len(method_columns) < 2:
        raise ValueError("performance matrix needs at least two method columns")
    numeric = performance.set_index("dataset")[method_columns].apply(pd.to_numeric, errors="coerce")
    numeric.index = numeric.index.astype(str)
    by_dataset = features.set_index(features["dataset"].astype(str))
    rows: list[dict[str, Any]] = []
    for index, dataset in enumerate(features["dataset"].astype(str)):
        scores = numeric.loc[dataset].dropna().sort_values(ascending=False)
        if len(scores) < 2:
            raise ValueError(f"dataset {dataset!r} has fewer than two finite method scores")
        winner, runner_up = str(scores.index[0]), str(scores.index[1])
        score_range = float(scores.max() - scores.min())
        margin = float(scores.iloc[0] - scores.iloc[1])
        ambiguity = 1.0 - margin / score_range if score_range > 1e-12 else 1.0
        rows.append(
            {
                "dataset": dataset,
                "platform": str(by_dataset.loc[dataset, "platform"]),
                "pc1": float(coordinates[index, 0]),
                "pc2": float(coordinates[index, 1]),
                "winner": winner,
                "winner_family": winner.split("@", 1)[0],
                "winning_spatial_weight": _parse_weight(winner),
                "best_score": float(scores.iloc[0]),
                "runner_up": runner_up,
                "runner_up_score": float(scores.iloc[1]),
                "top2_margin": margin,
                "selection_ambiguity": float(np.clip(ambiguity, 0.0, 1.0)),
            }
        )
    return pd.DataFrame(rows)


def _platform_response(
    features: pd.DataFrame, performance: pd.DataFrame
) -> dict[str, dict[str, float]]:
    indexed = performance.set_index("dataset")
    indexed.index = indexed.index.astype(str)
    weights = sorted({_parse_weight(column) for column in indexed.columns})
    response: dict[str, dict[str, float]] = {}
    for platform, group in features.groupby("platform", sort=True):
        dataset_ids = group["dataset"].astype(str).tolist()
        response[str(platform)] = {}
        for weight in weights:
            columns = [column for column in indexed if _parse_weight(column) == weight]
            values = indexed.loc[dataset_ids, columns].to_numpy(dtype=float)
            response[str(platform)][f"{weight:g}"] = float(np.nanmean(values))
    return response


def _draw_figure(
    selection: pd.DataFrame,
    response: dict[str, dict[str, float]],
    explained: np.ndarray,
    output_base: Path,
) -> list[Path]:
    apply_publication_style(FigureStyle(font_size=11, axes_linewidth=1.4))
    fig, axes = create_subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    ax_map, ax_response = axes

    for row in selection.itertuples(index=False):
        color = _WEIGHT_COLORS.get(float(row.winning_spatial_weight), PALETTE["neutral"])
        marker = _PLATFORM_MARKERS.get(str(row.platform), "o")
        size = 70.0 + 520.0 * float(np.clip(row.best_score, 0.0, 1.0))
        ax_map.scatter(
            row.pc1,
            row.pc2,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.92,
            zorder=3,
        )
        ax_map.annotate(
            str(row.dataset),
            (row.pc1, row.pc2),
            xytext=_LABEL_OFFSETS.get(str(row.dataset), (5, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    ax_map.axhline(0, color="#D8D8D8", linewidth=0.8, zorder=1)
    ax_map.axvline(0, color="#D8D8D8", linewidth=0.8, zorder=1)
    ax_map.set_xlabel(f"Target-free feature PC1 ({100 * explained[0]:.1f}% variance)")
    ax_map.set_ylabel(f"Target-free feature PC2 ({100 * explained[1]:.1f}% variance)")
    ax_map.set_title("A  Cross-platform method-selection topography", loc="left", fontweight="bold")

    weight_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=f"winning spatial weight = {weight:g}",
        )
        for weight, color in sorted(_WEIGHT_COLORS.items())
    ]
    platform_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            color="#555555",
            markerfacecolor="white",
            markersize=7,
            label=platform,
        )
        for platform, marker in _PLATFORM_MARKERS.items()
        if platform in set(selection["platform"])
    ]
    first_legend = ax_map.legend(handles=weight_handles, loc="lower left", fontsize=8)
    ax_map.add_artist(first_legend)
    ax_map.legend(handles=platform_handles, loc="upper right", fontsize=8)

    platforms = sorted(response)
    weights = sorted(float(value) for value in next(iter(response.values())))
    series = [[response[platform][f"{weight:g}"] for weight in weights] for platform in platforms]
    make_trend(
        ax_response,
        weights,
        series,
        platforms,
        colors=[_PLATFORM_COLORS.get(platform, PALETTE["neutral"]) for platform in platforms],
        xlabel="Spatial weight",
        ylabel="Mean ARI across five clustering families",
    )
    ax_response.set_xticks(weights)
    ax_response.set_ylim(bottom=min(0.0, ax_response.get_ylim()[0]))
    ax_response.grid(axis="y", color="#E6E6E6", linewidth=0.7)
    ax_response.set_title(
        "B  Platform-specific spatial-weight response", loc="left", fontweight="bold"
    )
    return finalize_figure(fig, output_base, formats=("svg", "png"), dpi=300)


def build_topography(results_dir: str | Path) -> dict[str, Any]:
    """Build figures and machine-readable artifacts from archived benchmark outputs."""

    root = Path(results_dir).resolve()
    features, performance, landscape = _load_inputs(root)
    feature_order = list(map(str, landscape["feature_order"]))
    coordinates, means, scales, loadings, explained = _pca_embedding(features, feature_order)
    selection = _selection_table(features, performance, coordinates)
    response = _platform_response(features, performance)

    figure_paths = _draw_figure(
        selection,
        response,
        explained,
        root / "figures" / "platform_topography",
    )
    table_path = root / "platform_topography.csv"
    with table_path.open("w", encoding="utf-8", newline="\n") as handle:
        selection.to_csv(handle, index=False, float_format="%.10g", lineterminator="\n")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": landscape.get("task", "domain_detection"),
        "feature_schema": landscape.get("feature_schema", "histoweave.target_free.v1"),
        "feature_order": feature_order,
        "preprocessing": {
            "missing_values": "column_median",
            "scaling": "population_zscore",
            "centers": dict(zip(feature_order, means.tolist(), strict=True)),
            "scales": dict(zip(feature_order, scales.tolist(), strict=True)),
        },
        "embedding": {
            "method": "svd_pca",
            "explained_variance_ratio": explained.tolist(),
            "loadings": {
                "pc1": dict(zip(feature_order, loadings[0].tolist(), strict=True)),
                "pc2": dict(zip(feature_order, loadings[1].tolist(), strict=True)),
            },
        },
        "datasets": selection.to_dict(orient="records"),
        "platform_response": response,
        "validation": {
            "dataset_count": int(len(selection)),
            "platform_count": int(selection["platform"].nunique()),
            "method_configuration_count": int(performance.shape[1] - 1),
            "target_derived_features": [],
            "finite_coordinates": bool(np.isfinite(coordinates).all()),
        },
    }
    json_path = root / "platform_topography.json"
    _write_json(json_path, payload)

    artifact_paths = [table_path, json_path, *figure_paths]
    manifest = {
        "schema_version": 1,
        "protocol": "histoweave.cross_platform_topography.v1",
        "inputs": [
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (root / "dataset_features.csv", root / "performance_matrix_mean.csv")
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path = root / "topography_manifest.json"
    _write_json(manifest_path, manifest)
    _LOGGER.info("wrote cross-platform topography to %s", root)
    return {**payload, "manifest": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(_ROOT / "7x15_cross_platform"),
        help="Directory containing dataset_features.csv and performance_matrix_mean.csv.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        build_topography(args.results_dir)
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _LOGGER.error("topography build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
