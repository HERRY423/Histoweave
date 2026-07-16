"""Build the DLPFC boundary-uncertainty case study from archived spot predictions.

The discovery score is target-free: each method votes on whether a spatial kNN edge
crosses a domain, and per-edge Bernoulli entropy measures disagreement. Manual layers
are consulted only after the map is fixed, to quantify enrichment at true boundaries
and at boundary spots uniquely missed by one method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.uncertainty import (  # noqa: E402
    boundary_uncertainty,
    uncertainty_enrichment,
)
from scripts.scientific_figure_pro import (  # noqa: E402
    PALETTE,
    FigureStyle,
    apply_publication_style,
    create_subplots,
    finalize_figure,
)

LOGGER = logging.getLogger(__name__)
RESULTS = BASE / "results"
FIGURES = BASE / "figures"
METHOD_COLUMNS = ("banksy", "gmm", "scanvi")
METHOD_COLORS = {
    "banksy": PALETTE["blue_main"],
    "gmm": PALETTE["red_strong"],
    "scanvi": PALETTE["green_3"],
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
    )


def _boolean(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    folded = values.astype(str).str.strip().str.casefold()
    invalid = ~folded.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError(f"invalid boolean values: {sorted(folded[invalid].unique())}")
    return folded.isin({"true", "1"}).to_numpy(dtype=bool)


def _unique_miss_masks(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    manual_boundary: np.ndarray,
) -> dict[str, np.ndarray]:
    correct = {name: labels == truth for name, labels in predictions.items()}
    masks: dict[str, np.ndarray] = {}
    for name in predictions:
        others = [other for other in predictions if other != name]
        others_correct = np.logical_and.reduce([correct[other] for other in others])
        masks[name] = manual_boundary & ~correct[name] & others_correct
    return masks


def _make_figure(
    frame: pd.DataFrame,
    manual_boundary: np.ndarray,
    uncertainty: np.ndarray,
    unique_misses: dict[str, np.ndarray],
    validation: dict[str, Any],
    *,
    high_threshold: float,
) -> list[Path]:
    apply_publication_style(FigureStyle(font_size=11, axes_linewidth=1.4))
    fig, axes = create_subplots(2, 2, figsize=(11.5, 9.2), constrained_layout=True)
    plot_x = frame["y"].to_numpy(dtype=float)
    plot_y = -frame["x"].to_numpy(dtype=float)
    point_size = 8

    axes[0].scatter(
        plot_x[~manual_boundary], plot_y[~manual_boundary], s=point_size, color="#D9D9D9"
    )
    axes[0].scatter(
        plot_x[manual_boundary],
        plot_y[manual_boundary],
        s=point_size,
        color=PALETTE["red_strong"],
    )
    axes[0].set_title("A  Manual cortical-layer boundaries", loc="left", fontweight="bold")

    uncertainty_points = axes[1].scatter(
        plot_x,
        plot_y,
        s=point_size,
        c=uncertainty,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        linewidths=0,
    )
    colorbar = fig.colorbar(uncertainty_points, ax=axes[1], fraction=0.046, pad=0.03)
    colorbar.set_label("Boundary disagreement entropy")
    axes[1].set_title("B  Target-free uncertainty map", loc="left", fontweight="bold")

    axes[2].scatter(plot_x, plot_y, s=5, color="#E2E2E2", alpha=0.55)
    high = uncertainty >= high_threshold
    for name, mask in unique_misses.items():
        captured = mask & high
        missed = mask & ~high
        axes[2].scatter(
            plot_x[captured],
            plot_y[captured],
            s=18,
            color=METHOD_COLORS[name],
            label=f"{name}: captured",
            linewidths=0,
        )
        axes[2].scatter(
            plot_x[missed],
            plot_y[missed],
            s=24,
            facecolors="none",
            edgecolors=METHOD_COLORS[name],
            linewidths=0.9,
            label=f"{name}: below P80 threshold",
        )
    axes[2].legend(fontsize=7, ncol=2, loc="upper right")
    axes[2].set_title(
        "C  Boundary spots uniquely missed by one method",
        loc="left",
        fontweight="bold",
    )

    categories = ["Manual boundary", "Unique single-method miss"]
    boundary_metrics = validation["manual_boundary"]
    miss_metrics = validation["any_unique_miss"]
    overall = [
        100 * boundary_metrics["positive_prevalence"],
        100 * miss_metrics["positive_prevalence"],
    ]
    high_values = [
        100 * boundary_metrics["positive_prevalence_high"],
        100 * miss_metrics["positive_prevalence_high"],
    ]
    positions = np.arange(len(categories))
    width = 0.34
    axes[3].bar(
        positions - width / 2,
        overall,
        width,
        color=PALETTE["neutral"],
        edgecolor="black",
        linewidth=0.8,
        label="All spots",
    )
    axes[3].bar(
        positions + width / 2,
        high_values,
        width,
        color=PALETTE["green_3"],
        edgecolor="black",
        linewidth=0.8,
        label="Uncertainty >= P80",
    )
    axes[3].set_xticks(positions)
    axes[3].set_xticklabels(categories, rotation=10, ha="right")
    axes[3].set_ylabel("Spot prevalence (%)")
    axes[3].legend()
    axes[3].grid(axis="y", color="#E6E6E6", linewidth=0.7)
    for index, metrics in enumerate((boundary_metrics, miss_metrics)):
        enrichment = metrics["enrichment"]
        recall = metrics["recall_at_high"]
        axes[3].text(
            index,
            max(overall[index], high_values[index]) + 1.2,
            f"{enrichment:.2f}x enrichment\n{100 * recall:.1f}% recall",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[3].set_ylim(0, max(high_values + overall) * 1.28)
    axes[3].set_title("D  Post-hoc biological validation", loc="left", fontweight="bold")

    for axis in axes[:3]:
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.spines[:].set_visible(False)
    return finalize_figure(
        fig,
        FIGURES / "fig4_uncertainty_boundary_map",
        formats=("svg", "png"),
        dpi=300,
    )


def build_case(
    per_spot_path: str | Path = RESULTS / "per_spot.csv",
    *,
    k: int = 6,
    high_quantile: float = 0.8,
) -> dict[str, Any]:
    """Compute uncertainty, validation, hotspots, figures, and a checksum manifest."""

    input_path = Path(per_spot_path).resolve()
    frame = pd.read_csv(input_path)
    required = {"obs_id", "x", "y", "manual_layer", "is_boundary", *METHOD_COLUMNS}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"per-spot table is missing required columns: {missing}")
    coords = frame[["x", "y"]].to_numpy(dtype=float)
    truth = frame["manual_layer"].astype(str).to_numpy()
    predictions = {name: frame[name].astype(str).to_numpy() for name in METHOD_COLUMNS}
    manual_boundary = _boolean(frame["is_boundary"])
    result = boundary_uncertainty(coords, predictions, k=k)
    unique_misses = _unique_miss_masks(truth, predictions, manual_boundary)
    any_unique_miss = np.logical_or.reduce(list(unique_misses.values()))

    validation: dict[str, Any] = {
        "protocol": "histoweave.boundary_uncertainty.dlpfc.v1",
        "discovery_uses_ground_truth": False,
        "edge_vote_label_permutation_invariant": True,
        "k": k,
        "high_uncertainty_quantile": high_quantile,
        "result_summary": result.summary(),
        "manual_boundary": uncertainty_enrichment(
            result.uncertainty,
            manual_boundary,
            high_quantile=high_quantile,
        ),
        "any_unique_miss": uncertainty_enrichment(
            result.uncertainty,
            any_unique_miss,
            high_quantile=high_quantile,
        ),
        "unique_miss_by_method": {
            name: uncertainty_enrichment(
                result.uncertainty,
                mask,
                high_quantile=high_quantile,
            )
            for name, mask in unique_misses.items()
        },
    }
    high_threshold = float(validation["manual_boundary"]["high_threshold"])

    enriched = frame.copy()
    enriched["boundary_uncertainty"] = result.uncertainty
    enriched["consensus_boundary_strength"] = result.consensus_boundary_strength
    for name in METHOD_COLUMNS:
        enriched[f"{name}_boundary_strength"] = result.per_method_boundary_strength[name]
        enriched[f"{name}_missed_boundary_score"] = result.missed_boundary_score[name]
        enriched[f"{name}_unique_manual_boundary_miss"] = unique_misses[name]
    enriched["any_unique_manual_boundary_miss"] = any_unique_miss
    enriched["high_boundary_uncertainty"] = result.uncertainty >= high_threshold
    enriched["unique_miss_methods"] = [
        ";".join(name for name in METHOD_COLUMNS if unique_misses[name][index])
        for index in range(len(enriched))
    ]

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    per_spot_output = RESULTS / "per_spot_uncertainty.csv"
    enriched.to_csv(per_spot_output, index=False, float_format="%.10g")
    hotspots = enriched[enriched["high_boundary_uncertainty"]].sort_values(
        "boundary_uncertainty", ascending=False
    )
    hotspot_output = RESULTS / "uncertainty_hotspots.csv"
    hotspots.to_csv(hotspot_output, index=False, float_format="%.10g")
    validation_path = RESULTS / "uncertainty_validation.json"
    _write_json(validation_path, validation)

    figure_paths = _make_figure(
        enriched,
        manual_boundary,
        result.uncertainty,
        unique_misses,
        validation,
        high_threshold=high_threshold,
    )
    artifact_paths = [per_spot_output, hotspot_output, validation_path, *figure_paths]
    manifest = {
        "schema_version": 1,
        "protocol": validation["protocol"],
        "input": {
            "path": str(input_path.relative_to(BASE)).replace("\\", "/"),
            "sha256": _sha256(input_path),
            "bytes": input_path.stat().st_size,
        },
        "artifacts": [
            {
                "path": str(path.relative_to(BASE)).replace("\\", "/"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
        ],
    }
    manifest_path = RESULTS / "uncertainty_manifest.json"
    _write_json(manifest_path, manifest)
    LOGGER.info("wrote DLPFC boundary-uncertainty case study to %s", BASE)
    return {**validation, "manifest": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-spot", default=str(RESULTS / "per_spot.csv"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--high-quantile", type=float, default=0.8)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    try:
        build_case(args.per_spot, k=args.k, high_quantile=args.high_quantile)
    except (FileNotFoundError, OSError, TypeError, ValueError, RuntimeError) as exc:
        LOGGER.error("uncertainty case-study build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
