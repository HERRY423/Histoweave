"""Build a self-contained HTML analysis report from a processed SpatialTable.

The report includes both static SVG fallback plots and, when JavaScript is
available, an interactive **Vitessce** viewer for spatial scatterplots and
expression heatmaps.  Vitessce loads from CDN so there is no Python dependency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..data import SpatialTable
from .svg import spatial_scatter_svg
from .vitessce_data import build_vitessce_view_config

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def build_report(data: SpatialTable, output_path: str | Path) -> Path:
    """Render an HTML report for ``data`` and write it to ``output_path``.

    The report summarizes QC, spatial maps (domains and, if present, annotations),
    an annotation breakdown, and the full pipeline/provenance trail.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = _build_context(data)
    html = _env().get_template("report.html.j2").render(**context)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _build_context(data: SpatialTable) -> dict:
    from .. import __version__

    coords = data.spatial
    plots = []
    if coords is not None:
        if "domain" in data.obs:
            plots.append(
                spatial_scatter_svg(
                    coords, list(data.obs["domain"].astype(str)), title="Spatial domains"
                )
            )
        if "cell_type" in data.obs:
            plots.append(
                spatial_scatter_svg(
                    coords, list(data.obs["cell_type"].astype(str)), title="Annotation"
                )
            )
        if not plots and "domain_truth" in data.obs:
            plots.append(
                spatial_scatter_svg(
                    coords, list(data.obs["domain_truth"].astype(str)), title="Ground truth"
                )
            )

    annotation_counts = []
    if "cell_type" in data.obs:
        counts = data.obs["cell_type"].value_counts()
        total = counts.sum()
        annotation_counts = [
            {"label": str(label), "count": int(n), "pct": 100 * n / total}
            for label, n in counts.items()
        ]

    n_domains = int(data.obs["domain"].nunique()) if "domain" in data.obs else 0
    manifest = data.uns.get("run_manifest", {})

    # Multi-method boundary uncertainty (P1 default when ≥2 partitions exist).
    uncertainty_block = _uncertainty_context(data)
    if uncertainty_block and coords is not None and "uncertainty_plot" in uncertainty_block:
        plots.append(uncertainty_block.pop("uncertainty_plot"))

    # Vitessce interactive viewer data (self-contained JSON, loads from CDN)
    vitessce_json = ""
    try:
        vc = build_vitessce_view_config(data, top_genes=30)
        vitessce_json = json.dumps(vc, allow_nan=False, default=str)
    except Exception:
        # Never let a Vitessce serialisation error crash the report.
        vitessce_json = "{}"

    return {
        "assay": data.uns.get("assay", "unknown"),
        "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "histoweave_version": __version__,
        "n_obs": data.n_obs,
        "n_vars": data.n_vars,
        "n_domains": n_domains,
        "qc": data.uns.get("qc"),
        "plots": plots,
        "annotation_counts": annotation_counts,
        "steps": manifest.get("steps", []),
        "compiler": manifest.get("compiler"),
        "vitessce_json": vitessce_json,
        "uncertainty": uncertainty_block,
    }


def _collect_method_predictions(data: SpatialTable) -> dict[str, list[str]]:
    """Gather multi-method partitions for uncertainty mapping.

    Sources (in order):
    1. ``uns['method_predictions']`` — explicit ``{method: labels}`` map.
    2. Multiple ``obs`` columns whose names start with ``domain`` (excluding truth).
    """
    predictions: dict[str, list[str]] = {}
    raw = data.uns.get("method_predictions")
    if isinstance(raw, dict) and len(raw) >= 2:
        for name, labels in raw.items():
            predictions[str(name)] = [str(v) for v in list(labels)]
        return predictions

    domain_cols = [
        col
        for col in data.obs.columns
        if str(col).startswith("domain") and str(col) not in {"domain_truth", "domain_balance"}
    ]
    # Prefer at least two distinct partition columns.
    if len(domain_cols) >= 2:
        for col in domain_cols:
            predictions[str(col)] = data.obs[col].astype(str).tolist()
    return predictions


def _uncertainty_context(data: SpatialTable) -> dict | None:
    """Build a JSON-friendly uncertainty summary + optional SVG plot."""
    import numpy as np

    coords = data.spatial
    if coords is None or data.n_obs < 8:
        return None
    predictions = _collect_method_predictions(data)
    if len(predictions) < 2:
        return None
    # Align lengths; skip malformed entries.
    cleaned: dict[str, np.ndarray] = {}
    for name, labels in predictions.items():
        array = np.asarray(labels)
        if array.shape[0] != data.n_obs:
            continue
        cleaned[name] = array
    if len(cleaned) < 2:
        return None
    try:
        from ..benchmark.uncertainty import boundary_uncertainty
        from .svg import continuous_scatter_svg
    except Exception:
        continuous_scatter_svg = None  # type: ignore[assignment]
        from ..benchmark.uncertainty import boundary_uncertainty

    try:
        result = boundary_uncertainty(
            np.asarray(coords, dtype=float), cleaned, k=min(6, data.n_obs - 1)
        )
    except Exception:
        return None

    block: dict = {
        "summary": result.summary(),
        "methods": list(result.method_names),
        "note": (
            "Target-free boundary uncertainty from cross-method edge votes. "
            "High values mark transitions where methods disagree — review priority, "
            "not a calibrated error probability."
        ),
    }
    # Persist on the table for downstream tooling (non-destructive copy path).
    data.uns.setdefault("boundary_uncertainty", result.summary())
    data.obs["boundary_uncertainty"] = result.uncertainty

    if continuous_scatter_svg is not None:
        try:
            block["uncertainty_plot"] = continuous_scatter_svg(
                np.asarray(coords, dtype=float),
                result.uncertainty,
                title="Boundary uncertainty (multi-method)",
            )
        except Exception:
            pass
    return block
