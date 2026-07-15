"""Audit H5AD inputs for cross-tissue spatial-neighborhood discovery.

This script is intentionally read-only.  It reports enough schema and count
information to decide whether SCTransform, scVI and spatial-neighborhood
analysis are statistically identifiable without materialising dense matrices.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_LOGGER = logging.getLogger(__name__)


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    _LOGGER.info("%s", message)


COUNT_TOLERANCE = 1e-6


def _decode(values: Any, limit: int = 20) -> list[str]:
    array = np.asarray(values).reshape(-1)[:limit]
    decoded: list[str] = []
    for value in array:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="replace"))
        else:
            decoded.append(str(value))
    return decoded


def _group_keys(group: h5py.Group | None) -> list[str]:
    return sorted(group.keys()) if group is not None else []


def _matrix_summary(node: h5py.Dataset | h5py.Group | None) -> dict[str, Any]:
    if node is None:
        return {"present": False}
    if isinstance(node, h5py.Dataset):
        sample = np.asarray(node[: min(node.shape[0], 64)])
        finite = np.isfinite(sample)
        return {
            "present": True,
            "encoding": "dense",
            "shape": list(node.shape),
            "dtype": str(node.dtype),
            "sample_min": float(np.nanmin(sample)) if sample.size else None,
            "sample_max": float(np.nanmax(sample)) if sample.size else None,
            "sample_finite": bool(finite.all()) if sample.size else True,
            "sample_integer_like": bool(
                np.all(np.abs(sample[finite] - np.rint(sample[finite])) <= COUNT_TOLERANCE)
            )
            if sample.size
            else True,
        }
    keys = set(node.keys())
    if {"data", "indices", "indptr"}.issubset(keys):
        data = np.asarray(node["data"][: min(node["data"].shape[0], 100_000)])
        finite = np.isfinite(data)
        shape = list(node.attrs.get("shape", []))
        return {
            "present": True,
            "encoding": str(node.attrs.get("encoding-type", "sparse")),
            "shape": shape,
            "dtype": str(node["data"].dtype),
            "nnz": int(node["data"].shape[0]),
            "sample_min": float(np.nanmin(data)) if data.size else None,
            "sample_max": float(np.nanmax(data)) if data.size else None,
            "sample_finite": bool(finite.all()) if data.size else True,
            "sample_integer_like": bool(
                np.all(np.abs(data[finite] - np.rint(data[finite])) <= COUNT_TOLERANCE)
            )
            if data.size
            else True,
        }
    return {
        "present": True,
        "encoding": str(node.attrs.get("encoding-type", "group")),
        "keys": sorted(keys),
    }


def _categorical_values(obs: h5py.Group, key: str, limit: int = 50) -> list[str] | None:
    if key not in obs:
        return None
    node = obs[key]
    if isinstance(node, h5py.Dataset):
        values = _decode(node, limit=limit)
        return sorted(set(values))[:limit]
    if isinstance(node, h5py.Group) and "categories" in node:
        return _decode(node["categories"], limit=limit)
    return None


def audit(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        obs = handle.get("obs")
        var = handle.get("var")
        layers = handle.get("layers")
        obsm = handle.get("obsm")
        uns = handle.get("uns")
        if not isinstance(obs, h5py.Group) or not isinstance(var, h5py.Group):
            raise ValueError(f"{path} does not contain AnnData obs/var groups")

        obs_keys = _group_keys(obs)
        var_keys = _group_keys(var)
        categorical_candidates = [
            "Animal_ID",
            "Animal_sex",
            "Behavior",
            "Bregma",
            "Cell_class",
            "Neuron_cluster_ID",
            "cluster",
            "cell_type",
            "domain_truth",
            "spatialLIBD",
            "layer_guess_reordered",
            "sample_id",
            "subject",
            "donor_id",
            "batch",
        ]
        categories = {
            key: values
            for key in categorical_candidates
            if (values := _categorical_values(obs, key)) is not None
        }
        layer_summary = (
            {key: _matrix_summary(layers[key]) for key in _group_keys(layers)}
            if isinstance(layers, h5py.Group)
            else {}
        )

        result: dict[str, Any] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "root_encoding": str(handle.attrs.get("encoding-type", "unknown")),
            "X": _matrix_summary(handle.get("X")),
            "layers": layer_summary,
            "obs_columns": obs_keys,
            "var_columns": var_keys,
            "obsm_keys": _group_keys(obsm) if isinstance(obsm, h5py.Group) else [],
            "uns_keys": _group_keys(uns) if isinstance(uns, h5py.Group) else [],
            "selected_categories": categories,
            "count_compatible_sources": [],
        }
        sources = {"X": result["X"], **{f"layers/{k}": v for k, v in layer_summary.items()}}
        for name, summary in sources.items():
            if (
                summary.get("present")
                and summary.get("sample_finite")
                and summary.get("sample_integer_like")
                and (summary.get("sample_min") is None or summary["sample_min"] >= 0)
            ):
                result["count_compatible_sources"].append(name)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    reports = [audit(path) for path in args.paths]
    _log(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
