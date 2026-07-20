"""Versioned, atomic, integrity-checked ``SpatialTable`` bundle persistence.

The ``.ttab`` format is an interim process boundary until the production path moves to
SpatialData/OME-Zarr.  It is deliberately strict: a completed bundle has a manifest,
SHA-256 for every artifact, safe encoded keys, and is committed by an atomic directory
rename.  A partial write is never presented as a valid result.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import tempfile
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from ..data import SpatialTable
from ..logging import get_logger, log_event

_logger = get_logger("histoweave.io.bundle")

BUNDLE_FORMAT = "histoweave.spatial-table"
BUNDLE_SCHEMA_VERSION = 1
_MANIFEST_NAME = "bundle.json"


class BundleError(RuntimeError):
    """Base class for portable bundle failures."""


class BundleIntegrityError(BundleError):
    """A bundle is incomplete, corrupt, unsafe, or uses an unsupported schema."""


class BundleSerializationError(BundleError):
    """An object cannot be represented without data loss."""


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder for the NumPy scalar/array values commonly stored in ``uns``."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_bundle(
    table: SpatialTable,
    path: str | Path,
    *,
    overwrite: bool = False,
    allow_lossy_shapes: bool = False,
) -> Path:
    """Atomically serialize ``table`` to a versioned bundle directory.

    Existing output is never silently merged.  Pass ``overwrite=True`` only for a
    directory that is already recognisable as a HistoWeave bundle; arbitrary directories
    are refused.  Non-serializable shapes fail closed unless ``allow_lossy_shapes`` is
    explicitly selected.
    """

    root = Path(path)
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)

    if root.exists():
        if not root.is_dir():
            raise FileExistsError(f"bundle output exists and is not a directory: {root}")
        if not overwrite:
            raise FileExistsError(
                f"bundle output already exists: {root}; choose a new path or set overwrite=True"
            )
        if not _looks_like_bundle(root):
            raise BundleIntegrityError(
                f"refusing to overwrite {root}: it is not recognisable as a HistoWeave bundle"
            )

    log_event(_logger, 20, "bundle_write_start", "writing bundle",
              path=str(root), n_obs=table.n_obs, n_vars=table.n_vars)
    temporary = Path(tempfile.mkdtemp(prefix=f".{root.name}.tmp-", dir=parent))
    try:
        _write_bundle_contents(table, temporary, allow_lossy_shapes=allow_lossy_shapes)
        _commit_directory(temporary, root, overwrite=overwrite)
    except Exception:
        log_event(
            _logger, 40, "bundle_write_error",
            "bundle write failed — cleaning up temporary directory",
            path=str(temporary),
        )
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    log_event(_logger, 20, "bundle_write_done", "bundle written", path=str(root))
    return root


def read_bundle(
    path: str | Path,
    *,
    verify: bool = True,
) -> SpatialTable:
    """Load a bundle, verifying its manifest and artifact checksums by default."""

    root = Path(path)
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.exists():
        log_event(_logger, 20, "bundle_read_start", "reading versioned bundle",
                  path=str(root), verify=verify)
        result = _read_versioned_bundle(root, verify=verify)
        log_event(_logger, 20, "bundle_read_done", "bundle loaded",
                  path=str(root), n_obs=result.n_obs, n_vars=result.n_vars)
        return result

    if (root / "X.npy").exists():
        warnings.warn(
            f"{root} is a legacy unversioned bundle; integrity cannot be verified. "
            "Rewrite it with write_bundle() before production use.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _read_legacy_bundle(root)
    raise FileNotFoundError(f"{root} is not a HistoWeave bundle (no {_MANIFEST_NAME} or X.npy)")


def inspect_bundle(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Return validated bundle metadata without materializing arrays or data frames."""

    root = Path(path)
    manifest = _load_manifest(root)
    artifacts = _validate_artifacts(root, manifest, verify=verify)
    return {
        **{key: value for key, value in manifest.items() if key != "artifacts"},
        "artifact_count": len(artifacts),
        "verified": verify,
    }


def _write_bundle_contents(
    table: SpatialTable,
    root: Path,
    *,
    allow_lossy_shapes: bool,
) -> None:
    artifacts: list[dict[str, Any]] = []

    artifacts.append(_save_array(root, "X.npy", table.X, role="X"))
    table.obs.to_parquet(root / "obs.parquet")
    artifacts.append(_artifact_metadata(root, "obs.parquet", role="obs"))
    table.var.to_parquet(root / "var.parquet")
    artifacts.append(_artifact_metadata(root, "var.parquet", role="var"))

    for role, directory, mapping in (
        ("obsm", "obsm", table.obsm),
        ("layer", "layers", table.layers),
        ("image", "images", table.images),
    ):
        # AnnData's .layers stores X under a None key — skip it.
        for key, value in sorted(
            (k, v) for k, v in mapping.items() if k is not None
        ):
            relative = f"{directory}/{_encode_key(key)}.npy"
            artifacts.append(_save_array(root, relative, value, role=role, key=key))

    uns = dict(table.uns)
    serializable_shapes: dict[str, Any] = {}
    dropped: list[str] = []
    for name, geometry in table.shapes.items():
        try:
            json.dumps(geometry, cls=_NumpyJSONEncoder)
        except TypeError as exc:
            if not allow_lossy_shapes:
                raise BundleSerializationError(
                    f"shape {name!r} is not JSON-serializable; refusing a lossy bundle. "
                    "Use SpatialData/GeoParquet or explicitly set allow_lossy_shapes=True."
                ) from exc
            dropped.append(name)
        else:
            serializable_shapes[name] = geometry

    if dropped:
        uns = {
            **uns,
            "_bundle_warnings": [
                f"shapes[{name!r}] was explicitly dropped during bundle serialization"
                for name in dropped
            ],
        }
    if serializable_shapes:
        _write_json(root / "shapes.json", serializable_shapes, description="shapes")
        artifacts.append(_artifact_metadata(root, "shapes.json", role="shapes"))

    _write_json(root / "uns.json", uns, description="uns")
    artifacts.append(_artifact_metadata(root, "uns.json", role="uns"))

    from .. import __version__

    manifest = {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created": datetime.now(UTC).isoformat(),
        "producer": {"name": "histoweave", "version": __version__},
        "table": {"shape": [table.n_obs, table.n_vars], "x_dtype": str(table.X.dtype)},
        "artifacts": artifacts,
    }
    (root / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def _read_versioned_bundle(root: Path, *, verify: bool) -> SpatialTable:
    manifest = _load_manifest(root)
    artifacts = _validate_artifacts(root, manifest, verify=verify)

    def required(role: str) -> dict[str, Any]:
        matches = [item for item in artifacts if item["role"] == role]
        if len(matches) != 1:
            raise BundleIntegrityError(
                f"bundle manifest requires exactly one {role!r} artifact; found {len(matches)}"
            )
        return matches[0]

    X = _load_array(root, required("X"))
    obs = pd.read_parquet(_safe_artifact_path(root, required("obs")["path"]))
    var = pd.read_parquet(_safe_artifact_path(root, required("var")["path"]))

    mappings: dict[str, dict[str, np.ndarray]] = {"obsm": {}, "layer": {}, "image": {}}
    for role in mappings:
        for artifact in (item for item in artifacts if item["role"] == role):
            key = artifact.get("key")
            if not isinstance(key, str) or not key:
                raise BundleIntegrityError(f"{role} artifact is missing a non-empty key")
            if key in mappings[role]:
                raise BundleIntegrityError(f"duplicate {role} artifact key {key!r}")
            mappings[role][key] = _load_array(root, artifact)

    uns = _load_json_artifact(root, required("uns"))
    if not isinstance(uns, dict):
        raise BundleIntegrityError("uns.json must contain a JSON object")

    shape_artifacts = [item for item in artifacts if item["role"] == "shapes"]
    if len(shape_artifacts) > 1:
        raise BundleIntegrityError("bundle has more than one shapes artifact")
    shapes = _load_json_artifact(root, shape_artifacts[0]) if shape_artifacts else {}
    if not isinstance(shapes, dict):
        raise BundleIntegrityError("shapes.json must contain a JSON object")

    table = SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm=mappings["obsm"],
        layers=mappings["layer"],
        images=mappings["image"],
        shapes=shapes,
        uns=uns,
    )
    expected_shape = manifest.get("table", {}).get("shape")
    if expected_shape != [table.n_obs, table.n_vars]:
        raise BundleIntegrityError(
            f"manifest shape {expected_shape!r} does not match loaded table {list(table.shape)!r}"
        )
    return table


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads((root / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BundleIntegrityError(f"bundle manifest is missing: {root / _MANIFEST_NAME}") from exc
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError(f"bundle manifest is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BundleIntegrityError("bundle manifest must contain a JSON object")
    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleIntegrityError(f"unsupported bundle format {manifest.get('format')!r}")
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleIntegrityError(
            f"unsupported bundle schema {manifest.get('schema_version')!r}; "
            f"this HistoWeave supports {BUNDLE_SCHEMA_VERSION}"
        )
    if not isinstance(manifest.get("artifacts"), list):
        raise BundleIntegrityError("bundle manifest is missing its artifacts list")
    return manifest


def _validate_artifacts(
    root: Path,
    manifest: dict[str, Any],
    *,
    verify: bool,
) -> list[dict[str, Any]]:
    artifacts = manifest["artifacts"]
    seen_paths: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise BundleIntegrityError("every artifact entry must be a JSON object")
        relative = artifact.get("path")
        role = artifact.get("role")
        if not isinstance(relative, str) or not relative or not isinstance(role, str):
            raise BundleIntegrityError("artifact entries require non-empty path and role strings")
        if relative in seen_paths:
            raise BundleIntegrityError(f"duplicate artifact path {relative!r}")
        seen_paths.add(relative)
        file_path = _safe_artifact_path(root, relative)
        if not file_path.is_file():
            raise BundleIntegrityError(f"bundle artifact is missing: {relative}")
        if verify:
            expected_bytes = artifact.get("bytes")
            actual_bytes = file_path.stat().st_size
            if expected_bytes != actual_bytes:
                raise BundleIntegrityError(
                    f"size mismatch for {relative}: expected {expected_bytes}, got {actual_bytes}"
                )
            expected_hash = artifact.get("sha256")
            actual_hash = _sha256(file_path)
            if expected_hash != actual_hash:
                raise BundleIntegrityError(
                    f"checksum mismatch for {relative}: expected {expected_hash}, got {actual_hash}"
                )
    return artifacts


def _save_array(
    root: Path,
    relative: str,
    value: Any,
    *,
    role: str,
    key: str | None = None,
) -> dict[str, Any]:
    destination = _safe_artifact_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(value), allow_pickle=False)
    return _artifact_metadata(root, relative, role=role, key=key)


def _load_array(root: Path, artifact: dict[str, Any]) -> np.ndarray:
    return np.load(_safe_artifact_path(root, artifact["path"]), allow_pickle=False)


def _artifact_metadata(
    root: Path,
    relative: str,
    *,
    role: str,
    key: str | None = None,
) -> dict[str, Any]:
    file_path = _safe_artifact_path(root, relative)
    result: dict[str, Any] = {
        "path": Path(relative).as_posix(),
        "role": role,
        "bytes": file_path.stat().st_size,
        "sha256": _sha256(file_path),
    }
    if key is not None:
        result["key"] = key
    return result


def _write_json(path: Path, value: Any, *, description: str) -> None:
    try:
        payload = json.dumps(value, cls=_NumpyJSONEncoder, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BundleSerializationError(f"{description} is not JSON-serializable: {exc}") from exc
    path.write_text(payload, encoding="utf-8")


def _load_json_artifact(root: Path, artifact: dict[str, Any]) -> Any:
    path = _safe_artifact_path(root, artifact["path"])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError(f"invalid JSON artifact {artifact['path']}: {exc}") from exc


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute():
        raise BundleIntegrityError(f"artifact path must be relative: {relative!r}")
    resolved_root = root.resolve()
    resolved = (root / candidate_relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise BundleIntegrityError(f"artifact path escapes the bundle: {relative!r}") from exc
    return resolved


def _encode_key(key: str) -> str:
    if not isinstance(key, str) or not key:
        raise BundleSerializationError("array mapping keys must be non-empty strings")
    encoded = base64.urlsafe_b64encode(key.encode("utf-8")).decode("ascii").rstrip("=")
    return f"k-{encoded}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _looks_like_bundle(root: Path) -> bool:
    return (root / _MANIFEST_NAME).is_file() or all(
        (root / name).is_file() for name in ("X.npy", "obs.parquet", "var.parquet")
    )


def _commit_directory(temporary: Path, root: Path, *, overwrite: bool) -> None:
    if not root.exists():
        _replace_with_retry(temporary, root)
        return
    if not overwrite:
        raise FileExistsError(f"bundle output already exists: {root}")

    backup = root.parent / f".{root.name}.backup-{uuid4().hex}"
    _replace_with_retry(root, backup)
    try:
        _replace_with_retry(temporary, root)
    except Exception:
        log_event(
            _logger, 40, "bundle_atomic_commit_error",
            "atomic commit failed — restoring backup",
            path=str(root),
        )
        _replace_with_retry(backup, root)
        raise
    else:
        _rmtree_with_retry(backup)


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 6) -> None:
    """Replace a path, tolerating short-lived Windows file-handle contention.

    Antivirus scanners and filesystem indexers can briefly open a newly written
    Parquet or NumPy artifact between serialization and the atomic directory rename.
    Windows then raises ``PermissionError`` even though neither path's permissions are
    wrong. Retrying only that specific exception keeps the commit atomic while allowing
    the external handle time to close; all other failures propagate at once.
    """
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01 * (2**attempt))


def _rmtree_with_retry(path: Path, *, attempts: int = 6) -> None:
    """Remove a committed bundle backup with the same bounded retry policy."""
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.01 * (2**attempt))


def _read_legacy_bundle(root: Path) -> SpatialTable:
    def load_arrays(directory: Path) -> dict[str, np.ndarray]:
        if not directory.is_dir():
            return {}
        return {p.stem: np.load(p, allow_pickle=False) for p in sorted(directory.glob("*.npy"))}

    X = np.load(root / "X.npy", allow_pickle=False)
    obs = pd.read_parquet(root / "obs.parquet")
    var = pd.read_parquet(root / "var.parquet")
    uns = (
        json.loads((root / "uns.json").read_text(encoding="utf-8"))
        if (root / "uns.json").exists()
        else {}
    )
    shapes = (
        json.loads((root / "shapes.json").read_text(encoding="utf-8"))
        if (root / "shapes.json").exists()
        else {}
    )
    return SpatialTable(
        X=X,
        obs=obs,
        var=var,
        obsm=load_arrays(root / "obsm"),
        layers=load_arrays(root / "layers"),
        images=load_arrays(root / "images"),
        shapes=shapes,
        uns=uns,
    )
