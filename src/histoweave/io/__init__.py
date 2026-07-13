"""Ingestion / adapters layer."""

from __future__ import annotations

from .base import Reader
from .bundle import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    BundleIntegrityError,
    BundleSerializationError,
    inspect_bundle,
    read_bundle,
    write_bundle,
)
from .readers import READERS, get_reader, read

__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BundleError",
    "BundleIntegrityError",
    "BundleSerializationError",
    "Reader",
    "READERS",
    "get_reader",
    "inspect_bundle",
    "read",
    "read_bundle",
    "write_bundle",
]
