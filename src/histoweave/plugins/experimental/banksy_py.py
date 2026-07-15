"""Compatibility import for histoweave.plugins.builtin.banksy_py.

The implementation became a built-in in 0.1.0. Importing this module remains
supported so existing notebooks do not need an immediate import-path migration.
"""

from __future__ import annotations

from ..builtin.banksy_py import BANKSYPyDomains

__all__ = ["BANKSYPyDomains"]
