"""Optional adapters for production spatial-analysis libraries.

The heavy dependencies wrapped here are deliberately imported inside ``run``.  A
plain HistoWeave installation can therefore inspect and select the methods without
installing every deep-learning or R-adjacent stack.
"""

from __future__ import annotations
