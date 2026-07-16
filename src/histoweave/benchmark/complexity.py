"""Empirical log-log complexity fitting for scaling sweeps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ComplexityFit:
    exponent: float | None
    coefficient: float | None
    r_squared: float | None
    n_points: int
    status: str

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def fit_complexity(scales: Iterable[int], values: Iterable[float]) -> ComplexityFit:
    """Fit ``value ~= coefficient * n_cells**exponent`` on successful points."""
    pairs = [
        (float(scale), float(value))
        for scale, value in zip(scales, values, strict=True)
        if float(scale) > 0 and float(value) > 0 and np.isfinite(float(value))
    ]
    if len(pairs) < 3:
        return ComplexityFit(None, None, None, len(pairs), "insufficient_points")
    x = np.log(np.asarray([item[0] for item in pairs]))
    y = np.log(np.asarray([item[1] for item in pairs]))
    exponent, intercept = np.polyfit(x, y, deg=1)
    fitted = exponent * x + intercept
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - float(np.sum((y - fitted) ** 2)) / total
    return ComplexityFit(float(exponent), float(np.exp(intercept)), float(r_squared), len(pairs), "ok")
