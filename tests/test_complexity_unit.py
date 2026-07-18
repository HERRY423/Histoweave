"""Unit tests for complexity curve fitting helpers."""

import pytest

from histoweave.benchmark.complexity import fit_complexity


def test_fit_complexity_linear_trend() -> None:
    xs = [10, 20, 40, 80]
    ys = [21.0, 41.0, 81.0, 161.0]
    fit = fit_complexity(xs, ys)
    assert fit.status == "ok"
    assert fit.exponent is not None
    assert fit.exponent == pytest.approx(1.0, abs=0.05)
    assert fit.r_squared is not None
    assert fit.r_squared > 0.99
