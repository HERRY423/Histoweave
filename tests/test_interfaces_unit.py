"""Unit tests for plugin interface validation helpers."""

import pytest

from histoweave.plugins.interfaces import (
    MethodCategory,
    MethodMaturity,
    ParamSpec,
)


def test_param_spec_type_and_bounds() -> None:
    spec = ParamSpec("n", "int", 3, minimum=1, maximum=10)
    spec.validate(5, method="demo")
    with pytest.raises(TypeError):
        spec.validate(1.5, method="demo")
    with pytest.raises(ValueError):
        spec.validate(0, method="demo")
    with pytest.raises(ValueError):
        spec.validate(11, method="demo")


def test_param_spec_choices() -> None:
    spec = ParamSpec("mode", "str", "a", choices=("a", "b"))
    spec.validate("b", method="demo")
    with pytest.raises(ValueError):
        spec.validate("c", method="demo")


def test_enums_are_stable() -> None:
    assert MethodCategory.DOMAIN_DETECTION.value == "domain_detection"
    assert MethodMaturity.VALIDATED.value == "validated"
    assert MethodMaturity.CONTRACT_VALIDATED.value == "contract_validated"
