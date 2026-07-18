import pytest

from histoweave.plugins import (
    Method,
    MethodCategory,
    MethodSpec,
    create_method,
    get_method,
    list_methods,
    register,
)
from histoweave.plugins.registry import _REGISTRY


def test_builtins_are_registered():
    names = {(m["category"], m["name"]) for m in list_methods()}
    assert ("qc", "basic_qc") in names
    assert ("normalization", "log1p_cp10k") in names
    assert ("domain_detection", "kmeans") in names
    assert ("domain_detection", "banksy_py") in names
    assert ("annotation", "marker_score") in names


def test_banksy_py_is_owned_by_builtin_package():
    from histoweave.plugins.experimental.banksy_py import BANKSYPyDomains as LegacyBANKSYPy

    cls = get_method("domain_detection", "banksy_py")
    assert cls.__module__ == "histoweave.plugins.builtin.banksy_py"
    assert cls.spec.maturity.value == "validated"
    assert LegacyBANKSYPy is cls


def test_filter_by_category():
    qc = list_methods(category="qc")
    assert qc and all(m["category"] == "qc" for m in qc)


def test_get_and_instantiate():
    cls = get_method("qc", "basic_qc")
    assert issubclass(cls, Method)
    inst = create_method("qc", "basic_qc", n_mads=3.0)
    assert inst.params["n_mads"] == 3.0


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        get_method("qc", "does_not_exist")


def test_unknown_param_rejected():
    with pytest.raises(TypeError):
        create_method("qc", "basic_qc", not_a_param=1)


def test_parameter_type_and_range_are_validated_before_execution():
    with pytest.raises(TypeError, match="expects int"):
        create_method("qc", "basic_qc", min_counts="ten")
    with pytest.raises(ValueError, match=">= 0"):
        create_method("qc", "basic_qc", min_counts=-1)
    with pytest.raises(ValueError, match="one of"):
        create_method("neighborhood", "spatial_graph", mode="triangle")


def test_register_requires_spec():
    with pytest.raises(TypeError):

        @register
        class Broken(Method):  # no spec
            def run(self, data):
                return data


def test_register_custom_method():
    key = (MethodCategory.NORMALIZATION, "unit_test_norm", "9.9")
    try:

        @register
        class MyNorm(Method):
            spec = MethodSpec(
                name="unit_test_norm",
                category=MethodCategory.NORMALIZATION,
                version="9.9",
            )

            def run(self, data):
                return data

        assert get_method("normalization", "unit_test_norm") is MyNorm
    finally:
        _REGISTRY.pop(key, None)
