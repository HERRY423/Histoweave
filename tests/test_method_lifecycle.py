import pytest

from histoweave.plugins import (
    BackendRequirement,
    Method,
    MethodCategory,
    MethodDeprecation,
    MethodDeprecationWarning,
    MethodImplementation,
    MethodReference,
    MethodSpec,
    ParamSpec,
    create_method,
    get_method,
    list_methods,
    migrate_method_params,
    register,
)


def _register_lifecycle_pair(name: str):
    @register
    class VersionTwo(Method):
        spec = MethodSpec(
            name=name,
            category=MethodCategory.QC,
            version="2.0.0",
            params=(ParamSpec("new_threshold", "int", 2),),
        )

        def run(self, data):
            return self.finalize(data.copy())

    @register
    class VersionOne(Method):
        spec = MethodSpec(
            name=name,
            category=MethodCategory.QC,
            version="1.0.0",
            params=(
                ParamSpec("old_threshold", "int", 1),
                ParamSpec("obsolete", "bool", False),
            ),
            deprecation=MethodDeprecation(
                since="0.2.0",
                remove_in="0.4.0",
                reason="The upstream method changed its parameter schema.",
                replacement=MethodReference("qc", name, "2.0.0"),
                parameter_renames=(("old_threshold", "new_threshold"),),
                removed_parameters=("obsolete",),
                notes="Drop obsolete only after reviewing the upstream defaults.",
            ),
        )

        def run(self, data):
            return self.finalize(data.copy())

    return VersionOne, VersionTwo


def test_registry_keeps_versions_and_defaults_to_latest_active_release():
    version_one, version_two = _register_lifecycle_pair("lifecycle_selection_test")

    assert get_method("qc", "lifecycle_selection_test") is version_two
    assert get_method("qc", "lifecycle_selection_test", "2.0.0") is version_two
    with pytest.warns(MethodDeprecationWarning, match="scheduled for removal"):
        assert get_method("qc", "lifecycle_selection_test", "1.0.0") is version_one

    default_rows = [row for row in list_methods("qc") if row["name"] == "lifecycle_selection_test"]
    all_rows = [
        row
        for row in list_methods("qc", all_versions=True)
        if row["name"] == "lifecycle_selection_test"
    ]
    assert [row["version"] for row in default_rows] == ["2.0.0"]
    assert [row["version"] for row in all_rows] == ["1.0.0", "2.0.0"]
    legacy = all_rows[0]
    assert legacy["deprecated"] is True
    assert legacy["deprecation"]["replacement"]["version"] == "2.0.0"


def test_parameter_migration_is_declarative_validated_and_never_drops_values():
    _register_lifecycle_pair("lifecycle_migration_test")

    migration = migrate_method_params(
        "qc",
        "lifecycle_migration_test",
        "1.0.0",
        {"old_threshold": 7},
    )
    assert migration["version"] == "2.0.0"
    assert migration["params"] == {"new_threshold": 7}
    assert len(migration["path"]) == 1
    method = create_method(
        migration["category"],
        migration["name"],
        version=migration["version"],
        **migration["params"],
    )
    assert method.params["new_threshold"] == 7

    with pytest.raises(ValueError, match="cannot migrate removed parameters"):
        migrate_method_params(
            "qc",
            "lifecycle_migration_test",
            "1.0.0",
            {"obsolete": True},
        )


def test_external_methods_require_a_real_declared_backend():
    with pytest.raises(ValueError, match="backend requirements"):
        MethodSpec(
            name="invalid_external_contract",
            category=MethodCategory.QC,
            version="1.0.0",
            wraps="example.Backend",
            implementation=MethodImplementation.EXTERNAL,
        )

    critical = {"cell2location", "banksy", "spatialde", "cellpose2", "scanvi"}
    specs = {row["name"]: row for row in list_methods() if row["name"] in critical}
    assert set(specs) == critical
    assert all(row["implementation"] == "external" for row in specs.values())
    assert all(row["backends"] for row in specs.values())
    assert all(row["wraps"] for row in specs.values())

    requirement = BackendRequirement("upstream", ">=1", "extra")
    assert requirement.install_extra == "extra"
