"""Unit tests for default pipeline construction."""

from histoweave.workflow import default_pipeline


def test_default_pipeline_has_expected_stages() -> None:
    steps = default_pipeline()
    categories = [
        step.category.value if hasattr(step.category, "value") else step.category for step in steps
    ]
    assert "qc" in categories
    assert "normalization" in categories
    assert "domain_detection" in categories
