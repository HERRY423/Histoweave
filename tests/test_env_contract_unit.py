"""Unit tests for SOTA environment contract surface."""

from histoweave.benchmark.sota_pipeline import env_contract


def test_env_contract_lists_required_outputs() -> None:
    contract = env_contract()
    assert "sota_benchmark_long.csv" in contract["outputs"]
    assert contract["task"] == "spatial_domain"
    assert set(contract["methods"]) >= {"spagcn", "banksy_py", "bayesspace"}
