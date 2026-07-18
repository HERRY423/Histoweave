"""Unit tests for provenance records."""

from histoweave.data import Provenance


def test_provenance_to_dict_roundtrip_fields() -> None:
    prov = Provenance(
        step="qc",
        method="basic_qc",
        method_version="0.1.0",
        params={"n_mads": 3},
        histoweave_version="0.1.0b1",
    )
    payload = prov.to_dict()
    assert payload["step"] == "qc"
    assert payload["method"] == "basic_qc"
    assert payload["params"]["n_mads"] == 3
    assert "timestamp" in payload
