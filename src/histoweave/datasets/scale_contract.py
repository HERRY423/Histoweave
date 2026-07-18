"""Scale contracts for large imaging / multi-section datasets (P2).

Large Xenium / MERFISH / CosMx tables need explicit resource envelopes so users
and CI know when to subsample, when sparse paths are mandatory, and when a
method is expected to OOM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ScaleContract:
    """Resource envelope for one registered dataset or analysis class."""

    name: str
    n_obs_nominal: int
    n_vars_nominal: int
    recommended_subsample: int | None
    sparse_required: bool
    peak_ram_gb_estimate: float
    notes: str = ""
    platforms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["platforms"] = list(self.platforms)
        return payload

    def plan_for(self, n_obs: int | None = None) -> dict[str, Any]:
        """Return an analysis plan given observed (or nominal) cell count."""
        n = int(self.n_obs_nominal if n_obs is None else n_obs)
        subsample = self.recommended_subsample
        needs_subsample = subsample is not None and n > subsample
        return {
            "name": self.name,
            "n_obs": n,
            "sparse_required": self.sparse_required or n >= 50_000,
            "subsample_to": subsample if needs_subsample else None,
            "peak_ram_gb_estimate": self.peak_ram_gb_estimate,
            "knn_backend": "cKDTree",
            "domain_methods_safe_without_tile": n < 20_000,
            "notes": self.notes,
        }


# Named contracts referenced by the registry and large-imaging tutorial.
SCALE_CONTRACTS: dict[str, ScaleContract] = {
    "visium_standard": ScaleContract(
        name="visium_standard",
        n_obs_nominal=4_000,
        n_vars_nominal=20_000,
        recommended_subsample=None,
        sparse_required=True,
        peak_ram_gb_estimate=4.0,
        notes="Typical Visium capture area; full transcriptome, sparse counts.",
        platforms=("visium", "visium_hd"),
    ),
    "xenium_50k": ScaleContract(
        name="xenium_50k",
        n_obs_nominal=50_000,
        n_vars_nominal=400,
        recommended_subsample=20_000,
        sparse_required=True,
        peak_ram_gb_estimate=12.0,
        notes="Subsample for interactive domain sweeps; keep full table for QC only.",
        platforms=("xenium",),
    ),
    "xenium_full_slide": ScaleContract(
        name="xenium_full_slide",
        n_obs_nominal=200_000,
        n_vars_nominal=5_000,
        recommended_subsample=30_000,
        sparse_required=True,
        peak_ram_gb_estimate=48.0,
        notes="Tile spatially or subsample for sklearn-family domain methods.",
        platforms=("xenium",),
    ),
    "merfish_100k": ScaleContract(
        name="merfish_100k",
        n_obs_nominal=100_000,
        n_vars_nominal=500,
        recommended_subsample=25_000,
        sparse_required=True,
        peak_ram_gb_estimate=24.0,
        notes="Panel is small; cell count dominates. Prefer densify-on-demand only.",
        platforms=("merfish", "merscope"),
    ),
    "merfish_atlas": ScaleContract(
        name="merfish_atlas",
        n_obs_nominal=500_000,
        n_vars_nominal=500,
        recommended_subsample=40_000,
        sparse_required=True,
        peak_ram_gb_estimate=64.0,
        notes="Allen-scale sections: section-wise analysis + consensus.",
        platforms=("merfish",),
    ),
}


def scale_contract_for_assay(assay: str, n_obs: int | None = None) -> ScaleContract:
    """Pick a default scale contract from assay + optional observed n_obs."""
    assay = str(assay).lower()
    n = int(n_obs or 0)
    if assay in {"visium", "visium_hd"}:
        return SCALE_CONTRACTS["visium_standard"]
    if assay == "xenium":
        if n >= 100_000:
            return SCALE_CONTRACTS["xenium_full_slide"]
        return SCALE_CONTRACTS["xenium_50k"]
    if assay in {"merfish", "merscope"}:
        if n >= 200_000:
            return SCALE_CONTRACTS["merfish_atlas"]
        return SCALE_CONTRACTS["merfish_100k"]
    # Conservative default for unknown imaging platforms.
    return ScaleContract(
        name="generic_large",
        n_obs_nominal=max(n, 10_000),
        n_vars_nominal=2_000,
        recommended_subsample=20_000 if n > 20_000 else None,
        sparse_required=True,
        peak_ram_gb_estimate=16.0,
        notes="Generic large-table envelope.",
        platforms=(assay,),
    )


def registry_scale_table() -> list[dict[str, Any]]:
    """Attach scale plans to every registered real dataset."""
    from .real import list_datasets

    rows: list[dict[str, Any]] = []
    for entry in list_datasets():
        contract = scale_contract_for_assay(entry["assay"], entry.get("n_obs"))
        plan = contract.plan_for(entry.get("n_obs"))
        rows.append(
            {
                "dataset": entry["name"],
                "assay": entry["assay"],
                "tissue": entry.get("tissue"),
                "n_obs": entry.get("n_obs"),
                "analysis_task": entry.get("analysis_task"),
                "scale_contract": contract.name,
                **plan,
            }
        )
    return rows
