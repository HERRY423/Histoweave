"""Pre-registered F3/F4 functional experiments for cryptic-state validation.

Three experiment classes upgrade computational F2 claims:

* **Perturbation** — CRISPR / CRISPRi / CRISPRa / small-molecule drugs
* **Lineage tracing** — genetic or barcode lineage reporters mapped to ROIs
* **Orthogonal platform** — independent assay (CODEX, MERFISH, Xenium, multiome,
  second tissue technology) confirming the same spatial program

This module is the frozen design registry. Package builders and return
analyzers import it; they do **not** invent wet-lab outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentSpec:
    """One pre-registered functional experiment."""

    experiment_id: str
    discovery_id: str  # D1_L6_myelin | D2_L3_plasticity | D3_LN_ca2
    class_: str  # perturbation | lineage | orthogonal
    modality: str  # crispr | drug | lineage_reporter | codex | merfish | xenium | multiome | …
    title: str
    hypothesis: str
    system: str  # organoid / slice culture / mouse / human tissue / cell line
    targets: tuple[str, ...]
    readouts: tuple[str, ...]
    primary_contrast: str
    pass_criteria: tuple[str, ...]
    claim_level_on_pass: str  # F3 | F4
    priority: int = 2
    n_replicates_min: int = 3
    controls: tuple[str, ...] = ()
    notes: str = ""
    related_roi: str = ""
    return_template: str = "functional_return_template.csv"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("class_")
        return d


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    # ===== D1 L6 myelin — perturbation =====
    ExperimentSpec(
        experiment_id="D1_crispri_myrf_olig2",
        discovery_id="D1_L6_myelin",
        class_="perturbation",
        modality="crispr",
        title="CRISPRi of MYRF / OLIG2 in human cortical organoids or slice culture",
        hypothesis=(
            "Reducing oligodendrocyte-lineage drivers shrinks or transcriptionally "
            "erases the L6-myelin microcompartment program (MBP/PLP1/MOBP down in "
            "ROI-matched deep-layer zones) without collapsing bulk Layer-6 identity."
        ),
        system="human iPSC cortical organoid ± slice culture; optional mouse L6 slice",
        targets=("MYRF", "OLIG2", "SOX10"),
        readouts=(
            "spatial transcriptomics or smFISH panel MBP/PLP1/MOBP",
            "IF MBP + SOX10",
            "layer marker controls (e.g. FOXP2/TLE4 for deep layers)",
        ),
        primary_contrast="perturbed vs non-targeting gRNA; ROI-matched deep layer vs rest",
        pass_criteria=(
            "MBP and/or PLP1 ROI-vs-rest Δ decreases vs control (padj≤0.05, n≥3)",
            "deep-layer identity markers not globally ablated (fold-change within ±20% or padj>0.05)",
            "effect direction pre-registered: myelin program ↓ under CRISPRi",
        ),
        claim_level_on_pass="F4",
        priority=1,
        n_replicates_min=3,
        controls=("non-targeting gRNA", "scrambled", "vehicle for any drug arm"),
        related_roi="ROI_151508_L6_n154",
        notes="Causal test of oligodendrocyte program necessity for the niche signature.",
    ),
    ExperimentSpec(
        experiment_id="D1_drug_demyelination_cuprizone",
        discovery_id="D1_L6_myelin",
        class_="perturbation",
        modality="drug",
        title="Cuprizone (or lysolecithin) demyelination with spatial readout",
        hypothesis=(
            "Induced demyelination remaps multi-method uncertainty niches in deep "
            "cortex and reduces L6-myelin microcompartment markers; recovery phase "
            "partially restores the niche."
        ),
        system="mouse cortex (cuprizone diet) or local LPC lesion; spatial Visium/Xenium",
        targets=("myelin integrity",),
        readouts=(
            "HistoWeave uncertainty niche pipeline on treated vs control sections",
            "MBP/PLP1 IF",
            "cryptic component size and myelin panel Δrest",
        ),
        primary_contrast="treated demyelinated vs control age-matched; recovery time course",
        pass_criteria=(
            "myelin panel Δrest in deep-layer cryptic components decreases under demyelination (padj≤0.05)",
            "cryptic component geometry remains measurable (not pure dropout of all spots)",
            "partial recovery of myelin panel at remyelination time point (direction pre-registered)",
        ),
        claim_level_on_pass="F4",
        priority=2,
        n_replicates_min=4,
        controls=("vehicle diet", "contralateral unlesioned hemisphere for LPC"),
        related_roi="ROI_151508_L6_n154",
        notes="Disease-mechanism stress test linking niche to demyelination vulnerability.",
    ),
    # ===== D1 — lineage =====
    ExperimentSpec(
        experiment_id="D1_lineage_opc_reporter",
        discovery_id="D1_L6_myelin",
        class_="lineage",
        modality="lineage_reporter",
        title="OPC lineage reporter mapped onto L6 cryptic ROI geometry",
        hypothesis=(
            "PDGFRA+ or OLIG2-lineage cells are enriched inside L6-myelin cryptic "
            "components relative to adjacent L6 non-ROI."
        ),
        system="Pdgfra-CreERT2;Rosa-tdTomato (or Olig2-lineage) mouse; adult pulse-chase",
        targets=("PDGFRA", "OLIG2", "MBP"),
        readouts=(
            "reporter+ cell density in ROI vs same-layer non-ROI",
            "co-IF MBP",
            "optional spatial RNA of reporter sorted cells",
        ),
        primary_contrast="reporter density ROI vs same-layer L6 non-ROI",
        pass_criteria=(
            "reporter+ density higher in ROI (padj≤0.05, n≥3 animals)",
            "≥30% of ROI MBP+ area co-localizes with lineage label OR lineage cells show MBP program",
        ),
        claim_level_on_pass="F3",
        priority=1,
        n_replicates_min=3,
        controls=("oil vehicle no tamoxifen", "non-cryptic L6 ROIs"),
        related_roi="ROI_151508_L6_n154",
    ),
    # ===== D1 — orthogonal platform =====
    ExperimentSpec(
        experiment_id="D1_orthogonal_merfish_xenium_myelin",
        discovery_id="D1_L6_myelin",
        class_="orthogonal",
        modality="merfish_or_xenium",
        title="Imaging-based ST (MERFISH/Xenium) confirmation of L6 myelin microcompartment",
        hypothesis=(
            "On an independent brain section/platform, multi-method uncertainty "
            "recovers a pure deep-layer cryptic niche with myelin-panel elevation "
            "surviving spatial-shift null."
        ),
        system="human DLPFC or mouse homologous deep cortex; MERFISH or Xenium myelin panel",
        targets=("MBP", "PLP1", "MOBP", "SOX10", "deep-layer controls"),
        readouts=(
            "HistoWeave discovery pipeline on orthogonal counts",
            "myelin panel Δrest + shift p",
            "component purity for deep layer / L6-homologue",
        ),
        primary_contrast="cryptic component vs rest; vs same deep-layer background",
        pass_criteria=(
            "≥1 pure deep-layer cryptic component with myelin Δrest>0 and shift p≤0.05",
            "SCGB/SAA artifact genes not required for pass",
            "platform ≠ original Visium 151508 bundle",
        ),
        claim_level_on_pass="F3",
        priority=1,
        n_replicates_min=2,
        controls=("technical replicate section", "white-matter positive control for MBP"),
        related_roi="ROI_151508_L6_n154",
        notes="Orthogonal platform confirmation without requiring CRISPR.",
    ),
    # ===== D2 L3 plasticity — perturbation =====
    ExperimentSpec(
        experiment_id="D2_crispri_enc1_hopx",
        discovery_id="D2_L3_plasticity",
        class_="perturbation",
        modality="crispr",
        title="CRISPRi ENC1 and/or HOPX in mid-layer cortical models",
        hypothesis=(
            "Knockdown of ENC1/HOPX reduces the L3-plasticity niche signature "
            "(GAP43/GRIA2/NRGN program) inside mid-layer ROIs more than bulk L3 markers."
        ),
        system="human cortical organoid or primary culture; optional in vivo AAV-CRISPRi",
        targets=("ENC1", "HOPX", "GAP43"),
        readouts=(
            "smFISH/IF ENC1 HOPX GAP43 GRIA2",
            "spatial or ROI-bulk RNA plasticity module score",
        ),
        primary_contrast="CRISPRi vs non-targeting; mid-layer ROI vs rest",
        pass_criteria=(
            "plasticity module score ↓ in mid-layer under CRISPRi (padj≤0.05, n≥3)",
            "MBP remains not elevated (no ectopic myelin program)",
        ),
        claim_level_on_pass="F4",
        priority=1,
        n_replicates_min=3,
        controls=("non-targeting gRNA",),
        related_roi="ROI_151508_L3_n138",
    ),
    ExperimentSpec(
        experiment_id="D2_drug_activity_block_ttx",
        discovery_id="D2_L3_plasticity",
        class_="perturbation",
        modality="drug",
        title="Activity blockade (TTX) or plasticity challenge on mid-layer niches",
        hypothesis=(
            "Silencing network activity shrinks L3-plasticity cryptic programs; "
            "enriched sensory experience expands them — linking niche to plasticity state."
        ),
        system="acute cortical slice or organoid; optional chronic monocular deprivation analogue",
        targets=("network activity", "plasticity genes"),
        readouts=("GAP43/NRGN/ENC1 module", "uncertainty niche size", "IF HOPX"),
        primary_contrast="TTX vs vehicle; or enriched vs standard housing (pre-registered arm)",
        pass_criteria=(
            "plasticity module Δ in L3 cryptic ROI moves in pre-registered direction (padj≤0.05)",
            "geometry of layer boundaries stable (not a global tissue collapse)",
        ),
        claim_level_on_pass="F4",
        priority=2,
        n_replicates_min=3,
        controls=("vehicle", "time-matched untreated"),
        related_roi="ROI_151508_L3_n138",
    ),
    # ===== D2 — lineage =====
    ExperimentSpec(
        experiment_id="D2_lineage_hopx_ip",
        discovery_id="D2_L3_plasticity",
        class_="lineage",
        modality="lineage_reporter",
        title="HOPX+ intermediate progenitor lineage contribution to L3 cryptic niches",
        hypothesis=(
            "HOPX-lineage descendants are enriched in L3 cryptic plasticity niches "
            "versus adjacent L3 non-ROI."
        ),
        system="Hopx-CreERT2 lineage mouse (developmental pulse) or human organoid barcoded iPSC",
        targets=("HOPX", "ENC1", "SATB2"),
        readouts=("lineage density ROI vs same-layer", "co-expression with plasticity panel"),
        primary_contrast="lineage+ density in L3 ROI vs L3 non-ROI",
        pass_criteria=(
            "lineage enrichment in ROI (padj≤0.05, n≥3)",
            "lineage cells show higher plasticity module than non-lineage L3 neighbours",
        ),
        claim_level_on_pass="F3",
        priority=2,
        n_replicates_min=3,
        controls=("no-tamoxifen", "L2/L4 control ROIs"),
        related_roi="ROI_151508_L3_n138",
    ),
    # ===== D2 — orthogonal =====
    ExperimentSpec(
        experiment_id="D2_orthogonal_multiome_l3",
        discovery_id="D2_L3_plasticity",
        class_="orthogonal",
        modality="multiome_or_snrna_spatial",
        title="snRNA-seq/multiome + spatial joint confirmation of L3 sub-compartments",
        hypothesis=(
            "Independent single-cell modality recovers a mid-layer state with "
            "ENC1/HOPX/GAP43 program that spatially maps into cryptic L3 components."
        ),
        system="matched human DLPFC multiome or snRNA + Visium/Xenium same donor",
        targets=("ENC1", "HOPX", "GAP43", "GRIA2", "MBP"),
        readouts=("cluster marker table", "spatial deconvolution into L3 ROI", "module scores"),
        primary_contrast="state enriched in L3 ROI vs other L3 cells",
        pass_criteria=(
            "≥1 mid-layer state with plasticity module ↑ and myelin ↓ vs other L3 (padj≤0.05)",
            "spatial mapping enriches that state inside pre-registered L3 cryptic ROIs",
        ),
        claim_level_on_pass="F3",
        priority=1,
        n_replicates_min=2,
        controls=("white-matter nuclei", "L6 control"),
        related_roi="ROI_151508_L3_n138",
    ),
    # ===== D3 LN Ca2+ — perturbation =====
    ExperimentSpec(
        experiment_id="D3_crispr_kcnn4_orai3",
        discovery_id="D3_LN_ca2",
        class_="perturbation",
        modality="crispr",
        title="CRISPR KO/KD of KCNN4 and/or ORAI3 in LN organoids or tonsil explants",
        hypothesis=(
            "Loss of KCNN4/ORAI3 collapses the Ca²⁺/MAPK niche signature and reduces "
            "activation tone without converting the niche into a classical GC program."
        ),
        system="tonsil explant, LN organoid, or activated B/T co-culture with spatial readout",
        targets=("KCNN4", "ORAI3", "MAP2K5"),
        readouts=(
            "Ca²⁺ flux imaging",
            "panel KCNN4/ORAI3/MEF2A/BCL6",
            "phospho-ERK optional",
        ),
        primary_contrast="KO/KD vs AAVS1/safe-harbor control",
        pass_criteria=(
            "Ca²⁺ module score ↓ (padj≤0.05, n≥3)",
            "BCL6/MKI67 not significantly up (GC counter still holds)",
            "viability >70% of control",
        ),
        claim_level_on_pass="F4",
        priority=1,
        n_replicates_min=3,
        controls=("safe-harbor gRNA", "untreated explant"),
        related_roi="xenium_rank3_n31",
    ),
    ExperimentSpec(
        experiment_id="D3_drug_ca_mek_inhibitors",
        discovery_id="D3_LN_ca2",
        class_="perturbation",
        modality="drug",
        title="Ca²⁺ flux and MEK/ERK inhibitors on LN spatial niches",
        hypothesis=(
            "Pharmacologic blockade of store-operated Ca²⁺ entry or MEK reduces the "
            "cryptic Ca²⁺ niche program in situ."
        ),
        system="human tonsil/LN explant culture with CODEX or Xenium end-point",
        targets=("SOCE", "MEK1/2"),
        readouts=("KCNN4/ORAI3/MEF2A module", "pERK", "GC panel BCL6/Ki67"),
        primary_contrast="inhibitor vs vehicle; dose pre-registered",
        pass_criteria=(
            "Ca²⁺ module ↓ at pre-registered dose (padj≤0.05)",
            "GC panel not increased (no compensatory GC conversion)",
        ),
        claim_level_on_pass="F4",
        priority=2,
        n_replicates_min=3,
        controls=("vehicle", "inactive analogue if available"),
        related_roi="xenium_rank3_n31",
        notes="Example tool compounds: 2-APB/SOCE blockers; trametinib/MEK class — final choice by local pharmacology SOP.",
    ),
    # ===== D3 — lineage =====
    ExperimentSpec(
        experiment_id="D3_lineage_immune_barcode",
        discovery_id="D3_LN_ca2",
        class_="lineage",
        modality="lineage_barcode",
        title="Immune lineage barcoding / CITE-seq to assign niche cell of origin",
        hypothesis=(
            "The Ca²⁺ cryptic niche is enriched for a specific immune lineage "
            "(e.g. B or innate-like) rather than random LN parenchyma mixture."
        ),
        system="human LN/tonsil CITE-seq + spatial; or mouse lineage-traced immune subsets",
        targets=("MS4A1", "CD3E", "CD68", "KCNN4", "ORAI3"),
        readouts=("lineage fraction in niche vs rest", "module score per lineage"),
        primary_contrast="lineage composition niche vs bulk LN",
        pass_criteria=(
            "≥1 lineage shows enrichment OR exclusive high Ca²⁺ module (padj≤0.05)",
            "niche is not a random sample of bulk LN (χ² or permutation p≤0.05)",
        ),
        claim_level_on_pass="F3",
        priority=1,
        n_replicates_min=3,
        controls=("GC polygon cells", "adipose domain if present"),
        related_roi="xenium_rank3_n31",
    ),
    # ===== D3 — orthogonal =====
    ExperimentSpec(
        experiment_id="D3_orthogonal_codex_second_ln",
        discovery_id="D3_LN_ca2",
        class_="orthogonal",
        modality="codex_or_xenium",
        title="CODEX/IMC protein or second-donor Xenium confirmation",
        hypothesis=(
            "Protein-level KCNN4/ORAI3 (or second LN Xenium) recovers a non-GC "
            "cryptic niche with same-domain hard DE / protein elevation."
        ),
        system="second human LN donor; CODEX/IMC or Xenium",
        targets=("KCNN4", "ORAI3", "MEF2A", "BCL6", "CD20", "CD3"),
        readouts=("protein intensity ROI", "or Xenium same-domain DE"),
        primary_contrast="niche vs same LN domain; vs true GC",
        pass_criteria=(
            "KCNN4 or ORAI3 ↑ vs same-domain (padj≤0.05)",
            "BCL6 not GC-like high in niche vs true GC",
            "independent donor or independent platform from first Xenium run",
        ),
        claim_level_on_pass="F3",
        priority=1,
        n_replicates_min=2,
        controls=("pathologist-marked GC", "T-zone"),
        related_roi="xenium_rank3_n31",
    ),
)


def experiments_for(
    discovery_id: str | None = None,
    class_: str | None = None,
) -> list[ExperimentSpec]:
    rows = list(EXPERIMENTS)
    if discovery_id:
        rows = [e for e in rows if e.discovery_id == discovery_id]
    if class_:
        rows = [e for e in rows if e.class_ == class_]
    return sorted(rows, key=lambda e: (e.priority, e.experiment_id))


def registry_summary() -> dict[str, Any]:
    by_class: dict[str, int] = {}
    by_disc: dict[str, int] = {}
    for e in EXPERIMENTS:
        by_class[e.class_] = by_class.get(e.class_, 0) + 1
        by_disc[e.discovery_id] = by_disc.get(e.discovery_id, 0) + 1
    return {
        "protocol": "histoweave.functional_experiments.v1",
        "n_experiments": len(EXPERIMENTS),
        "by_class": by_class,
        "by_discovery": by_disc,
        "claim_levels": sorted({e.claim_level_on_pass for e in EXPERIMENTS}),
    }


__all__ = [
    "EXPERIMENTS",
    "ExperimentSpec",
    "experiments_for",
    "registry_summary",
]
