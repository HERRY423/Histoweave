"""Pre-registered functional modules for cryptic-niche interpretation.

These gene sets are *hypothesis scaffolds* for disease mechanism and
developmental / spatial-organisation claims. They are deliberately small,
literature-grounded, and frozen before scoring so that enrichment cannot be
tuned post hoc.

Sources (representative; not a systematic review):
* Myelin / oligodendrocyte: MBP, PLP1, MOBP, CNP, MAG (CNS myelination canon)
* Laminar / plasticity programs: ENC1, HOPX, GAP43, GRIA2, NRGN, CAMK2*
* Ca²⁺ / immune LN: KCNN4, ORAI3, MAP2K5, MEF2A, FCGR2B, MALT1, ALOX5
* Disease maps: demyelinating disease, neuroinflammation, LN autoimmunity

Scoring is hypergeometric-style over the *tested gene universe* of each DE
table (genes present in the marker CSV), not the whole genome — honest for
Visium/Xenium panels.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionalModule:
    """One pre-registered functional gene set."""

    module_id: str
    axis: str  # disease | development | spatial_organisation | immune
    title: str
    genes: frozenset[str]
    direction: str  # up | down | either — expected enrichment direction in niche
    disease_links: tuple[str, ...] = ()
    organisation_principle: str = ""
    validation_next: str = ""


# ---------------------------------------------------------------------------
# Module registry (frozen)
# ---------------------------------------------------------------------------

MODULES: tuple[FunctionalModule, ...] = (
    # --- D1 L6 myelin / disease ---
    FunctionalModule(
        module_id="myelin_sheath_core",
        axis="disease",
        title="CNS myelin sheath core",
        genes=frozenset(
            {
                "MBP",
                "PLP1",
                "MOBP",
                "MAG",
                "CNP",
                "MOG",
                "CLDN11",
                "OPALIN",
                "GJC2",
                "TF",
            }
        ),
        direction="up",
        disease_links=(
            "multiple sclerosis / demyelination vulnerability",
            "leukodystrophy & myelin maintenance failure",
            "age-related white-matter rarefaction at grey–deep interfaces",
        ),
        organisation_principle=(
            "Layer 6 is not a molecularly uniform mantle: multi-method uncertainty "
            "isolates a compact myelin-rich micro-compartment *inside* L6 that "
            "single-partition anatomy folds into one label."
        ),
        validation_next="IF MBP/PLP1 on ROI_151508_L6; optional MAG; spatial proteomics",
    ),
    FunctionalModule(
        module_id="oligodendrocyte_lineage",
        axis="development",
        title="Oligodendrocyte / myelinating lineage",
        genes=frozenset(
            {
                "OLIG1",
                "OLIG2",
                "SOX10",
                "PDGFRA",
                "CSPG4",
                "BCAS1",
                "ENPP6",
                "MYRF",
                "MBP",
                "PLP1",
                "MOBP",
            }
        ),
        direction="up",
        disease_links=("remyelination failure after injury", "OPC differentiation blocks"),
        organisation_principle=(
            "Developmental myelination is spatially punctate; adult L6 cryptic niches "
            "may retain or re-deploy oligodendrocyte programs as micro-domains."
        ),
        validation_next="IF SOX10/OLIG2 + MBP co-stain; RNAscope on same ROI",
    ),
    FunctionalModule(
        module_id="mitochondrial_stress",
        axis="disease",
        title="Mitochondrial / oxidative stress (down in myelin niche)",
        genes=frozenset(
            {
                "VDAC2",
                "ATP5PD",
                "COX6C",
                "NDUFA4",
                "NDUFA13",
                "SLC25A4",
                "PET100",
                "ATP1A3",
            }
        ),
        direction="down",
        disease_links=(
            "metabolic vulnerability of deep cortex",
            "mitochondrial neuropathies intersecting myelinated compartments",
        ),
        organisation_principle=(
            "Myelin-rich L6 micro-domains co-vary with *relative* depletion of "
            "neuronal/mitochondrial transcripts — a compartment trade-off, not noise."
        ),
        validation_next="Multiplex IF MBP + mitochondrial markers; metabolic imaging optional",
    ),
    # --- D2 L3 program / plasticity / development ---
    FunctionalModule(
        module_id="midlayer_plasticity",
        axis="development",
        title="Mid-layer plasticity / synaptic program",
        genes=frozenset(
            {
                "ENC1",
                "HOPX",
                "GAP43",
                "GRIA2",
                "NRGN",
                "CAMK2B",
                "CAMK2A",
                "NEFL",
                "RGS4",
                "GNG3",
                "SNCB",
                "CACNG3",
                "CHL1",
                "PVALB",
            }
        ),
        direction="up",
        disease_links=(
            "cortical plasticity disorders",
            "layer-selective neurodegeneration risk (e.g. selective mid-layer stress)",
            "psychiatric intermediate phenotypes with laminar expression bias",
        ),
        organisation_principle=(
            "Manual Layer 3 is a developmental *zone*, not a single state: cryptic "
            "niches concentrate plasticity/synaptic transcripts *within* L3 while "
            "suppressing myelin — redefining L3 as multi-compartment."
        ),
        validation_next="IF ENC1/HOPX vs same-layer L3; RNAscope GAP43/GRIA2",
    ),
    FunctionalModule(
        module_id="astroglial_boundary",
        axis="spatial_organisation",
        title="Astroglial / boundary program (often anti-correlated in L3 niche)",
        genes=frozenset({"GFAP", "S100B", "AQP4", "ALDH1L1", "SLC1A2", "SLC1A3", "GJA1"}),
        direction="down",
        disease_links=("reactive gliosis boundaries", "glial scar interfaces"),
        organisation_principle=(
            "Cryptic L3 niches are depleted for astroglial boundary markers relative "
            "to rest — consistent with *interior* mid-layer programs, not edge ribbons."
        ),
        validation_next="IF GFAP vs ENC1 on L3 ROI to confirm anti-correlation",
    ),
    FunctionalModule(
        module_id="vascular_ecm",
        axis="spatial_organisation",
        title="Vascular / ECM niche (exploratory in L3)",
        genes=frozenset(
            {
                "MGP",
                "COL1A2",
                "FABP4",
                "PECAM1",
                "VWF",
                "CLDN5",
                "SPARC",
                "COL4A1",
                "FN1",
            }
        ),
        direction="either",
        disease_links=("BBB microheterogeneity", "vascular cognitive impairment interfaces"),
        organisation_principle=(
            "Some L3 cryptic components co-enrich ECM/vascular transcripts (MGP, COL1A2, "
            "FABP4), suggesting *neurovascular micro-patches* inside a laminar label."
        ),
        validation_next="IF CD31/CLDN5 + ENC1; exclude sectioning vessel artifact",
    ),
    # --- D3 Xenium LN ---
    FunctionalModule(
        module_id="ca2_mapk_signaling",
        axis="disease",
        title="Ca²⁺ / MAPK transcriptional module",
        genes=frozenset(
            {
                "KCNN4",
                "ORAI3",
                "MAP2K5",
                "MEF2A",
                "CAMK2D",
                "CAMK4",
                "ITPR1",
                "ITPR2",
                "STIM1",
                "ORAI1",
                "PRKCB",
                "NFAT5",
                "NFATC1",
                "NFATC2",
            }
        ),
        direction="up",
        disease_links=(
            "lymphocyte activation / exhaustion tone",
            "autoimmune LN hyper-responsiveness",
            "B-cell receptor proximal Ca²⁺ flux disorders",
        ),
        organisation_principle=(
            "Pathology 'Lymph node' is not a single molecular field: uncertainty "
            "isolates a Ca²⁺/MAPK-high micro-niche *inside* bulk parenchyma that "
            "GC/B/T polygon programs miss. KCNN4 (KCa3.1) + ORAI3 (CRAC) mark "
            "activation-linked Ca²⁺ throughput; neighbourhood analysis shows "
            "T-like external enrichment with rare GC-like contacts "
            "(see discovery_xenium_lymph/KCNN4_ORAI3_NEIGHBORHOOD.md)."
        ),
        validation_next=(
            "CODEX KCNN4+ORAI3+CD3+CD20+PDPN+BCL6; test T-rim protein co-localisation"
        ),
    ),
    FunctionalModule(
        module_id="innate_effector_ln",
        axis="immune",
        title="Innate / effector LN module",
        genes=frozenset(
            {
                "FCGR2B",
                "ALOX5",
                "MARCO",
                "MALT1",
                "HHEX",
                "NF2",
                "COBLL1",
                "RXRA",
                "TRIB2",
            }
        ),
        direction="up",
        disease_links=(
            "innate–adaptive interface pathology",
            "LN sinus / macrophage niche remodeling in inflammation",
        ),
        organisation_principle=(
            "Cryptic LN niches can be innate-skewed rather than classical GC — "
            "redefining 'LN parenchyma' as multi-niche immune architecture."
        ),
        validation_next="CODEX panel FCGR2B/MARCO/CD68 vs BCL6 to exclude GC mislabel",
    ),
    FunctionalModule(
        module_id="classical_gc_counter",
        axis="spatial_organisation",
        title="Classical GC counter-program (expect NOT enriched in D3)",
        genes=frozenset(
            {
                "BCL6",
                "MKI67",
                "TOP2A",
                "PCNA",
                "LMO2",
                "CXCL13",
                "AICDA",
                "RGS13",
            }
        ),
        # Direction "down" is the scientific expectation, but scoring uses the
        # pre-declared *negative-control non-enrichment* rule in
        # run_functional_validation.GC_COUNTER_RULES (not hypergeom of downs):
        # zero significant UP + mean log2FC ≤ 0. See FUNCTIONAL_VALIDATION.md
        # § Statistical note: D3 GC counter.
        direction="down",
        disease_links=("GC hyperplasia differential diagnosis",),
        organisation_principle=(
            "If classical GC is *not* enriched while Ca²⁺/MAPK is, the niche is a "
            "**new organisational unit**, not a missed germinal center polygon."
        ),
        validation_next="Confirm BCL6/Ki67 low on ROI IF relative to true GC",
    ),
)

MODULE_BY_ID = {m.module_id: m for m in MODULES}

# Which modules apply to which discovery.
# Optional direction override: (module_id, direction) where direction is
# "up" | "down" | "either" | None (use module default).
DiscoveryModuleSpec = str | tuple[str, str | None]

DISCOVERY_MODULES: dict[str, tuple[DiscoveryModuleSpec, ...]] = {
    "D1_L6_myelin": (
        "myelin_sheath_core",
        "oligodendrocyte_lineage",
        "mitochondrial_stress",
    ),
    "D2_L3_plasticity": (
        "midlayer_plasticity",
        "astroglial_boundary",
        "vascular_ecm",
        ("myelin_sheath_core", "down"),  # L3 niche depletes myelin vs rest
    ),
    "D3_LN_ca2": (
        "ca2_mapk_signaling",
        "innate_effector_ln",
        ("classical_gc_counter", "either"),  # absence / non-enrichment of GC
    ),
}
