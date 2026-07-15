"""Canonical compiler examples shared by the mock provider and LLM prompts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _step(category: str, method: str, purpose: str) -> dict[str, Any]:
    return {"category": category, "method": method, "params": {}, "purpose": purpose}


_BASE = (
    _step("qc", "basic_qc", "Filter and audit low-quality observations."),
    _step("normalization", "log1p_cp10k", "Normalize library size and log-transform."),
)

_TemplateRow = tuple[
    str,
    str,
    tuple[str, ...],
    tuple[dict[str, Any], ...],
    list[dict[str, str]],
    str,
]

_ROWS: tuple[_TemplateRow, ...] = (
    (
        "tumor",
        "Map tumor architecture and annotate malignant and stromal spatial domains.",
        (
            "tumor",
            "tumour",
            "cancer",
            "carcinoma",
            "malignant",
            "annotate",
            "annotation",
            "cell type",
            "cell-type",
        ),
        (
            _step("domain_detection", "banksy", "Detect spatial tumor compartments."),
            _step("annotation", "marker_score", "Annotate malignant and stromal states."),
        ),
        [],
        "Resolve spatial compartments before assigning marker-supported tumor states.",
    ),
    (
        "brain",
        "Identify cortical layers and spatially variable genes in the brain.",
        (
            "brain",
            "cortical",
            "cortex",
            "neuron",
            "glia",
            "spatially variable",
            "spatial variable",
            "variable gene",
            "svg",
        ),
        (
            _step("domain_detection", "banksy", "Resolve spatially coherent brain layers."),
            _step("svg", "spatialde", "Rank genes varying across brain anatomy."),
        ),
        [],
        "Detect spatial layers before ranking anatomy-associated expression programs.",
    ),
    (
        "developmental",
        "Segment nuclei and resolve developmental tissue domains and cell identities.",
        (
            "development",
            "developmental",
            "embryo",
            "embryonic",
            "organoid",
            "morphogenesis",
            "segment",
            "segmentation",
            "nuclei",
            "nucleus",
        ),
        (
            _step("segmentation", "cellpose2", "Segment nuclei in the tissue image."),
            _step("domain_detection", "banksy", "Resolve developmental tissue domains."),
            _step("annotation", "marker_score", "Annotate emerging cell identities."),
        ),
        [],
        "Segment cells, resolve spatial domains, then assign developmental identities.",
    ),
    (
        "immune",
        "Map immune cell mixtures and spatial ligand-receptor communication.",
        (
            "immune",
            "lymphocyte",
            "macrophage",
            "deconvol",
            "cell proportion",
            "ligand",
            "receptor",
            "communication",
            "immune-escape",
            "immune escape",
        ),
        (
            _step("domain_detection", "banksy", "Detect immune spatial compartments."),
            _step("deconvolution", "marker_deconv", "Estimate immune cell-type mixtures."),
            _step("neighborhood", "spatial_graph", "Build the immune neighbourhood graph."),
            _step("ccc", "liana_plus", "Rank spatial ligand-receptor interactions."),
        ),
        [],
        "Estimate immune composition and neighbourhoods before communication inference.",
    ),
    (
        "drug",
        "Compare treated tissue to map spatial drug-response programs.",
        ("drug", "treatment", "treated", "therapy", "therapeutic", "perturbation"),
        (
            _step("integration", "combat", "Control expression-level treatment batches."),
            _step("domain_detection", "banksy", "Map spatial response compartments."),
            _step("svg", "spatialde", "Rank spatially patterned response genes."),
        ),
        [
            {
                "concept": "condition-aware drug-response testing",
                "reason": "the registry has no differential-treatment testing category",
                "degraded_to": "ComBat-integrated domains plus spatially variable gene ranking",
            }
        ],
        "Control batch structure, map response domains, and rank spatial response programs.",
    ),
    (
        "cross_section",
        "Integrate serial cross-sections and find conserved spatial domains.",
        (
            "cross-section",
            "cross section",
            "serial section",
            "serial tissue",
            "multiple section",
            "across section",
            "batch",
            "integrat",
            "combat",
            "harmony",
        ),
        (
            _step("integration", "combat", "Correct expression shifts across sections."),
            _step("domain_detection", "banksy", "Detect conserved spatial domains."),
        ),
        [
            {
                "concept": "geometric registration across tissue sections",
                "reason": "the registry integrates expression but does not align tissue geometry",
                "degraded_to": "expression-level ComBat correction followed by BANKSY domains",
            }
        ],
        "Correct section-level expression shifts before detecting conserved domains.",
    ),
    (
        "generic",
        "Find spatial domains in this tissue.",
        (),
        (_step("domain_detection", "banksy", "Detect spatial domains."),),
        [],
        "Apply the standard quality-control, normalization, and domain workflow.",
    ),
)

MOCK_TEMPLATES: tuple[dict[str, Any], ...] = tuple(
    {
        "name": name,
        "question": question,
        "keywords": keywords,
        "plan": {
            "rationale": rationale,
            "steps": [*_BASE, *steps],
            "gaps": gaps,
            "assay_assumed": "unknown",
        },
    }
    for name, question, keywords, steps, gaps, rationale in _ROWS
)


def template_for_question(question: str) -> dict[str, Any]:
    """Return a defensive copy of the first matching canonical plan."""
    folded = question.casefold()
    for template in MOCK_TEMPLATES:
        if not template["keywords"] or any(word in folded for word in template["keywords"]):
            return deepcopy(template["plan"])
    raise AssertionError("the final compiler template must be a catch-all")
