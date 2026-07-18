"""Generate method catalog + per-category guides from the live plugin registry.

Run from the repository root::

    python scripts/generate_method_docs.py

Outputs (checked into git so docs build without a live install)::

    docs/methods/catalog.md
    docs/methods/categories/*.md
    docs/methods/generated/*.md   # one page per registered method

Hand-written pages under ``docs/methods/*.md`` (e.g. banksy_py.md) take
precedence in the index for deep "when to use" guidance; generated pages
always cover 100% of registered methods so the guide inventory cannot lag.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.plugins import list_methods  # noqa: E402
from histoweave.plugins.coverage import method_coverage_report  # noqa: E402

logger = logging.getLogger(__name__)

METHODS_DIR = ROOT / "docs" / "methods"
CATALOG_PATH = METHODS_DIR / "catalog.md"
CATEGORIES_DIR = METHODS_DIR / "categories"
GENERATED_DIR = METHODS_DIR / "generated"

# Prefer these hand-written deep guides when linking from the index.
HAND_WRITTEN = {
    "banksy_py",
    "spectral",
    "gaussian_mixture",
    "spagcn",
    "cell2location",
}

CATEGORY_BLURBS = {
    "domain_detection": (
        "Recover contiguous spatial domains / layers / niches. "
        "Prefer non-oracle *K* estimation in real analyses "
        "(`k_policy='estimate'`; see statistical-review)."
    ),
    "svg": (
        "Rank genes by spatial autocorrelation or spatial variance. "
        "Always report FDR-adjusted p-values when available."
    ),
    "deconvolution": (
        "Estimate cell-type proportions in multi-cellular spots "
        "(Visium-scale). Requires a scRNA reference for most SOTA methods."
    ),
    "annotation": (
        "Assign cell / spot labels from expression (and optionally spatial "
        "context). Distinct task from spatial-domain recovery."
    ),
    "qc": "Filter low-quality observations and genes before modelling.",
    "normalization": "Stabilize library-size and technical scale differences.",
    "ingestion": "Load vendor outputs into a SpatialTable / SpatialData path.",
    "integration": "Batch correction and multi-modal representation learning.",
    "neighborhood": "Build spatial graphs used by domain / SVG methods.",
    "segmentation": "Image-based cell segmentation for imaging assays.",
    "ccc": "Cell–cell communication scoring on spatial neighbourhoods.",
}

WHEN_TO = {
    "domain_detection": {
        "validated": "Default shortlist for spatial-domain recovery with evidence.",
        "production": "Reliable baseline / capacity ablation; known failure modes.",
        "beta": "Field or SOTA comparator; pin backends and isolated envs.",
        "experimental": "Research incubator only — not for primary claims.",
    },
    "svg": {
        "production": "Fast classical spatial statistics with FDR hooks.",
        "beta": "Published SVG method; install the optional extra.",
        "experimental": "Research SVG variants — dual-report against morans_i.",
    },
    "deconvolution": {
        "beta": "Published deconvolution with a scRNA reference atlas.",
        "experimental": "Teaching / baseline proportion recovery only.",
        "production": "Stable baseline for smoke tests.",
    },
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def _link_for(method: dict, *, prefix: str = "") -> str:
    name = method["name"]
    if name in HAND_WRITTEN:
        target = f"{name}.md"
    else:
        target = f"generated/{_slug(name)}.md"
    return f"[{name}]({prefix}{target})"


def _when_to_use(method: dict) -> list[str]:
    cat = method["category"]
    mat = method["maturity"]
    bullets = []
    cat_map = WHEN_TO.get(cat, {})
    if mat in cat_map:
        bullets.append(cat_map[mat])
    elif cat in CATEGORY_BLURBS:
        bullets.append(CATEGORY_BLURBS[cat].split(".")[0] + ".")
    summary = (method.get("summary") or "").strip()
    if summary:
        bullets.append(summary.rstrip(".") + ".")
    track = (method.get("metadata") or {}).get("track")
    if track == "sota":
        bullets.append("SOTA track — fail-closed if the official backend is missing.")
    if track == "research":
        bullets.append("Research incubator — do not treat as a production recommendation.")
    if track == "baseline":
        bullets.append("Teaching / baseline method — not a field-standard claim.")
    if method.get("implementation") == "external" and method.get("backends"):
        backends = ", ".join(b["name"] for b in method["backends"])
        bullets.append(f"Requires external backend(s): `{backends}`.")
    return bullets or ["See MethodSpec summary in the registry."]


def _when_not(method: dict) -> list[str]:
    notes = []
    mat = method["maturity"]
    if mat == "experimental":
        notes.append("Do not use as the sole method for a primary manuscript claim.")
    if method.get("implementation") == "external":
        notes.append("Skip if optional extras / containers are not installed (fail-closed).")
    param_names = {
        p.get("name") if isinstance(p, dict) else getattr(p, "name", None)
        for p in (method.get("params") or [])
    }
    if method["category"] == "domain_detection" and "n_domains" in param_names:
        notes.append("Avoid oracle *K* in real analyses; estimate or justify fixed *K*.")
    if not notes:
        notes.append("Prefer alternatives listed in the category guide when assumptions fail.")
    return notes


def render_method_page(method: dict) -> str:
    name = method["name"]
    lines = [
        "<!-- Auto-generated by scripts/generate_method_docs.py — do not edit by hand. -->",
        f"# {name}",
        "",
        f"**Category:** `{method['category']}` · "
        f"**Maturity:** `{method['maturity']}` · "
        f"**Implementation:** `{method['implementation']}` · "
        f"**Version:** `{method['version']}`",
        "",
        (method.get("summary") or "_No summary declared._").strip(),
        "",
        "## When to use",
        "",
    ]
    for b in _when_to_use(method):
        lines.append(f"- {b}")
    lines += ["", "## When not to use", ""]
    for b in _when_not(method):
        lines.append(f"- {b}")

    wraps = method.get("wraps") or ""
    if wraps:
        lines += ["", f"**Wraps:** `{wraps}`"]
    modalities = method.get("modalities") or []
    if modalities:
        lines += ["", f"**Modalities:** {', '.join(f'`{m}`' for m in modalities)}"]
    params = method.get("params") or []
    if params:
        lines += [
            "",
            "## Parameters",
            "",
            "| Name | Type | Default | Description |",
            "|------|------|---------|-------------|",
        ]
        for p in params:
            if isinstance(p, dict):
                help_text = str(p.get("help") or p.get("description") or "").replace("|", "\\|")
                lines.append(
                    f"| `{p.get('name', '')}` | `{p.get('type', '')}` | "
                    f"`{p.get('default', '')}` | {help_text} |"
                )
            else:
                lines.append(f"| `{getattr(p, 'name', p)}` | | | |")
    assumptions = method.get("assumptions") or []
    if assumptions:
        lines += ["", "## Assumptions", ""]
        for a in assumptions:
            lines.append(f"- {a}")
    meta = method.get("metadata") or {}
    if meta.get("validation_evidence"):
        ev = meta["validation_evidence"]
        lines += ["", "## Evidence", ""]
        if isinstance(ev, dict):
            for k, v in ev.items():
                lines.append(f"- **{k}:** {v}")
        else:
            lines.append(str(ev))
    lines += [
        "",
        "---",
        "",
        "See also: [Method catalog](../catalog.md) · "
        f"[Category: {method['category']}](../categories/{method['category']}.md) · "
        "[Method selection](../../method-selection.md)",
        "",
    ]
    return "\n".join(lines)


def render_category_page(category: str, methods: list[dict]) -> str:
    blurb = CATEGORY_BLURBS.get(category, "")
    lines = [
        "<!-- Auto-generated by scripts/generate_method_docs.py -->",
        f"# {category.replace('_', ' ').title()}",
        "",
        blurb,
        "",
        f"**{len(methods)} registered method(s)** in this category.",
        "",
        "| Method | Maturity | Implementation | Summary | Guide |",
        "|--------|----------|----------------|---------|-------|",
    ]
    for m in sorted(methods, key=lambda x: (x["maturity"], x["name"])):
        summary = (m.get("summary") or "").replace("|", "\\|")
        if len(summary) > 80:
            summary = summary[:77] + "..."
        lines.append(
            f"| `{m['name']}` | `{m['maturity']}` | `{m['implementation']}` | "
            f"{summary} | {_link_for(m, prefix='../')} |"
        )
    lines += ["", "## Decision notes", ""]
    for m in sorted(methods, key=lambda x: x["name"]):
        lines.append(f"### `{m['name']}`")
        lines.append("")
        for b in _when_to_use(m)[:3]:
            lines.append(f"- {b}")
        lines.append(f"- Full page: {_link_for(m, prefix='../')}")
        lines.append("")
    lines += [
        "---",
        "",
        "[Full catalog](../catalog.md) · [Method guide index](../index.md)",
        "",
    ]
    return "\n".join(lines)


def render_catalog(methods: list[dict], report: dict) -> str:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in methods:
        by_cat[m["category"]].append(m)
    lines = [
        "<!-- Auto-generated by scripts/generate_method_docs.py — do not edit by hand. -->",
        "# Method catalog (complete inventory)",
        "",
        f"This catalog lists **all {len(methods)} registered methods** in the live "
        "plugin registry. It is generated from `histoweave.plugins.list_methods()` "
        "so it cannot silently lag the code.",
        "",
        "## Coverage snapshot",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total methods | {report['total_methods']} |",
        f"| Release (production ∪ beta ∪ validated) | {report['counts']['release_methods']} |",
        f"| Research incubator | {report['counts']['research_candidates']} |",
        f"| Baseline / teaching | {report['counts']['baseline_methods']} |",
        f"| Unclassified | {report['counts']['unclassified_methods']} |",
        f"| SOTA track plugins | {report['counts']['sota_plugins']} |",
        "",
    ]
    if report["counts"]["unclassified_methods"]:
        lines.append(
            '!!! warning "Unclassified methods"\n\n    '
            + ", ".join(f"`{n}`" for n in report["unclassified_names"])
            + "\n"
        )
    else:
        lines.append(
            "All registered methods are classified into a release-manifest track "
            "(production / beta / validated / research / baseline).\n"
        )
    lines += [
        "## By category",
        "",
    ]
    for cat in sorted(by_cat):
        lines.append(f"- [{cat}](categories/{cat}.md) ({len(by_cat[cat])})")
    lines += ["", "## Full table", ""]
    lines += [
        "| Category | Method | Maturity | Track | Implementation | Guide |",
        "|----------|--------|----------|-------|----------------|-------|",
    ]
    for m in sorted(methods, key=lambda x: (x["category"], x["name"])):
        track = (m.get("metadata") or {}).get("track") or m["maturity"]
        lines.append(
            f"| `{m['category']}` | `{m['name']}` | `{m['maturity']}` | "
            f"`{track}` | `{m['implementation']}` | {_link_for(m)} |"
        )
    lines += [
        "",
        "---",
        "",
        "Regenerate: `python scripts/generate_method_docs.py`",
        "",
    ]
    return "\n".join(lines)


def render_index(methods: list[dict]) -> str:
    """Refresh the human-facing index while preserving deep-guide callouts."""
    field = [
        m
        for m in methods
        if m["maturity"] in {"validated", "beta", "production"}
        and m["category"]
        in {
            "domain_detection",
            "svg",
            "deconvolution",
            "annotation",
            "segmentation",
            "ccc",
            "integration",
        }
    ]
    lines = [
        "# Method guide (when to use / when not)",
        "",
        "HistoWeave documents **every registered method**. Use this page to start,",
        "the [complete catalog](catalog.md) for the full inventory, and category",
        "pages for decision notes within a task.",
        "",
        f"**Inventory:** {len(methods)} registered · "
        f"{len(field)} field-facing (production/beta/validated in analysis categories) · "
        f"{len(HAND_WRITTEN)} deep hand-written guides.",
        "",
        "## Deep guides (start here)",
        "",
        "| Method | Maturity | Start here when… |",
        "|--------|----------|------------------|",
        "| [BANKSY (native)](banksy_py.md) | validated | Robust spatial-domain default without R |",
        (
            "| [Spectral clustering](spectral.md) | validated | "
            "Contiguous domains, known/defensible *k* |"
        ),
        (
            "| [Gaussian mixture](gaussian_mixture.md) | validated | "
            "Soft domains / elliptical compartments |"
        ),
        "| [SpaGCN](spagcn.md) | beta | Visium-scale graph-conv SOTA comparison |",
        "| [cell2location](cell2location.md) | beta | Spot deconvolution with a scRNA reference |",
        "",
        "## Complete coverage",
        "",
        "- **[Full catalog](catalog.md)** — every registered method in one table",
        "- **Category guides:**",
        "",
    ]
    by_cat: dict[str, int] = defaultdict(int)
    for m in methods:
        by_cat[m["category"]] += 1
    for cat in sorted(by_cat):
        lines.append(f"    - [{cat}](categories/{cat}.md) ({by_cat[cat]})")
    lines += [
        "",
        '!!! tip "Selection under uncertainty"',
        "    Prefer a short multi-method ensemble + boundary-uncertainty map when two",
        "    configurations are within ~0.03 ARI, or when the recommender does **not**",
        "    beat the global-best baseline.  Use non-oracle *K* (`k_policy='estimate'`)",
        "    for realistic domain benchmarks — see [Statistical review](../statistical-review.md).",
        "",
        "Related: [Method selection guide](../method-selection.md) ·",
        "[Method lifecycle](../method-lifecycle.md) ·",
        "[Research incubator](../research-methods.md) ·",
        "[Contributing](https://github.com/HERRY423/Histoweave/blob/main/CONTRIBUTING.md)",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    methods = list_methods()
    report = method_coverage_report()
    CATEGORIES_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Purge stale generated pages so renames do not leave orphans.
    for path in GENERATED_DIR.glob("*.md"):
        path.unlink()

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for m in methods:
        by_cat[m["category"]].append(m)
        page = render_method_page(m)
        (GENERATED_DIR / f"{_slug(m['name'])}.md").write_text(page, encoding="utf-8")

    for cat, rows in by_cat.items():
        (CATEGORIES_DIR / f"{cat}.md").write_text(render_category_page(cat, rows), encoding="utf-8")

    CATALOG_PATH.write_text(render_catalog(methods, report), encoding="utf-8")
    (METHODS_DIR / "index.md").write_text(render_index(methods), encoding="utf-8")

    logger.info(
        "generated catalog + %s method pages + %s category pages",
        len(methods),
        len(by_cat),
    )
    if report["counts"]["unclassified_methods"]:
        logger.warning("unclassified methods: %s", report["unclassified_names"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
