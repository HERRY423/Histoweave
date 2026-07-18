"""Method guide inventory must cover every registered plugin."""

from __future__ import annotations

from pathlib import Path

from histoweave.plugins import list_methods

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "docs" / "methods"
CATALOG = METHODS_DIR / "catalog.md"
GENERATED = METHODS_DIR / "generated"
CATEGORIES = METHODS_DIR / "categories"


def test_catalog_and_generated_pages_cover_all_registered_methods() -> None:
    methods = list_methods()
    assert methods, "registry is empty"
    assert CATALOG.is_file(), "docs/methods/catalog.md missing — run generate_method_docs.py"
    catalog_text = CATALOG.read_text(encoding="utf-8")

    missing_catalog: list[str] = []
    missing_pages: list[str] = []
    for method in methods:
        name = method["name"]
        if f"`{name}`" not in catalog_text:
            missing_catalog.append(name)
        page = GENERATED / f"{name}.md"
        # slug is usually the raw name; accept either
        if not page.is_file():
            # try simple slug
            alt = GENERATED / f"{name.replace('/', '_')}.md"
            if not alt.is_file():
                missing_pages.append(name)

    assert not missing_catalog, f"methods missing from catalog.md: {missing_catalog}"
    assert not missing_pages, f"methods missing generated pages: {missing_pages}"
    assert len(list(GENERATED.glob("*.md"))) >= len(methods)


def test_every_category_has_a_guide_page() -> None:
    categories = {m["category"] for m in list_methods()}
    missing = [c for c in sorted(categories) if not (CATEGORIES / f"{c}.md").is_file()]
    assert not missing, f"missing category guides: {missing}"


def test_method_guide_index_points_at_catalog() -> None:
    index = (METHODS_DIR / "index.md").read_text(encoding="utf-8")
    assert "catalog.md" in index
    assert "registered" in index.lower()


def test_contributing_and_code_of_conduct_exist() -> None:
    assert (ROOT / "CONTRIBUTING.md").is_file()
    assert (ROOT / "CODE_OF_CONDUCT.md").is_file()
    contrib = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    coc = (ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert "Code of Conduct" in contrib
    assert "release_manifest" in contrib
    assert "generate_method_docs" in contrib
    assert "Contributor Covenant" in coc
    assert "conduct@histoweave-spatial.org" in coc
    assert "Enforcement guidelines" in coc
    # Docs site mirrors
    assert (ROOT / "docs" / "contributing.md").is_file()
    assert (ROOT / "docs" / "code-of-conduct.md").is_file()


def test_generated_docs_have_no_print_calls() -> None:
    """Repo logging contract scans docs for print( — keep guides clean."""
    offenders: list[str] = []
    for path in METHODS_DIR.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        if "print(" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"method docs must not document print(: {offenders}"
