#!/usr/bin/env python3
"""Build / verify the in-repo reference-artefact MANIFEST.

Writes ``reference_artefacts/MANIFEST.json`` listing every tracked summary file
under the four primary artefact directories, with size and SHA-256.

Usage
-----
::

    python scripts/build_reference_artefact_manifest.py
    python scripts/build_reference_artefact_manifest.py --check   # CI / local gate

Exit code 1 on --check if files are missing, oversized, or hash-mismatched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "reference_artefacts" / "MANIFEST.json"

# Directory → protocol label (documentation only)
ARTEFACT_ROOTS: dict[str, str] = {
    "independent_personalisation_results": "histoweave.independent_personalisation.v1",
    "protocol_endpoints_results": "histoweave.protocol_endpoints.bundle",
    "non_oracle_k_sota": "histoweave.non_oracle_k_sota.v1",
    "pareto_isus_results": "histoweave.pareto_isus.reference",
    "benchmark_external_validation": "histoweave.external_validation.recommender_loocv.v1",
    "parallel_experiment_table": "histoweave.parallel_experiment_table.v1",
}

# Extensions allowed inside artefact roots for the manifest
_ALLOWED_SUFFIX = {".json", ".csv", ".md", ".svg", ".png", ".py", ".txt"}

# Soft size budget (bytes) for any single tracked summary file
_MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MiB — summaries must stay small

# Files that must exist for a green verification gate
_REQUIRED: tuple[str, ...] = (
    "independent_personalisation_results/independent_personalisation_summary.json",
    "independent_personalisation_results/independent_personalisation_report.md",
    "independent_personalisation_results/cross_lab_reproducibility.json",
    "protocol_endpoints_results/protocol_endpoints_summary.json",
    "protocol_endpoints_results/protocol_endpoints_report.md",
    "protocol_endpoints_results/oracle_k_leakage.json",
    "protocol_endpoints_results/selective_regret_coverage.json",
    "non_oracle_k_sota/summary.json",
    "non_oracle_k_sota/benchmark_long.csv",
    "pareto_isus_results/pareto_report.json",
    "pareto_isus_results/isus_calibration.json",
    "benchmark_external_validation/decision_validation.json",
    "parallel_experiment_table/parallel_experiment_summary.csv",
    "parallel_experiment_table/parallel_experiment_table.csv",
    "parallel_experiment_table/report_parallel_experiment.md",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dirname, protocol in ARTEFACT_ROOTS.items():
        root = ROOT / dirname
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _ALLOWED_SUFFIX:
                continue
            # Skip local logs if any slipped in
            if path.name.endswith(".log"):
                continue
            rel = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
            rows.append(
                {
                    "path": rel,
                    "bytes": size,
                    "sha256": _sha256(path),
                    "protocol_family": protocol,
                    "required": rel in _REQUIRED,
                }
            )
    return rows


def build_manifest() -> dict[str, object]:
    files = _collect()
    return {
        "schema": "histoweave.reference_artefacts.manifest.v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {
            "track_summaries": True,
            "ignore_raw_h5ad": True,
            "max_file_bytes": _MAX_FILE_BYTES,
            "note": (
                "Summary JSON/MD/CSV/figures live in git. "
                "Raw expression matrices (*.h5ad) and /data/ stay local."
            ),
        },
        "required": list(_REQUIRED),
        "n_files": len(files),
        "total_bytes": sum(int(r["bytes"]) for r in files),
        "files": files,
    }


def write_manifest(payload: dict[str, object]) -> Path:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return MANIFEST_PATH


def check_manifest() -> list[str]:
    """Return a list of error strings (empty ⇒ OK)."""
    errors: list[str] = []
    for rel in _REQUIRED:
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing required artefact: {rel}")
            continue
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            errors.append(
                f"oversized artefact {rel}: {size} bytes > {_MAX_FILE_BYTES}"
            )
        if size == 0:
            errors.append(f"empty artefact: {rel}")

    if not MANIFEST_PATH.is_file():
        errors.append(f"manifest missing — run without --check first: {MANIFEST_PATH}")
        return errors

    recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    by_path = {row["path"]: row for row in recorded.get("files", [])}
    for rel in _REQUIRED:
        path = ROOT / rel
        if not path.is_file():
            continue
        row = by_path.get(rel)
        if row is None:
            errors.append(f"required file not listed in MANIFEST: {rel}")
            continue
        digest = _sha256(path)
        if digest != row.get("sha256"):
            errors.append(
                f"hash mismatch for {rel}: disk={digest[:12]}… "
                f"manifest={str(row.get('sha256'))[:12]}… "
                "(re-run scripts/build_reference_artefact_manifest.py)"
            )
        if int(row.get("bytes") or -1) != path.stat().st_size:
            errors.append(f"size mismatch for {rel}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify required files + MANIFEST hashes (no write)",
    )
    args = parser.parse_args(argv)

    if args.check:
        errors = check_manifest()
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            return 1
        print(f"OK: {len(_REQUIRED)} required artefacts present; MANIFEST hashes match.")
        return 0

    payload = build_manifest()
    path = write_manifest(payload)
    print(f"Wrote {path} ({payload['n_files']} files, {payload['total_bytes']} bytes)")
    # Always validate required set after build
    errors = check_manifest()
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
