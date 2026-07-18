"""Analyse wet-lab IF return tables and upgrade the claim ladder.

Expected inputs under ``results/if_return/``:

* ``IF_*.csv`` or any ``*.csv`` with columns:
  ``barcode, section_id, niche_id, ENC1, HOPX, MBP`` (PLP1 optional)

* Or a single ``if_intensities.csv`` covering all niches.

Niche ids must match the lab package:
  ``151508_L3``, ``151508_L6``, ``151669_L3``

When protein gates pass, writes:

* ``results/if_return/VALIDATED_BIOLOGY_REPORT.md``  ← narrative upgrade
* ``results/if_return/protein_gate_results.json``
* updates ``CLAIM_LADDER.md`` protein level to PASS/FAIL

Dry-run (RNA proxy as fake IF channels — **not** protein validation)::

    python analyze_if_return.py --simulate-from-rna
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.multiple_testing import fdr_adjust  # noqa: E402

logger = logging.getLogger("analyze_if_return")
BASE = Path(__file__).resolve().parent
LAB = BASE / "results" / "if_lab_package"
RETURN = BASE / "results" / "if_return"

NICHES = {
    "151508_L3": {
        "class": "L3_program",
        "expected_layer": "Layer 3",
        "slice_id": "dlpfc_151508",
        "roi_dir": LAB / "niches" / "151508_L3",
    },
    "151508_L6": {
        "class": "L6_myelin",
        "expected_layer": "Layer 6",
        "slice_id": "dlpfc_151508",
        "roi_dir": LAB / "niches" / "151508_L6",
    },
    "151669_L3": {
        "class": "L3_program",
        "expected_layer": "Layer 3",
        "slice_id": "dlpfc_151669",
        "roi_dir": LAB / "niches" / "151669_L3",
    },
}


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def rank_sum_p(x: np.ndarray, y: np.ndarray) -> float:
    from math import erfc, sqrt

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return 1.0
    combined = np.concatenate([x, y])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=float)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[order[j + 1]] == combined[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg
        i = j + 1
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    _, counts = np.unique(combined, return_counts=True)
    tie = (counts**3 - counts).sum() / (len(combined) * (len(combined) - 1))
    sigma2 = n1 * n2 / 12.0 * ((n1 + n2 + 1) - tie)
    sigma = float(np.sqrt(max(sigma2, 1e-12)))
    z = (u1 - mu - 0.5 * np.sign(u1 - mu)) / sigma
    return float(erfc(abs(z) / sqrt(2.0)))


def load_if_tables() -> pd.DataFrame:
    RETURN.mkdir(parents=True, exist_ok=True)
    files = sorted(RETURN.glob("*.csv"))
    files = [f for f in files if f.name.upper().startswith("IF") or f.name == "if_intensities.csv"]
    if not files:
        # accept any csv that has ENC1 column
        files = [
            f
            for f in sorted(RETURN.glob("*.csv"))
            if "barcode" in f.read_text(encoding="utf-8", errors="ignore")[:500]
        ]
    if not files:
        raise FileNotFoundError(
            f"No IF return CSVs in {RETURN}. "
            "Place tables with columns barcode,niche_id,ENC1,HOPX,MBP"
        )
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        # skip comment-only
        if "barcode" not in df.columns:
            continue
        df["source_file"] = f.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError("IF CSVs found but none contain a barcode column")
    out = pd.concat(frames, ignore_index=True)
    # normalise column names
    rename = {
        c: c.upper() if c.lower() in {"enc1", "hopx", "mbp", "plp1"} else c for c in out.columns
    }
    out = out.rename(columns=rename)
    if "niche_id" not in out.columns and "NICHE_ID" in out.columns:
        out["niche_id"] = out["NICHE_ID"]
    return out


def simulate_if_from_rna() -> pd.DataFrame:
    """Build a fake IF table from lab-package RNA columns for pipeline dry-run."""
    rows = []
    for nid, meta in NICHES.items():
        roi_path = meta["roi_dir"] / "roi_barcodes.csv"
        same_path = meta["roi_dir"] / "background_same_layer.csv"
        rest_path = meta["roi_dir"] / "background_rest.csv"
        if not roi_path.is_file():
            logger.warning("simulate: missing %s", roi_path)
            continue
        # ROI
        for _, r in pd.read_csv(roi_path).iterrows():
            rows.append(
                {
                    "barcode": r["barcode"],
                    "section_id": meta["slice_id"].replace("dlpfc_", ""),
                    "niche_id": nid,
                    "in_roi": True,
                    "control_set": "roi",
                    "domain_truth": r.get("domain_truth", meta["expected_layer"]),
                    "ENC1": float(r.get("rna_ENC1", np.nan)),
                    "HOPX": float(r.get("rna_HOPX", np.nan)),
                    "MBP": float(r.get("rna_MBP", np.nan)),
                    "modality": "SIMULATED_FROM_RNA_NOT_PROTEIN",
                }
            )
        # Same-layer background (primary control for L3)
        if same_path.is_file():
            df_same = pd.read_csv(same_path)
            if len(df_same) > 2000:
                df_same = df_same.sample(2000, random_state=0)
            for _, r in df_same.iterrows():
                rows.append(
                    {
                        "barcode": r["barcode"],
                        "section_id": meta["slice_id"].replace("dlpfc_", ""),
                        "niche_id": nid,
                        "in_roi": False,
                        "control_set": "same_layer",
                        "domain_truth": r.get("domain_truth", meta["expected_layer"]),
                        "ENC1": float(r.get("rna_ENC1", np.nan)),
                        "HOPX": float(r.get("rna_HOPX", np.nan)),
                        "MBP": float(r.get("rna_MBP", np.nan)),
                        "modality": "SIMULATED_FROM_RNA_NOT_PROTEIN",
                    }
                )
        # Rest-of-section background (primary control for L6)
        if rest_path.is_file():
            df_rest = pd.read_csv(rest_path)
            if len(df_rest) > 2000:
                df_rest = df_rest.sample(2000, random_state=0)
            for _, r in df_rest.iterrows():
                rows.append(
                    {
                        "barcode": r["barcode"],
                        "section_id": meta["slice_id"].replace("dlpfc_", ""),
                        "niche_id": nid,
                        "in_roi": False,
                        "control_set": "rest",
                        "domain_truth": r.get("domain_truth", ""),
                        "ENC1": float(r.get("rna_ENC1", np.nan)),
                        "HOPX": float(r.get("rna_HOPX", np.nan)),
                        "MBP": float(r.get("rna_MBP", np.nan)),
                        "modality": "SIMULATED_FROM_RNA_NOT_PROTEIN",
                    }
                )
    return pd.DataFrame(rows)


def run_contrasts(if_df: pd.DataFrame, niche_id: str, meta: dict[str, Any]) -> pd.DataFrame:
    sub = if_df[if_df["niche_id"].astype(str) == niche_id].copy()
    if sub.empty:
        # try match without requiring niche_id on all rows — join on barcode lists
        roi = pd.read_csv(meta["roi_dir"] / "roi_barcodes_minimal.csv")
        bg_same = pd.read_csv(meta["roi_dir"] / "background_same_layer.csv")
        bg_rest = pd.read_csv(meta["roi_dir"] / "background_rest.csv")
        if_df = if_df.copy()
        if_df["barcode"] = if_df["barcode"].astype(str)
        roi_bc = set(roi["barcode"].astype(str))
        same_bc = set(bg_same["barcode"].astype(str))
        rest_bc = set(bg_rest["barcode"].astype(str))
        rows = []
        for _, r in if_df.iterrows():
            bc = str(r["barcode"])
            if bc in roi_bc:
                rows.append(
                    {**r.to_dict(), "niche_id": niche_id, "in_roi": True, "control_set": "roi"}
                )
            elif bc in same_bc:
                rows.append(
                    {
                        **r.to_dict(),
                        "niche_id": niche_id,
                        "in_roi": False,
                        "control_set": "same_layer",
                    }
                )
            elif bc in rest_bc:
                rows.append(
                    {**r.to_dict(), "niche_id": niche_id, "in_roi": False, "control_set": "rest"}
                )
        sub = pd.DataFrame(rows)

    if "in_roi" not in sub.columns:
        raise ValueError(f"{niche_id}: IF table needs in_roi or joinable barcodes")

    in_mask = sub["in_roi"].astype(bool).to_numpy()
    # controls
    if "control_set" in sub.columns:
        same_out = (~in_mask) & (sub["control_set"].to_numpy() == "same_layer")
        rest_out = (~in_mask) & (
            (sub["control_set"].to_numpy() == "rest")
            | (sub["control_set"].to_numpy() == "same_layer")
        )
    else:
        layer = meta["expected_layer"]
        if "domain_truth" in sub.columns:
            same_out = (~in_mask) & (sub["domain_truth"].astype(str).to_numpy() == layer)
        else:
            same_out = ~in_mask
        rest_out = ~in_mask

    rows = []
    for gene in ("ENC1", "HOPX", "MBP"):
        if gene not in sub.columns:
            continue
        vals = pd.to_numeric(sub[gene], errors="coerce").to_numpy(dtype=float)
        for contrast, mask_out in (("vs_same_layer", same_out), ("vs_rest", rest_out)):
            if mask_out.sum() < 3 or in_mask.sum() < 3:
                continue
            mean_in = float(np.nanmean(vals[in_mask]))
            mean_out = float(np.nanmean(vals[mask_out]))
            p = rank_sum_p(vals[in_mask], vals[mask_out])
            rows.append(
                {
                    "niche_id": niche_id,
                    "class": meta["class"],
                    "gene": gene,
                    "contrast": contrast,
                    "mean_in_roi": mean_in,
                    "mean_out": mean_out,
                    "delta": mean_in - mean_out,
                    "p_raw": p,
                    "n_in": int(in_mask.sum()),
                    "n_out": int(mask_out.sum()),
                }
            )
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab["padj"] = fdr_adjust(tab["p_raw"].to_numpy(), method="bh")
    return tab


def protein_gates(niche_id: str, meta: dict[str, Any], tests: pd.DataFrame) -> dict[str, Any]:
    if tests.empty:
        return {"pass": False, "reason": "no_tests", "niche_id": niche_id}
    cls = meta["class"]
    if cls == "L3_program":
        sub = tests[tests["contrast"] == "vs_same_layer"]

        def ok(gene: str) -> bool:
            g = sub[sub["gene"] == gene]
            return (not g.empty) and bool(g["delta"].iloc[0] > 0 and g["padj"].iloc[0] <= 0.05)

        enc, hop = ok("ENC1"), ok("HOPX")
        mbp = sub[sub["gene"] == "MBP"]
        mbp_not_up = mbp.empty or not bool(mbp["delta"].iloc[0] > 0 and mbp["padj"].iloc[0] <= 0.05)
        passed = bool((enc or hop) and mbp_not_up)
        return {
            "niche_id": niche_id,
            "class": cls,
            "pass": passed,
            "enc1_up": enc,
            "hopx_up": hop,
            "mbp_not_up": mbp_not_up,
            "level": 3,
        }
    sub = tests[tests["contrast"] == "vs_rest"]
    mbp = sub[sub["gene"] == "MBP"]
    mbp_ok = (not mbp.empty) and bool(mbp["delta"].iloc[0] > 0 and mbp["padj"].iloc[0] <= 0.05)
    return {
        "niche_id": niche_id,
        "class": cls,
        "pass": mbp_ok,
        "mbp_up_vs_rest": mbp_ok,
        "level": 3,
    }


def write_validated_report(
    gates: dict[str, dict[str, Any]],
    all_tests: pd.DataFrame,
    *,
    simulated: bool,
) -> str:
    l3_151508 = gates.get("151508_L3", {}).get("pass", False)
    l6_151508 = gates.get("151508_L6", {}).get("pass", False)
    l3_669 = gates.get("151669_L3", {}).get("pass", False)
    dual = l3_151508 and l6_151508
    cross = l3_151508 and l3_669

    if simulated:
        banner = (
            "> ⚠️ **SIMULATED FROM RNA — NOT PROTEIN IF.** "
            "This dry-run only proves the analysis path. "
            "Do not cite as validated biology.\n"
        )
        status = "SIMULATED_PROXY"
    else:
        banner = ""
        status = "PROTEIN_IF"

    lines = [
        "# Validated biology report" + (" (RNA simulation dry-run)" if simulated else ""),
        "",
        banner,
        f"**Status tag:** `{status}`",
        "",
        "## Gate results",
        "",
        "| Niche | Class | Protein/Proxy pass | Detail |",
        "|-------|-------|:------------------:|--------|",
    ]
    for nid, g in gates.items():
        lines.append(
            f"| `{nid}` | {g.get('class')} | **{'PASS' if g.get('pass') else 'FAIL'}** | `{g}` |"
        )

    lines += ["", "## Narrative upgrade", ""]
    if simulated:
        lines.append(
            "Simulation complete. Replace with real IF CSVs in `results/if_return/` "
            "and re-run **without** `--simulate-from-rna`."
        )
    elif dual and cross:
        lines.append(
            "### Claim language now allowed\n\n"
            "On section **151508**, multi-method cryptic niches resolve into **two "
            "protein-validated programs**: an L3-associated ENC1/HOPX niche and an "
            "L6-associated MBP niche. The L3 program **replicates on donor 151669** "
            "by the same pre-registered criteria. These are **not** a single cell state."
        )
    elif dual:
        lines.append(
            "### Claim language now allowed (single-section dual niche)\n\n"
            "On **151508**, IF validates **two distinct** niches (L3 program vs L6 myelin) "
            "under pre-registered tests. Cross-donor L3 protein replication is "
            f"{'PASS' if cross else 'not yet demonstrated'}."
        )
    elif l3_151508 or l6_151508:
        which = []
        if l3_151508:
            which.append("L3")
        if l6_151508:
            which.append("L6")
        lines.append(
            f"### Partial validation\n\n"
            f"Protein gates pass for **{' and '.join(which)}** on 151508 only. "
            "Keep remaining niches as candidates; do not generalise."
        )
    else:
        lines.append(
            "### No narrative upgrade\n\n"
            "Protein gates did not pass. Retain **geometric / RNA-directional candidate** language."
        )

    lines += [
        "",
        "## Contrasts (padj)",
        "",
        "| Niche | Gene | Contrast | Δ | padj |",
        "|-------|------|----------|--:|-----:|",
    ]
    if not all_tests.empty:
        for _, r in all_tests.sort_values(["niche_id", "gene", "contrast"]).iterrows():
            lines.append(
                f"| {r['niche_id']} | {r['gene']} | {r['contrast']} | "
                f"{r['delta']:.4f} | {r['padj']:.2e} |"
            )
    lines += [
        "",
        "## Methods note",
        "",
        "Pre-registered in `IF_PROTOCOL.md` / `prepare_if_lab_package.py`. "
        "Mann–Whitney U + BH-FDR within the tested contrast set.",
        "",
    ]
    return "\n".join(lines)


def update_claim_ladder(gates: dict[str, dict[str, Any]], *, simulated: bool) -> None:
    ladder_path = BASE / "CLAIM_LADDER.md"
    if not ladder_path.is_file():
        return
    dual = gates.get("151508_L3", {}).get("pass") and gates.get("151508_L6", {}).get("pass")
    if simulated:
        status = "PENDING (sim only)"
    elif dual:
        status = "**PASS (protein IF)**"
    elif any(g.get("pass") for g in gates.values()):
        status = "PARTIAL"
    else:
        status = "FAIL / PENDING"
    text = ladder_path.read_text(encoding="utf-8")
    # rewrite level 3 line if present
    lines = []
    for line in text.splitlines():
        if line.startswith("| 3 |"):
            lines.append(
                f"| 3 | **Protein IF same-layer** | {status} | "
                f"analyze_if_return.py · see VALIDATED_BIOLOGY_REPORT.md |"
            )
        else:
            lines.append(line)
    ladder_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _setup()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulate-from-rna",
        action="store_true",
        help="Dry-run using RNA intensities as fake IF (NOT protein validation).",
    )
    args = parser.parse_args(argv)
    RETURN.mkdir(parents=True, exist_ok=True)

    try:
        if args.simulate_from_rna:
            if not (LAB / "niches").is_dir():
                logger.error("Run prepare_if_lab_package.py first")
                return 2
            if_df = simulate_if_from_rna()
            if_df.to_csv(RETURN / "IF_SIMULATED_FROM_RNA.csv", index=False)
            logger.warning("SIMULATION MODE — not protein IF")
        else:
            if_df = load_if_tables()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.info(
            "No wet-lab IF yet. Generated package is ready; use --simulate-from-rna "
            "only to test the analyzer, or drop real CSVs into %s",
            RETURN,
        )
        return 2

    all_tests = []
    gates: dict[str, dict[str, Any]] = {}
    for nid, meta in NICHES.items():
        if not meta["roi_dir"].is_dir():
            logger.warning("niche package missing for %s — skip", nid)
            continue
        try:
            tests = run_contrasts(if_df, nid, meta)
        except Exception as exc:
            logger.exception("%s contrast failed: %s", nid, exc)
            gates[nid] = {"pass": False, "reason": str(exc), "niche_id": nid}
            continue
        tests.to_csv(RETURN / f"contrasts_{nid}.csv", index=False)
        all_tests.append(tests)
        g = protein_gates(nid, meta, tests)
        gates[nid] = g
        logger.info("%s pass=%s %s", nid, g.get("pass"), g)

    tests_df = pd.concat(all_tests, ignore_index=True) if all_tests else pd.DataFrame()
    if not tests_df.empty:
        tests_df.to_csv(RETURN / "protein_contrasts_all.csv", index=False)
    (RETURN / "protein_gate_results.json").write_text(
        json.dumps({"simulated": bool(args.simulate_from_rna), "gates": gates}, indent=2),
        encoding="utf-8",
    )
    report = write_validated_report(gates, tests_df, simulated=bool(args.simulate_from_rna))
    out_report = RETURN / "VALIDATED_BIOLOGY_REPORT.md"
    out_report.write_text(report, encoding="utf-8")
    (BASE / "VALIDATED_BIOLOGY_REPORT.md").write_text(report, encoding="utf-8")
    update_claim_ladder(gates, simulated=bool(args.simulate_from_rna))
    logger.info("Wrote %s", out_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
