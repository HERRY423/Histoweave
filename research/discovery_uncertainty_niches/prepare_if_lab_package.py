"""Build a wet-lab IF package for the top 2–3 cryptic niches.

Primary targets (same-section dual class on 151508):
  1. L3 niche n=138
  2. L6 niche n=154
Optional third (cross-donor L3 direction):
  3. 151669 L3 n=137  OR  priority cohort L3 from IF_priority list

Outputs under ``results/if_lab_package/``:
  - master_manifest.json
  - per-niche ROI + background controls CSVs
  - spatial multipanel RNA-proxy maps (PNG/SVG) for pathologist briefing
  - QuPath-style geojson (spot circles) if possible
  - LAB_BRIEF.md  (Chinese+English hand-off for core facility)
  - claim_ladder.md

This does **not** invent protein IF data. It prepares everything needed for
real IF and runs a clearly labelled **RNA multiplex spatial pre-validation**
so the experimental design is already stress-tested on the same barcodes.
"""

from __future__ import annotations

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
from histoweave.datasets import get_dataset  # noqa: E402

logger = logging.getLogger("if_lab_package")
BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "if_lab_package"

# Top 2–3 niches for real IF (frozen design).
NICHES: list[dict[str, Any]] = [
    {
        "id": "151508_L3",
        "slice_id": "dlpfc_151508",
        "roi_csv": BASE / "results" / "panel_validation" / "ROI_151508_L3_n138.csv",
        "class": "L3_program",
        "expected_layer": "Layer 3",
        "priority": 1,
        "hypothesis": (
            "ENC1 and/or HOPX protein higher in ROI than same-layer L3 non-ROI; "
            "MBP not higher in ROI (padj≤0.05)."
        ),
    },
    {
        "id": "151508_L6",
        "slice_id": "dlpfc_151508",
        "roi_csv": BASE / "results" / "panel_validation" / "ROI_151508_L6_n154.csv",
        "class": "L6_myelin",
        "expected_layer": "Layer 6",
        "priority": 1,
        "hypothesis": (
            "MBP protein higher in ROI than non-ROI rest of section (padj≤0.05); "
            "same-layer L6 contrast secondary."
        ),
    },
    {
        "id": "151669_L3",
        "slice_id": "dlpfc_151669",
        "roi_csv": BASE / "results" / "panel_validation" / "ROI_151669_L3_n137.csv",
        "class": "L3_program",
        "expected_layer": "Layer 3",
        "priority": 2,
        "hypothesis": "Cross-donor replication of L3 protein criteria.",
    },
]

ANTIBODIES = [
    {
        "target": "ENC1",
        "role": "L3 primary",
        "suggested_host": "rabbit",
        "dilution_note": "titrate",
    },
    {"target": "HOPX", "role": "L3 primary", "suggested_host": "mouse", "dilution_note": "titrate"},
    {
        "target": "MBP",
        "role": "myelin primary",
        "suggested_host": "rat/chicken",
        "dilution_note": "titrate",
    },
    {
        "target": "DAPI",
        "role": "nuclear counterstain",
        "suggested_host": "—",
        "dilution_note": "standard",
    },
]


def _setup() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )


def _to_dense(m) -> np.ndarray:
    if hasattr(m, "toarray"):
        return np.asarray(m.toarray(), dtype=float)
    return np.asarray(m, dtype=float)


def log1p_norm(X: np.ndarray) -> np.ndarray:
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * 1e4)


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


def load_slice_frame(slice_id: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return obs-level frame with coords, layer, barcode + expression matrix genes."""
    data = get_dataset(slice_id).load(cache_dir=ROOT / "datasets_cache")
    if "domain_truth" in data.obs.columns:
        keep = data.obs["domain_truth"].notna().to_numpy()
        lab = data.obs["domain_truth"].astype(str).to_numpy()
        keep = keep & ~np.isin(lab, ["NA", "nan", "None", ""])
        data = data.subset_obs(keep)
    coords = np.asarray(data.spatial, dtype=float)
    X = log1p_norm(_to_dense(data.X))
    genes = list(map(str, data.var_names))
    frame = pd.DataFrame(
        {
            "barcode": list(map(str, data.obs_names)),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "domain_truth": data.obs["domain_truth"].astype(str).to_numpy(),
        }
    )
    for g in ("ENC1", "HOPX", "MBP", "PLP1", "GAP43"):
        if g in genes:
            frame[f"rna_{g}"] = X[:, genes.index(g)]
    return frame, X, genes


def attach_roi_flags(frame: pd.DataFrame, roi: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    roi_bc = set(roi["barcode"].astype(str))
    out["in_roi"] = out["barcode"].astype(str).isin(roi_bc)
    return out


def rna_proxy_tests(frame: pd.DataFrame, niche: dict[str, Any]) -> pd.DataFrame:
    """Pre-registered RNA contrasts mirroring the IF design (proxy only)."""
    layer = niche["expected_layer"]
    in_roi = frame["in_roi"].to_numpy()
    same_layer = frame["domain_truth"].to_numpy() == layer
    rest = ~in_roi
    same_layer_out = same_layer & rest
    rows = []
    for g, col in [("ENC1", "rna_ENC1"), ("HOPX", "rna_HOPX"), ("MBP", "rna_MBP")]:
        if col not in frame.columns:
            continue
        vals = frame[col].to_numpy(dtype=float)
        for contrast, mask_out in (
            ("vs_same_layer", same_layer_out),
            ("vs_rest", rest),
        ):
            if mask_out.sum() < 5 or in_roi.sum() < 5:
                continue
            mean_in = float(vals[in_roi].mean())
            mean_out = float(vals[mask_out].mean())
            p = rank_sum_p(vals[in_roi], vals[mask_out])
            rows.append(
                {
                    "niche_id": niche["id"],
                    "class": niche["class"],
                    "gene": g,
                    "contrast": contrast,
                    "mean_in_roi": mean_in,
                    "mean_out": mean_out,
                    "delta": mean_in - mean_out,
                    "p_raw": p,
                    "n_in": int(in_roi.sum()),
                    "n_out": int(mask_out.sum()),
                    "modality": "RNA_proxy_not_protein",
                }
            )
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab["padj"] = fdr_adjust(tab["p_raw"].to_numpy(), method="bh")
    return tab


def evaluate_rna_proxy_gates(niche: dict[str, Any], tests: pd.DataFrame) -> dict[str, Any]:
    """Mirror IF pass criteria on RNA (labelled as proxy)."""
    if tests.empty:
        return {"pass_proxy": False, "reason": "no_tests"}
    cls = niche["class"]
    if cls == "L3_program":
        sub = tests[tests["contrast"] == "vs_same_layer"]
        enc = sub[sub["gene"] == "ENC1"]
        hop = sub[sub["gene"] == "HOPX"]
        mbp = sub[sub["gene"] == "MBP"]
        enc_ok = (not enc.empty) and bool(
            (enc["delta"].iloc[0] > 0) and (enc["padj"].iloc[0] <= 0.05)
        )
        hop_ok = (not hop.empty) and bool(
            (hop["delta"].iloc[0] > 0) and (hop["padj"].iloc[0] <= 0.05)
        )
        mbp_not_up = (mbp.empty) or bool(
            not ((mbp["delta"].iloc[0] > 0) and (mbp["padj"].iloc[0] <= 0.05))
        )
        return {
            "pass_proxy": bool((enc_ok or hop_ok) and mbp_not_up),
            "enc1_same_layer_up_fdr": enc_ok,
            "hopx_same_layer_up_fdr": hop_ok,
            "mbp_not_up_same_layer": mbp_not_up,
            "level": "RNA_proxy",
        }
    # L6
    sub = tests[tests["contrast"] == "vs_rest"]
    mbp = sub[sub["gene"] == "MBP"]
    mbp_ok = (not mbp.empty) and bool((mbp["delta"].iloc[0] > 0) and (mbp["padj"].iloc[0] <= 0.05))
    return {
        "pass_proxy": mbp_ok,
        "mbp_rest_up_fdr": mbp_ok,
        "level": "RNA_proxy",
    }


def write_geojson_spots(frame: pd.DataFrame, path: Path, radius_px: float = 50.0) -> None:
    """Minimal GeoJSON of ROI spot centroids as circles approximated by points."""
    feats = []
    for _, row in frame[frame["in_roi"]].iterrows():
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "barcode": row["barcode"],
                    "domain_truth": row["domain_truth"],
                    "radius_px": radius_px,
                },
                "geometry": {"type": "Point", "coordinates": [float(row["x"]), float(row["y"])]},
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}, indent=2),
        encoding="utf-8",
    )


def make_figures(
    niche_frames: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One figure per niche: tissue coords + ENC1, HOPX, MBP + ROI outline
    for nid, frame in niche_frames.items():
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        x, y = frame["x"].to_numpy(), -frame["y"].to_numpy()
        roi = frame["in_roi"].to_numpy()
        for ax, col, title in zip(
            axes,
            ["domain_truth", "rna_ENC1", "rna_HOPX", "rna_MBP"],
            ["Manual layer", "RNA ENC1", "RNA HOPX", "RNA MBP"],
            strict=True,
        ):
            if col == "domain_truth":
                cats = {v: i for i, v in enumerate(sorted(frame["domain_truth"].unique()))}
                c = frame["domain_truth"].map(cats).to_numpy()
                sc = ax.scatter(x, y, c=c, s=3, cmap="tab10", linewidths=0)
            else:
                vals = frame[col].to_numpy(dtype=float) if col in frame else np.zeros(len(frame))
                vmax = float(np.nanpercentile(vals, 99)) or 1.0
                sc = ax.scatter(x, y, c=vals, s=3, cmap="magma", linewidths=0, vmin=0, vmax=vmax)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            ax.scatter(x[roi], y[roi], s=14, facecolors="none", edgecolors="lime", linewidths=0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=10)
        fig.suptitle(f"{nid} — ROI (lime) · RNA multiplex pre-IF briefing", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"briefing_{nid}.png", dpi=160, bbox_inches="tight")
        fig.savefig(out_dir / f"briefing_{nid}.svg", bbox_inches="tight")
        plt.close(fig)

    # Combined 151508 L3 vs L6 comparison strip
    if "151508_L3" in niche_frames and "151508_L6" in niche_frames:
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        for row, nid in enumerate(["151508_L3", "151508_L6"]):
            frame = niche_frames[nid]
            x, y = frame["x"].to_numpy(), -frame["y"].to_numpy()
            roi = frame["in_roi"].to_numpy()
            for col_i, (col, title) in enumerate(
                [("rna_ENC1", "ENC1"), ("rna_HOPX", "HOPX"), ("rna_MBP", "MBP")]
            ):
                ax = axes[row, col_i]
                vals = frame[col].to_numpy(dtype=float)
                vmax = float(np.nanpercentile(vals, 99)) or 1.0
                sc = ax.scatter(x, y, c=vals, s=3, cmap="magma", linewidths=0, vmin=0, vmax=vmax)
                ax.scatter(
                    x[roi], y[roi], s=12, facecolors="none", edgecolors="lime", linewidths=0.45
                )
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{nid} · {title}", fontsize=9)
                fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(
            "Same-section dual niches on 151508 (do NOT pool L3+L6 as one state)",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "briefing_151508_L3_vs_L6.png", dpi=160, bbox_inches="tight")
        fig.savefig(out_dir / "briefing_151508_L3_vs_L6.svg", bbox_inches="tight")
        plt.close(fig)


def write_lab_brief(niches_meta: list[dict[str, Any]], proxy_gates: dict[str, Any]) -> str:
    lines = [
        "# IF Lab Brief / 实验交接说明",
        "",
        "## English",
        "",
        "### Goal",
        "Protein-validate **two distinct** cryptic niches on DLPFC Visium section "
        "**151508** (same tissue block ideal), plus optional cross-donor L3 on **151669**.",
        "",
        "| Priority | Niche | Spots | Class | Do not mix with |",
        "|---------:|-------|------:|-------|-----------------|",
    ]
    for n in niches_meta:
        lines.append(
            f"| {n['priority']} | `{n['id']}` | {n['n_roi']} | {n['class']} | "
            f"{'other class on same section' if n['slice_id'] == 'dlpfc_151508' else '—'} |"
        )
    lines += [
        "",
        "### Antibodies (minimal)",
        "",
        "| Target | Role |",
        "|--------|------|",
    ]
    for ab in ANTIBODIES:
        lines.append(f"| **{ab['target']}** | {ab['role']} |")
    lines += [
        "",
        "### Pass criteria (pre-registered — protein)",
        "",
        "- **L3:** ENC1 **or** HOPX higher in ROI vs **same-layer L3 non-ROI** (padj≤0.05); "
        "MBP **not** significantly higher in ROI.",
        "- **L6:** MBP higher in ROI vs **rest of section** (padj≤0.05).",
        "- **Cross-donor (optional):** 151669 L3 meets L3 criteria.",
        "",
        "### Files for the core",
        "",
        "- `niches/<id>/roi_barcodes.csv` — Visium barcodes in ROI",
        "- `niches/<id>/background_same_layer.csv` — same-layer non-ROI controls",
        "- `niches/<id>/background_rest.csv` — all non-ROI",
        "- `niches/<id>/roi.geojson` — spot centroids for overlay",
        "- `briefing_*.png` — RNA spatial maps with ROI outline (pathologist briefing)",
        "",
        "### Return format",
        "CSV with columns: `barcode, ENC1, HOPX, MBP [, PLP1]` (background-subtracted mean intensity).",
        "Drop under `results/if_return/` and run:",
        "",
        "```bash",
        "python research/discovery_uncertainty_niches/analyze_if_return.py",
        "```",
        "",
        "That command alone upgrades the claim ladder when protein gates pass.",
        "",
        "## 中文",
        "",
        "### 目标",
        "在 **151508** 同一张（或同供体）切片上验证 **两个不同** 生态位："
        "**L3 型 n=138** 与 **L6 型 n=154**；可选第三位点 **151669 L3** 做跨供体。",
        "",
        "**禁止** 把 L3 ROI 与 L6 ROI 合并成同一种“cryptic 状态”。",
        "",
        "### 抗体",
        "ENC1、HOPX、MBP（+DAPI）；可选 PLP1。",
        "",
        "### 通过标准（蛋白，预注册）",
        "- L3：相对 **同层 L3 非 ROI**，ENC1 或 HOPX 升高（padj≤0.05），且 MBP 不升高。",
        "- L6：相对 **全切片非 ROI**，MBP 升高（padj≤0.05）。",
        "",
        "### 回传",
        "按 barcode 的 IF 强度表 → `results/if_return/` → 运行 `analyze_if_return.py` "
        "自动生成「经验证生物学」报告。",
        "",
        "## RNA proxy pre-check (not protein)",
        "",
    ]
    for nid, g in proxy_gates.items():
        lines.append(f"- `{nid}`: proxy_pass={g.get('pass_proxy')} · {g}")
    lines += [
        "",
        "_RNA proxy can fail while protein still passes (or vice versa). "
        "It only stress-tests ROI/control design._",
        "",
    ]
    return "\n".join(lines)


def write_claim_ladder(proxy_gates: dict[str, Any], protein_status: str = "PENDING") -> str:
    l3_proxy = all(
        proxy_gates.get(k, {}).get("pass_proxy")
        for k in ("151508_L3", "151669_L3")
        if k in proxy_gates
    )
    l6_proxy = proxy_gates.get("151508_L6", {}).get("pass_proxy", False)
    lines = [
        "# Claim ladder — geometric candidate → validated biology",
        "",
        "| Level | Name | Status | Evidence |",
        "|------:|------|--------|----------|",
        "| 0 | Geometric candidate | **DONE** | Multi-method uncertainty + contiguous cryptic components |",
        "| 1 | RNA direction (vs rest) | **DONE** | L3 14/15 cohort; donor-stratified CI excludes 0 |",
        "| 2 | RNA same-layer hard | **FAIL / weak** | Shift-null same-layer rarely ≤0.05 |",
        f"| 2b | RNA multiplex proxy on IF ROIs | "
        f"**L3_proxy={'PASS' if l3_proxy else 'MIXED/FAIL'} · "
        f"L6_proxy={'PASS' if l6_proxy else 'FAIL'}** | "
        f"Same barcodes/contrasts as IF design |",
        f"| 3 | **Protein IF same-layer** | **{protein_status}** | "
        f"Requires wet-lab return → `analyze_if_return.py` |",
        "| 4 | Multi-donor protein | PENDING | 151508 + 151669 L3 both IF-pass |",
        "",
        "## Narrative rules",
        "",
        "| If… | Allowed language |",
        "|-----|------------------|",
        "| Level ≤1 only | geometric / RNA-directional candidate |",
        "| Level 2b pass, IF pending | molecularly corroborated candidate; **not** validated protein biology |",
        "| Level 3 pass on 151508 L3 **and** L6 | **validated dual-niche biology on one section** |",
        "| Level 4 pass | cross-donor validated L3 niche program |",
        "",
        "**Never** claim a single unified “cryptic cell state” for L3+L6.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)
    niches_dir = OUT / "niches"
    niches_dir.mkdir(exist_ok=True)

    niche_frames: dict[str, pd.DataFrame] = {}
    niches_meta: list[dict[str, Any]] = []
    proxy_gates: dict[str, Any] = {}
    all_tests: list[pd.DataFrame] = []

    for niche in NICHES:
        if not niche["roi_csv"].is_file():
            logger.warning("missing ROI %s — skip", niche["roi_csv"])
            continue
        roi = pd.read_csv(niche["roi_csv"])
        frame, _X, _genes = load_slice_frame(niche["slice_id"])
        frame = attach_roi_flags(frame, roi)
        niche_frames[niche["id"]] = frame

        nd = niches_dir / niche["id"]
        nd.mkdir(exist_ok=True)
        roi_out = frame[frame["in_roi"]].copy()
        layer = niche["expected_layer"]
        bg_same = frame[(~frame["in_roi"]) & (frame["domain_truth"] == layer)].copy()
        bg_rest = frame[~frame["in_roi"]].copy()
        roi_out.to_csv(nd / "roi_barcodes.csv", index=False)
        bg_same.to_csv(nd / "background_same_layer.csv", index=False)
        bg_rest.to_csv(nd / "background_rest.csv", index=False)
        # compact barcode-only lists for facilities
        roi_out[["barcode", "x", "y", "domain_truth"]].to_csv(
            nd / "roi_barcodes_minimal.csv", index=False
        )
        write_geojson_spots(frame, nd / "roi.geojson")

        tests = rna_proxy_tests(frame, niche)
        tests.to_csv(nd / "rna_proxy_contrasts.csv", index=False)
        all_tests.append(tests)
        gates = evaluate_rna_proxy_gates(niche, tests)
        proxy_gates[niche["id"]] = gates
        (nd / "rna_proxy_gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

        meta = {
            **{k: v for k, v in niche.items() if k != "roi_csv"},
            "n_roi": int(frame["in_roi"].sum()),
            "n_same_layer_bg": int(bg_same.shape[0]),
            "n_rest_bg": int(bg_rest.shape[0]),
            "rna_proxy_gates": gates,
        }
        niches_meta.append(meta)
        logger.info(
            "%s n_roi=%s proxy_pass=%s",
            niche["id"],
            meta["n_roi"],
            gates.get("pass_proxy"),
        )

    if all_tests:
        pd.concat(all_tests, ignore_index=True).to_csv(
            OUT / "rna_proxy_all_contrasts.csv", index=False
        )

    make_figures(niche_frames, OUT)

    manifest = {
        "protocol": "histoweave.if_lab_package.v1",
        "primary_section": "151508",
        "niches": niches_meta,
        "antibodies": ANTIBODIES,
        "protein_status": "PENDING_WET_LAB",
        "return_dir": "research/discovery_uncertainty_niches/results/if_return",
        "analyzer": "research/discovery_uncertainty_niches/analyze_if_return.py",
    }
    (OUT / "master_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUT / "LAB_BRIEF.md").write_text(write_lab_brief(niches_meta, proxy_gates), encoding="utf-8")
    (OUT / "claim_ladder.md").write_text(write_claim_ladder(proxy_gates), encoding="utf-8")
    (BASE / "IF_LAB_BRIEF.md").write_text(
        (OUT / "LAB_BRIEF.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (BASE / "CLAIM_LADDER.md").write_text(
        (OUT / "claim_ladder.md").read_text(encoding="utf-8"), encoding="utf-8"
    )

    # Template empty IF return for core facility
    tmpl = OUT / "IF_RETURN_TEMPLATE.csv"
    tmpl.write_text(
        "barcode,section_id,niche_id,ENC1,HOPX,MBP,PLP1,notes\n"
        "# one row per Visium barcode quantified; leave PLP1 empty if unused\n",
        encoding="utf-8",
    )
    logger.info("IF lab package written to %s", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
