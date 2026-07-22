#!/usr/bin/env python
"""Non-oracle K dual-track SOTA benchmark on 5 DLPFC slices.

Compares SpaGCN and STAGATE under:

* ``k_policy=oracle`` (opt-in ablation baseline)
* ``k_policy=estimate`` with estimators:
  - ``silhouette`` (legacy expression-only)
  - ``spatial_silhouette`` (neighbourhood-smoothed)
  - ``ensemble`` (expression + spatial vote)

Records :class:`DualTrackKReport` per slice × estimator and measures ARI loss
relative to oracle-K and recovery by spatial-aware estimators.

Environment
-----------
``HISTOWEAVE_SOTA_DEVICE``  cpu|cuda (default cpu)
``HISTOWEAVE_STAGATE_EPOCHS`` default 200
``HISTOWEAVE_SPAGCN_MAX_EPOCHS`` default 200
``HISTOWEAVE_NON_ORACLE_SEEDS`` default ``42`` (comma-separated)
``HISTOWEAVE_NON_ORACLE_FORCE`` 1 to ignore checkpoints
``KMP_DUPLICATE_LIB_OK`` recommended on Windows
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histoweave.benchmark.k_selection import (  # noqa: E402
    DualTrackKReport,
    estimate_n_domains,
    oracle_n_domains,
)
from histoweave.benchmark.sota_pipeline import load_dlpfc_slice  # noqa: E402
from histoweave.plugins import MethodCategory, create_method  # noqa: E402

OUT = Path(__file__).resolve().parent
CKPT = OUT / "checkpoints"
FIG = OUT / "figures"
SLICES = ("151673", "151674", "151507", "151669", "151670")
METHODS = ("spagcn", "stagate")
# Order matters for figure narrative: oracle → drop → recovery.
K_MODES: tuple[tuple[str, str | None], ...] = (
    ("oracle", None),
    ("estimate", "silhouette"),
    ("estimate", "spatial_silhouette"),
    ("estimate", "ensemble"),
)
PROTOCOL = "histoweave.non_oracle_k_sota.v1"

logger = logging.getLogger("non_oracle_k_sota")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _seeds() -> list[int]:
    raw = os.environ.get("HISTOWEAVE_NON_ORACLE_SEEDS", "42")
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _force() -> bool:
    return os.environ.get("HISTOWEAVE_NON_ORACLE_FORCE", "0") == "1"


def _mode_key(k_policy: str, estimator: str | None) -> str:
    if k_policy == "oracle":
        return "oracle"
    return f"estimate:{estimator}"


def _ckpt_path(method: str, sid: str, seed: int, mode: str, k_used: int) -> Path:
    safe = mode.replace(":", "_")
    return CKPT / f"{method}__{sid}__seed{seed}__{safe}__k{k_used}.json"


def _resolve_k(
    table,
    *,
    k_policy: str,
    estimator: str | None,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    oracle = oracle_n_domains(table)
    if k_policy == "oracle":
        return oracle, {
            "k_policy": "oracle",
            "estimator": None,
            "k_used": oracle,
            "oracle_k": oracle,
            "k_match": True,
            "selection": None,
        }
    assert estimator is not None
    selection = estimate_n_domains(
        table,
        method=estimator,  # type: ignore[arg-type]
        random_state=seed,
        max_obs=int(os.environ.get("HISTOWEAVE_K_MAX_OBS", "2500")),
        k_max=int(os.environ.get("HISTOWEAVE_K_MAX", "12")),
    )
    dual = DualTrackKReport(
        dataset=str(table.uns.get("slice_id", "dataset")),
        oracle_k=oracle,
        estimated_k=selection.k,
        estimator=estimator,
        k_match=oracle == selection.k,
        selection=selection.to_dict(),
    )
    return int(selection.k), {
        "k_policy": "estimate",
        "estimator": estimator,
        "k_used": int(selection.k),
        "oracle_k": oracle,
        "k_match": dual.k_match,
        "selection": dual.to_dict(),
    }


def _run_method(
    method: str,
    table,
    *,
    n_domains: int,
    seed: int,
) -> np.ndarray:
    spagcn_epochs = int(os.environ.get("HISTOWEAVE_SPAGCN_MAX_EPOCHS", "200"))
    stagate_epochs = int(os.environ.get("HISTOWEAVE_STAGATE_EPOCHS", "200"))
    params: dict[str, Any] = {
        "n_domains": int(n_domains),
        "random_state": int(seed),
    }
    if method == "spagcn":
        params["max_epochs"] = spagcn_epochs
    elif method == "stagate":
        params["n_epochs"] = stagate_epochs
    out = create_method(MethodCategory.DOMAIN_DETECTION, method, **params).run(table.copy())
    return out.obs["domain"].astype(str).to_numpy()


def run_cell(
    method: str,
    sid: str,
    seed: int,
    k_policy: str,
    estimator: str | None,
) -> dict[str, Any]:
    from sklearn.metrics import adjusted_rand_score

    table, n_truth = load_dlpfc_slice(sid, repo_root=ROOT)
    # Prevent silent oracle leak via uns.
    table.uns = dict(table.uns)
    table.uns.pop("n_domains", None)
    table.uns["slice_id"] = sid

    k_used, k_meta = _resolve_k(table, k_policy=k_policy, estimator=estimator, seed=seed)
    mode = _mode_key(k_policy, estimator)
    ckpt = _ckpt_path(method, sid, seed, mode, k_used)

    if ckpt.is_file() and not _force():
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        if payload.get("status") == "success" and payload.get("ari") is not None:
            logger.info(
                "reuse %s %s %s k=%s ARI=%.4f",
                method,
                sid,
                mode,
                k_used,
                payload["ari"],
            )
            return payload

    t0 = time.perf_counter()
    row: dict[str, Any] = {
        "protocol": PROTOCOL,
        "dataset": sid,
        "method": method,
        "seed": seed,
        "k_policy": k_policy,
        "estimator": estimator,
        "mode": mode,
        "k_used": k_used,
        "oracle_k": k_meta["oracle_k"],
        "k_match": k_meta["k_match"],
        "n_domains_truth": n_truth,
        "n_obs": int(table.n_obs),
        "selection": k_meta.get("selection"),
    }
    try:
        pred = _run_method(method, table, n_domains=k_used, seed=seed)
        truth = table.obs["domain_truth"].astype(str).to_numpy()
        ari = float(adjusted_rand_score(truth, pred))
        elapsed = time.perf_counter() - t0
        row.update(
            {
                "ari": ari if np.isfinite(ari) else None,
                "seconds": round(elapsed, 3),
                "status": "success" if np.isfinite(ari) else "failed",
                "error": None,
            }
        )
        logger.info(
            "%s %s %s k=%s (oracle=%s) ARI=%.4f (%.1fs)",
            method,
            sid,
            mode,
            k_used,
            k_meta["oracle_k"],
            ari,
            elapsed,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        row.update(
            {
                "ari": None,
                "seconds": round(elapsed, 3),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
        )
        logger.exception("%s %s %s FAILED: %s", method, sid, mode, exc)

    CKPT.mkdir(parents=True, exist_ok=True)
    ckpt.write_text(json.dumps(row, allow_nan=False, default=str), encoding="utf-8")
    return row


def compute_dual_track_table(seeds: list[int]) -> list[dict[str, Any]]:
    """DualTrackKReport for every slice × non-oracle estimator (seed-stable)."""
    estimators = ("silhouette", "spatial_silhouette", "ensemble")
    rows: list[dict[str, Any]] = []
    seed = seeds[0]
    for sid in SLICES:
        table, _ = load_dlpfc_slice(sid, repo_root=ROOT)
        table.uns = dict(table.uns)
        table.uns.pop("n_domains", None)
        table.uns["dataset_name"] = sid
        table.uns["slice_id"] = sid
        for est in estimators:
            # compare_k_policies uses estimate_n_domains defaults; call directly
            # so we can cap max_obs on large Visium slides.
            selection = estimate_n_domains(
                table,
                method=est,  # type: ignore[arg-type]
                random_state=seed,
                max_obs=int(os.environ.get("HISTOWEAVE_K_MAX_OBS", "2500")),
                k_max=int(os.environ.get("HISTOWEAVE_K_MAX", "12")),
            )
            oracle = oracle_n_domains(table)
            report = DualTrackKReport(
                dataset=sid,
                oracle_k=oracle,
                estimated_k=selection.k,
                estimator=est,
                k_match=oracle == selection.k,
                selection=selection.to_dict(),
            )
            payload = report.to_dict()
            payload["protocol"] = PROTOCOL
            payload["seed_for_estimation"] = seed
            rows.append(payload)
            logger.info(
                "DualTrack %s %s: oracle_k=%s estimated_k=%s match=%s",
                sid,
                est,
                report.oracle_k,
                report.estimated_k,
                report.k_match,
            )
    return rows


def summarize(cells: list[dict[str, Any]], dual: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [c for c in cells if c.get("status") == "success" and c.get("ari") is not None]
    by_mode: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        for k_policy, estimator in K_MODES:
            mode = _mode_key(k_policy, estimator)
            rows = [c for c in ok if c["method"] == method and c["mode"] == mode]
            if not rows:
                continue
            aris = [float(c["ari"]) for c in rows]
            by_mode[f"{method}|{mode}"] = {
                "method": method,
                "mode": mode,
                "n": len(aris),
                "mean_ari": float(np.mean(aris)),
                "std_ari": float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0,
                "mean_k_used": float(np.mean([c["k_used"] for c in rows])),
                "k_match_rate": float(np.mean([bool(c["k_match"]) for c in rows])),
            }

    recovery: dict[str, Any] = {}
    for method in METHODS:
        oracle_key = f"{method}|oracle"
        sil_key = f"{method}|estimate:silhouette"
        spat_key = f"{method}|estimate:spatial_silhouette"
        ens_key = f"{method}|estimate:ensemble"
        if oracle_key not in by_mode or sil_key not in by_mode:
            continue
        oracle_ari = by_mode[oracle_key]["mean_ari"]
        sil_ari = by_mode[sil_key]["mean_ari"]
        drop = oracle_ari - sil_ari
        spat_ari = by_mode.get(spat_key, {}).get("mean_ari")
        ens_ari = by_mode.get(ens_key, {}).get("mean_ari")
        recovery[method] = {
            "oracle_mean_ari": oracle_ari,
            "silhouette_mean_ari": sil_ari,
            "spatial_silhouette_mean_ari": spat_ari,
            "ensemble_mean_ari": ens_ari,
            "ari_drop_silhouette_vs_oracle": drop,
            "ari_recovered_by_spatial_silhouette": (
                None if spat_ari is None else float(spat_ari - sil_ari)
            ),
            "ari_recovered_by_ensemble": (None if ens_ari is None else float(ens_ari - sil_ari)),
            "fraction_of_drop_recovered_ensemble": (
                None
                if ens_ari is None or drop <= 1e-12
                else float(max(0.0, ens_ari - sil_ari) / drop)
            ),
            "fraction_of_drop_recovered_spatial_silhouette": (
                None
                if spat_ari is None or drop <= 1e-12
                else float(max(0.0, spat_ari - sil_ari) / drop)
            ),
        }

    k_match = {
        est: float(np.mean([r["k_match"] for r in dual if r["estimator"] == est]))
        for est in ("silhouette", "spatial_silhouette", "ensemble")
        if any(r["estimator"] == est for r in dual)
    }
    return {
        "protocol": PROTOCOL,
        "n_cells": len(cells),
        "n_success": len(ok),
        "by_mode": by_mode,
        "recovery": recovery,
        "dual_track_k_match_rate": k_match,
        "slices": list(SLICES),
        "methods": list(METHODS),
    }


def write_long_csv(cells: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset",
        "method",
        "seed",
        "k_policy",
        "estimator",
        "mode",
        "k_used",
        "oracle_k",
        "k_match",
        "ari",
        "seconds",
        "status",
        "error",
        "n_domains_truth",
        "n_obs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in cells:
            writer.writerow(row)


def write_dual_csv(dual: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "dataset",
        "oracle_k",
        "estimated_k",
        "estimator",
        "k_match",
        "seed_for_estimation",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in dual:
            writer.writerow({k: row.get(k) for k in fields})


def make_figure(
    cells: list[dict[str, Any]], summary: dict[str, Any], dual: list[dict[str, Any]]
) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from scientific_figure_pro import (  # type: ignore
        PALETTE,
        apply_publication_style,
        create_subplots,
        finalize_figure,
    )

    apply_publication_style()
    FIG.mkdir(parents=True, exist_ok=True)

    modes = ["oracle", "estimate:silhouette", "estimate:spatial_silhouette", "estimate:ensemble"]
    mode_labels = {
        "oracle": "Oracle-K",
        "estimate:silhouette": "Estimate\nsilhouette",
        "estimate:spatial_silhouette": "Estimate\nspatial_sil",
        "estimate:ensemble": "Estimate\nensemble",
    }
    colors = {
        "oracle": PALETTE["blue_main"],
        "estimate:silhouette": PALETTE["red_strong"],
        "estimate:spatial_silhouette": PALETTE["teal"],
        "estimate:ensemble": PALETTE["green_3"],
    }

    fig, axes = create_subplots(1, 3, figsize=(12.5, 4.2))
    ax0, ax1, ax2 = axes

    # --- Panel A: mean ARI by method × K mode ---
    x = np.arange(len(METHODS), dtype=float)
    width = 0.18
    for i, mode in enumerate(modes):
        means = []
        stds = []
        for method in METHODS:
            key = f"{method}|{mode}"
            rec = summary["by_mode"].get(key)
            means.append(rec["mean_ari"] if rec else np.nan)
            stds.append(rec["std_ari"] if rec else 0.0)
        offset = (i - 1.5) * width
        ax0.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            color=colors[mode],
            label=mode_labels[mode].replace("\n", " "),
            edgecolor="white",
            linewidth=0.5,
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels([m.upper() if m == "spagcn" else m.upper() for m in METHODS])
    ax0.set_ylabel("Mean ARI (5 DLPFC slices)")
    ax0.set_title("A  SOTA ARI under K policy")
    ax0.legend(loc="upper right", fontsize=7)
    ax0.set_ylim(0, max(0.55, ax0.get_ylim()[1] * 1.05))

    # --- Panel B: ARI drop from oracle and recovery ---
    # For each method: silhouette drop (negative), then recovery of spatial_sil / ensemble
    method_labels = list(METHODS)
    drop_vals = []
    rec_spat = []
    rec_ens = []
    for method in METHODS:
        r = summary["recovery"].get(method, {})
        drop_vals.append(r.get("ari_drop_silhouette_vs_oracle", np.nan))
        rec_spat.append(r.get("ari_recovered_by_spatial_silhouette", np.nan))
        rec_ens.append(r.get("ari_recovered_by_ensemble", np.nan))

    x2 = np.arange(len(METHODS), dtype=float)
    w2 = 0.25
    ax1.bar(
        x2 - w2,
        [-d if np.isfinite(d) else np.nan for d in drop_vals],
        width=w2,
        color=PALETTE["red_strong"],
        label="ΔARI silhouette − oracle (loss)",
    )
    # Show loss as negative bars for narrative clarity
    ax1.cla()
    ax1.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--")
    ax1.bar(
        x2 - w2,
        [-d if (d is not None and np.isfinite(d)) else np.nan for d in drop_vals],
        width=w2,
        color=PALETTE["red_strong"],
        label="Silhouette loss vs oracle",
    )
    ax1.bar(
        x2,
        [v if (v is not None and np.isfinite(v)) else np.nan for v in rec_spat],
        width=w2,
        color=PALETTE["teal"],
        label="Recovered by spatial_silhouette",
    )
    ax1.bar(
        x2 + w2,
        [v if (v is not None and np.isfinite(v)) else np.nan for v in rec_ens],
        width=w2,
        color=PALETTE["green_3"],
        label="Recovered by ensemble",
    )
    ax1.set_xticks(x2)
    ax1.set_xticklabels([m for m in method_labels])
    ax1.set_ylabel("ARI change")
    ax1.set_title("B  Oracle-K removal: loss & recovery")
    ax1.legend(loc="best", fontsize=7)

    # Annotate recovery fractions
    for i, method in enumerate(METHODS):
        r = summary["recovery"].get(method, {})
        frac = r.get("fraction_of_drop_recovered_ensemble")
        if frac is not None and np.isfinite(frac):
            ax1.text(
                x2[i] + w2,
                (rec_ens[i] or 0) + 0.005,
                f"{100 * frac:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["green_3"],
            )

    # --- Panel C: Dual-track K match rate ---
    est_order = ["silhouette", "spatial_silhouette", "ensemble"]
    match_rates = [summary["dual_track_k_match_rate"].get(e, np.nan) for e in est_order]
    # Also mean |k_hat - k_oracle|
    abs_err = []
    for est in est_order:
        errs = [
            abs(int(r["estimated_k"]) - int(r["oracle_k"])) for r in dual if r["estimator"] == est
        ]
        abs_err.append(float(np.mean(errs)) if errs else np.nan)
    x3 = np.arange(len(est_order), dtype=float)
    bars = ax2.bar(
        x3,
        match_rates,
        color=[PALETTE["red_strong"], PALETTE["teal"], PALETTE["green_3"]],
        edgecolor="white",
    )
    ax2.set_xticks(x3)
    ax2.set_xticklabels(["silhouette", "spatial_sil", "ensemble"], rotation=15)
    ax2.set_ylabel("Exact K match rate vs oracle")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("C  DualTrackK: estimator vs true K")
    for bar, err in zip(bars, abs_err, strict=True):
        if np.isfinite(err):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"MAE={err:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle(
        "Non-oracle K on DLPFC: SpaGCN & STAGATE without Oracle-K protection",
        fontsize=12,
        y=1.02,
    )
    finalize_figure(fig, FIG / "non_oracle_k_ari_recovery", formats=("svg", "png"))
    logger.info("figure written to %s", FIG / "non_oracle_k_ari_recovery.svg")


def write_report(summary: dict[str, Any], dual: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Non-oracle K SOTA benchmark (DLPFC × SpaGCN / STAGATE)",
        "",
        f"Protocol: `{PROTOCOL}`",
        "",
        "When domain methods receive the true layer count (`k_policy=oracle`), ARI is",
        "inflated relative to real blind analyses. This benchmark removes that protection",
        "(`k_policy=estimate`) and compares legacy expression-only `silhouette` against",
        "spatial-aware `spatial_silhouette` and multi-criterion `ensemble` K estimators,",
        "with dual-track (oracle vs estimated) K recording on every slice.",
        "",
        "## Dual-track K match rates",
        "",
        "| Estimator | Exact match rate |",
        "|---|---|",
    ]
    for est, rate in summary.get("dual_track_k_match_rate", {}).items():
        lines.append(f"| `{est}` | {rate:.2f} |")
    lines += [
        "",
        "## Mean ARI by method × K mode",
        "",
        "| Method | Mode | Mean ARI | Std | K match |",
        "|---|---|---:|---:|---:|",
    ]
    for _key, rec in sorted(summary.get("by_mode", {}).items()):
        lines.append(
            f"| {rec['method']} | `{rec['mode']}` | {rec['mean_ari']:.4f} | "
            f"{rec['std_ari']:.4f} | {rec['k_match_rate']:.2f} |"
        )
    lines += ["", "## Loss and recovery vs Oracle-K", ""]
    for method, rec in summary.get("recovery", {}).items():
        lines += [
            f"### {method}",
            "",
            f"- Oracle mean ARI: **{rec['oracle_mean_ari']:.4f}**",
            f"- Silhouette estimate mean ARI: **{rec['silhouette_mean_ari']:.4f}** "
            f"(drop **{rec['ari_drop_silhouette_vs_oracle']:.4f}**)",
            f"- Spatial silhouette mean ARI: **{rec.get('spatial_silhouette_mean_ari')}** "
            f"(recovered **{rec.get('ari_recovered_by_spatial_silhouette')}**)",
            f"- Ensemble mean ARI: **{rec.get('ensemble_mean_ari')}** "
            f"(recovered **{rec.get('ari_recovered_by_ensemble')}**, "
            f"{(rec.get('fraction_of_drop_recovered_ensemble') or 0) * 100:.0f}% of drop)",
            "",
        ]
    lines += [
        "## Dual-track K per slice",
        "",
        "| Slice | Estimator | Oracle K | Estimated K | Match |",
        "|---|---|---:|---:|---|",
    ]
    for row in dual:
        lines.append(
            f"| {row['dataset']} | `{row['estimator']}` | {row['oracle_k']} | "
            f"{row['estimated_k']} | {row['k_match']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "1. **Oracle-K is an unrealistic upper bound** for published SOTA numbers when",
        "   `n_domains` is taken from expert layers.",
        "2. **Expression-only silhouette** under `k_policy=estimate` can mis-specify K",
        "   and drag SpaGCN / STAGATE ARI down even when the method itself is strong.",
        "3. **spatial_silhouette / ensemble** inject coordinate-aware structure into the",
        "   K decision without reading labels — a practical non-oracle default.",
        "",
        "Figure: `figures/non_oracle_k_ari_recovery.svg`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run-k-only",
        action="store_true",
        help="Only compute DualTrackK reports (no SpaGCN/STAGATE runs).",
    )
    parser.add_argument(
        "--methods",
        default="spagcn,stagate",
        help="Comma-separated methods (default: spagcn,stagate).",
    )
    args = parser.parse_args(argv)
    _setup_logging()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    seeds = _seeds()
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    logger.info("slices=%s methods=%s seeds=%s", SLICES, methods, seeds)

    dual = compute_dual_track_table(seeds)
    dual_path = OUT / "dual_track_k.json"
    dual_path.write_text(json.dumps(dual, indent=2, allow_nan=False), encoding="utf-8")
    write_dual_csv(dual, OUT / "dual_track_k.csv")
    logger.info("DualTrackK written (%d rows)", len(dual))

    cells: list[dict[str, Any]] = []
    if not args.dry_run_k_only:
        for sid in SLICES:
            for method in methods:
                for seed in seeds:
                    for k_policy, estimator in K_MODES:
                        cells.append(run_cell(method, sid, seed, k_policy, estimator))
        write_long_csv(cells, OUT / "benchmark_long.csv")
        (OUT / "benchmark_long.json").write_text(
            json.dumps(cells, indent=2, allow_nan=False, default=str),
            encoding="utf-8",
        )
    else:
        logger.info("dry-run-k-only: skipping SOTA method runs")

    summary = summarize(cells, dual)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    if cells:
        make_figure(cells, summary, dual)
    write_report(summary, dual, OUT / "report_non_oracle_k_sota.md")
    logger.info("done — summary in %s", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
