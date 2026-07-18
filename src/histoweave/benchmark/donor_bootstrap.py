"""Donor-stratified bootstrap CIs for multi-component discovery panels.

Used on DLPFC cryptic-niche cohorts where multiple pure-layer components sit
inside a few biological donors.  Hierarchical resampling avoids treating
sections from the same donor as independent biological replicates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Maynard et al. 2021 spatialLIBD donor ↔ section mapping.
DLPFC_SECTION_TO_DONOR: dict[str, str] = {
    "151507": "Br5292",
    "151508": "Br5292",
    "151509": "Br5292",
    "151510": "Br5292",
    "151669": "Br5595",
    "151670": "Br5595",
    "151671": "Br5595",
    "151672": "Br5595",
    "151673": "Br8100",
    "151674": "Br8100",
    "151675": "Br8100",
    "151676": "Br8100",
}


def section_id_from_slice(slice_id: str) -> str:
    """Normalise ``dlpfc_151508`` / ``151508`` → ``151508``."""
    text = str(slice_id).strip()
    if text.startswith("dlpfc_"):
        text = text[len("dlpfc_") :]
    return text


def donor_for_slice(slice_id: str) -> str:
    section = section_id_from_slice(slice_id)
    return DLPFC_SECTION_TO_DONOR.get(section, f"unknown_{section}")


@dataclass
class DonorBootstrapResult:
    """JSON-serialisable donor-stratified bootstrap summary."""

    protocol: str = "histoweave.donor_bootstrap.v1"
    n_boot: int = 0
    n_components: int = 0
    n_donors: int = 0
    donors: list[str] = field(default_factory=list)
    filter: str = "expected_class==L3_program & direction_ok"
    # Point estimates (donor-equal-weight mean of within-donor component means)
    point: dict[str, float] = field(default_factory=dict)
    # Percentile CIs
    ci_level: float = 0.95
    ci: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-donor observed means (not bootstrapped)
    donor_means: dict[str, dict[str, float]] = field(default_factory=dict)
    # Unstratified (component-level) bootstrap for comparison
    unstratified: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentile_ci(samples: np.ndarray, *, level: float = 0.95) -> dict[str, float]:
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
    }


def _component_bootstrap(
    values: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    level: float,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "std": float("nan"),
        }
    if n == 1:
        v = float(values[0])
        return {"mean": v, "median": v, "ci_low": v, "ci_high": v, "std": 0.0}
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = float(np.mean(values[idx]))
    return _percentile_ci(draws, level=level)


def donor_stratified_bootstrap_l3(
    frame: pd.DataFrame,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
    require_direction_ok: bool = True,
    slice_col: str = "slice_id",
    l3_col: str = "l3_delta_rest",
    myelin_col: str = "myelin_delta_rest",
    class_col: str = "expected_class",
    direction_col: str = "direction_ok",
    weight_col: str | None = "n",
) -> DonorBootstrapResult:
    """Bootstrap CIs for L3-program effect sizes with donor stratification.

    Parameters
    ----------
    frame
        Cohort component table (e.g. ``cohort_component_panel.csv``).
    require_direction_ok
        If True (default), keep only rows with ``direction_ok``.
    weight_col
        If set, within-donor means weight components by this column (spot count).

    Returns
    -------
    DonorBootstrapResult
        Donor-equal-weight means of within-donor component means, with percentile
        CIs from hierarchical bootstrap (resample components within each donor).
    """
    df = frame.copy()
    if class_col in df.columns:
        df = df[df[class_col].astype(str) == "L3_program"]
    if require_direction_ok and direction_col in df.columns:
        df = df[df[direction_col].astype(bool)]
    if df.empty:
        return DonorBootstrapResult(
            n_boot=n_boot,
            notes=["no L3 components after filter"],
        )

    df = df.copy()
    df["_donor"] = df[slice_col].map(donor_for_slice)
    donors = sorted(df["_donor"].unique())
    rng = np.random.default_rng(seed)

    # Observed donor means
    donor_means: dict[str, dict[str, float]] = {}
    for donor in donors:
        sub = df[df["_donor"] == donor]
        w = sub[weight_col].to_numpy(dtype=float) if weight_col and weight_col in sub else None
        if w is not None and w.sum() > 0:
            l3_m = float(np.average(sub[l3_col].to_numpy(dtype=float), weights=w))
            my_m = float(np.average(sub[myelin_col].to_numpy(dtype=float), weights=w))
        else:
            l3_m = float(sub[l3_col].mean())
            my_m = float(sub[myelin_col].mean())
        donor_means[donor] = {
            "n_components": int(len(sub)),
            "n_spots": int(sub[weight_col].sum())
            if weight_col and weight_col in sub
            else int(len(sub)),
            "l3_delta_rest": l3_m,
            "myelin_delta_rest": my_m,
            "direction_rate": float(((sub[l3_col] > 0) & (sub[myelin_col] < 0)).mean()),
        }

    point_l3 = float(np.mean([donor_means[d]["l3_delta_rest"] for d in donors]))
    point_my = float(np.mean([donor_means[d]["myelin_delta_rest"] for d in donors]))
    point_dir = float(np.mean([donor_means[d]["direction_rate"] for d in donors]))

    # Hierarchical bootstrap: within each donor resample components; average donors
    boot_l3 = np.empty(n_boot, dtype=float)
    boot_my = np.empty(n_boot, dtype=float)
    boot_dir = np.empty(n_boot, dtype=float)
    by_donor = {d: df[df["_donor"] == d].reset_index(drop=True) for d in donors}

    for b in range(n_boot):
        d_l3: list[float] = []
        d_my: list[float] = []
        d_dir: list[float] = []
        for donor in donors:
            sub = by_donor[donor]
            n = len(sub)
            idx = rng.integers(0, n, size=n)
            samp = sub.iloc[idx]
            if weight_col and weight_col in samp.columns:
                w = samp[weight_col].to_numpy(dtype=float)
                if w.sum() <= 0:
                    w = np.ones(n)
                l3_m = float(np.average(samp[l3_col].to_numpy(dtype=float), weights=w))
                my_m = float(np.average(samp[myelin_col].to_numpy(dtype=float), weights=w))
            else:
                l3_m = float(samp[l3_col].mean())
                my_m = float(samp[myelin_col].mean())
            d_l3.append(l3_m)
            d_my.append(my_m)
            d_dir.append(float(((samp[l3_col] > 0) & (samp[myelin_col] < 0)).mean()))
        boot_l3[b] = float(np.mean(d_l3))
        boot_my[b] = float(np.mean(d_my))
        boot_dir[b] = float(np.mean(d_dir))

    # Unstratified component bootstrap (over-states independence)
    l3_all = df[l3_col].to_numpy(dtype=float)
    my_all = df[myelin_col].to_numpy(dtype=float)
    dir_all = ((df[l3_col] > 0) & (df[myelin_col] < 0)).to_numpy(dtype=float)

    result = DonorBootstrapResult(
        n_boot=n_boot,
        n_components=int(len(df)),
        n_donors=len(donors),
        donors=donors,
        filter=("expected_class==L3_program" + (" & direction_ok" if require_direction_ok else "")),
        point={
            "l3_delta_rest": point_l3,
            "myelin_delta_rest": point_my,
            "direction_rate": point_dir,
            "l3_positive_fraction": float((df[l3_col] > 0).mean()),
            "myelin_negative_fraction": float((df[myelin_col] < 0).mean()),
        },
        ci_level=ci_level,
        ci={
            "l3_delta_rest": _percentile_ci(boot_l3, level=ci_level),
            "myelin_delta_rest": _percentile_ci(boot_my, level=ci_level),
            "direction_rate": _percentile_ci(boot_dir, level=ci_level),
        },
        donor_means=donor_means,
        unstratified={
            "l3_delta_rest": _component_bootstrap(
                l3_all, n_boot=n_boot, seed=seed + 1, level=ci_level
            ),
            "myelin_delta_rest": _component_bootstrap(
                my_all, n_boot=n_boot, seed=seed + 2, level=ci_level
            ),
            "direction_rate": _component_bootstrap(
                dir_all, n_boot=n_boot, seed=seed + 3, level=ci_level
            ),
        },
        notes=[
            "Donor-equal weight: each donor contributes one mean (components "
            "resampled within donor; optional spot-count weights inside donor).",
            "Unstratified bootstrap treats every component as independent "
            "(anticonservative if sections share donor effects).",
            "direction_rate = fraction of components with l3_delta>0 and myelin_delta<0.",
        ],
    )
    return result


def load_cohort_panel(path: str | Any) -> pd.DataFrame:
    """Load cohort_component_panel.csv."""
    return pd.read_csv(path)
