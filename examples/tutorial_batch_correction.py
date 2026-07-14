"""Tutorial 3 companion — compare ComBat and Harmony batch correction.

    python examples/tutorial_batch_correction.py

Builds a two-slide dataset with a deliberate batch offset and reports how well each
method mixes batches while preserving cell-type structure.
See docs/tutorials/03_batch_effect_correction.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import histoweave as ts
from histoweave.data import SpatialTable
from histoweave.plugins import create_method


def _build_two_batch_dataset(seed: int = 0) -> SpatialTable:
    rng = np.random.default_rng(seed)
    n_genes = 60
    centers = [rng.normal(0, 5, size=n_genes) for _ in range(3)]

    def make_slide(offset: float, spots_per_type: int = 60):
        blocks = [rng.normal(c, 1.0, size=(spots_per_type, n_genes)) for c in centers]
        X = np.vstack(blocks) + offset
        ct = np.repeat([f"type{i}" for i in range(3)], spots_per_type)
        return X, ct

    X_a, ct_a = make_slide(0.0)
    X_b, ct_b = make_slide(6.0)
    X = np.vstack([X_a, X_b])
    n = X.shape[0]
    obs = pd.DataFrame(
        {
            "batch": ["slideA"] * len(ct_a) + ["slideB"] * len(ct_b),
            "cell_type": np.concatenate([ct_a, ct_b]),
        },
        index=[f"spot{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene{j}" for j in range(n_genes)])
    data = SpatialTable(X=X, obs=obs, var=var)
    data.obsm["spatial"] = rng.random((n, 2)) * 100
    return data


def _batch_dist(emb: np.ndarray, obs: pd.DataFrame) -> float:
    a = (obs["batch"] == "slideA").to_numpy()
    return float(np.linalg.norm(emb[a].mean(0) - emb[~a].mean(0)))


def _celltype_spread(emb: np.ndarray, obs: pd.DataFrame) -> float:
    cents = np.stack(
        [emb[(obs["cell_type"] == t).to_numpy()].mean(0) for t in obs["cell_type"].unique()]
    )
    return float(
        np.mean([np.linalg.norm(a - b) for i, a in enumerate(cents) for b in cents[i + 1:]])
    )


def main() -> None:
    from sklearn.decomposition import PCA

    print(f"HistoWeave v{ts.__version__}\n")
    data = _build_two_batch_dataset()
    print("Input:", repr(data))

    raw = PCA(n_components=15, random_state=0).fit_transform(np.asarray(data.X))
    rows = [("RAW", _batch_dist(raw, data.obs), _celltype_spread(raw, data.obs))]

    try:
        h = create_method(
            "integration", "harmony",
            batch_key="batch", n_pcs=15, theta=2.0, max_iter_harmony=20, seed=0,
        ).run(data)
        he = h.obsm["X_pca_harmony"]
        rows.append(("HARMONY", _batch_dist(he, h.obs), _celltype_spread(he, h.obs)))
    except ModuleNotFoundError as exc:
        print("Harmony unavailable (install extra 'harmony'):", exc)

    c = create_method("integration", "combat", batch_key="batch").run(data)
    ce = PCA(n_components=15, random_state=0).fit_transform(np.asarray(c.X))
    rows.append(("COMBAT", _batch_dist(ce, c.obs), _celltype_spread(ce, c.obs)))

    print("\n{:<9} {:>12} {:>18}".format("method", "batch_dist", "celltype_spread"))
    print("-" * 41)
    for name, bd, cs in rows:
        print(f"{name:<9} {bd:>12.2f} {cs:>18.2f}")
    print("\nLower batch_dist = better mixed; high celltype_spread = biology preserved.")


if __name__ == "__main__":
    main()
