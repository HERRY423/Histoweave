"""Phase B discovery (exploratory): does a MULTI-METHOD CONSENSUS spatial domain
recover the tertiary-lymphoid-structure (TLS) / immune niche in 10x Visium breast
cancer more faithfully than any SINGLE spatial-domain method?

Falsifiable framing
-------------------
* TARGET (defined independently of any clustering): the TLS niche = spots with a
  high canonical B/T/chemokine signature that are ALSO spatially organised
  (Moran's I on the signature). We define TLS foci as spots co-high in BOTH a
  B-cell axis AND a T-cell axis (both > percentile threshold).
* TEST: for every single method, take the domain that best overlaps the TLS foci
  (max F1 / Jaccard of any one domain vs foci). Do the same for the consensus
  clustering. Hypothesis is SUPPORTED if the consensus recovers the foci better
  than the best single method, and/or if single methods systematically SPLIT the
  foci across several domains (fragmentation) while the consensus keeps them
  together.
* VALIDATION: within the recovered domain, canonical B/T/chemokine markers and an
  independent immune signature must be enriched vs the rest of the tissue.

Honesty: this is EXPLORATORY on one public 10x sample. It is a methodological /
niche-recovery observation, not a validated clinical finding. No claim is made
beyond what the markers + spatial statistics on this sample support.

Usage:
    <py_with_scanpy> analyze_tls_consensus.py <bc_h5ad> <labels_dir> <out_dir>
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _log(message: object) -> None:
    """Emit script progress through the project logging contract."""
    logging.getLogger(__name__).info("%s", message)


# ---- marker panels ----------------------------------------------------------
B_MARKERS = ["MS4A1", "CD79A", "CD79B", "CD19", "CR2", "LTB"]
T_MARKERS = ["CD3D", "CD3E", "CD8A"]
CHEMOKINES = ["CXCL13", "CCL19", "CCL21", "SELL"]
TLS_SIG = B_MARKERS + T_MARKERS + CHEMOKINES  # 13-gene canonical TLS signature
# independent immune validation signature (not used to define foci beyond overlap)
IMMUNE_SIG = [
    "PTPRC",
    "CD52",
    "CD2",
    "CD3G",
    "TRBC2",
    "CCL5",
    "GZMK",
    "CD8B",
    "MS4A1",
    "BANK1",
    "CD19",
    "IGHG1",
    "IGKC",
    "CXCR5",
]


def _score(adata, genes, name):
    import scanpy as sc

    present = [g for g in genes if g in adata.var_names]
    sc.tl.score_genes(adata, present, score_name=name, ctrl_size=50)
    return present


def _morans_i(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = len(values)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]
    z = values - values.mean()
    denom = (z**2).sum()
    num = 0.0
    W = 0.0
    for i in range(n):
        for j in idx[i]:
            num += z[i] * z[j]
            W += 1.0
    return float((n / W) * (num / denom)) if denom > 0 else float("nan")


def _contiguity(mask: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """Fraction of selected spots that have >=1 selected spot among k nearest."""
    from sklearn.neighbors import NearestNeighbors

    sel = np.where(mask)[0]
    if len(sel) < 2:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords[sel])
    idx = idx[:, 1:]
    selset = set(sel.tolist())
    have = sum(1 for row in idx if any(j in selset for j in row))
    return have / len(sel)


def _best_domain_overlap(labels: np.ndarray, foci_mask: np.ndarray):
    """Return (best_f1, best_jaccard, best_domain, n_domains_touching_foci)."""
    best_f1, best_j, best_dom = 0.0, 0.0, None
    touch = 0
    foci = foci_mask.astype(bool)
    nfoci = foci.sum()
    for dom in np.unique(labels):
        dmask = labels == dom
        inter = np.logical_and(dmask, foci).sum()
        if inter == 0:
            continue
        # fraction of this domain's spots that fall in foci
        if inter / dmask.sum() > 0.15 or inter / max(nfoci, 1) > 0.15:
            touch += 1
        prec = inter / dmask.sum()
        rec = inter / max(nfoci, 1)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        union = np.logical_or(dmask, foci).sum()
        j = inter / union if union > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_j, best_dom = f1, j, int(dom)
    return best_f1, best_j, best_dom, touch


def _fragmentation(labels: np.ndarray, foci_mask: np.ndarray) -> int:
    """How many distinct domains are needed to cover >=80% of foci spots."""
    foci = foci_mask.astype(bool)
    nfoci = foci.sum()
    if nfoci == 0:
        return 0
    counts = {}
    for dom in np.unique(labels):
        c = np.logical_and(labels == dom, foci).sum()
        if c > 0:
            counts[dom] = c
    order = sorted(counts.values(), reverse=True)
    cum, k = 0, 0
    for c in order:
        cum += c
        k += 1
        if cum >= 0.8 * nfoci:
            break
    return k


def consensus_labels(label_mat: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Co-association consensus: average agreement across methods -> agglomerative."""
    from sklearn.cluster import AgglomerativeClustering

    n, m = label_mat.shape
    # co-association matrix (fraction of methods that co-cluster i,j)
    co = np.zeros((n, n), dtype=np.float32)
    for c in range(m):
        lab = label_mat[:, c]
        same = (lab[:, None] == lab[None, :]).astype(np.float32)
        co += same
    co /= m
    dist = 1.0 - co
    np.fill_diagonal(dist, 0.0)
    cl = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    return cl.fit_predict(dist)


def main() -> None:
    import anndata as ad
    import scanpy as sc

    bc_h5ad, labels_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    a = ad.read_h5ad(bc_h5ad)
    coords = np.asarray(a.obsm["spatial"], dtype=float)
    # normalized copy for scoring
    norm = a.copy()
    if norm.X.max() > 50:  # looks like counts
        sc.pp.normalize_total(norm, target_sum=1e4)
        sc.pp.log1p(norm)

    b_present = _score(norm, B_MARKERS, "B_score")
    t_present = _score(norm, T_MARKERS, "T_score")
    chemo_present = _score(norm, CHEMOKINES, "chemo_score")
    tls_present = _score(norm, TLS_SIG, "TLS_score")
    imm_present = _score(norm, IMMUNE_SIG, "immune_score")

    tls = norm.obs["TLS_score"].to_numpy()
    bsc = norm.obs["B_score"].to_numpy()
    tsc = norm.obs["T_score"].to_numpy()

    # spatial organisation of the TLS signature
    moran = _morans_i(tls, coords, k=6)

    # TLS foci = co-high B AND T (both > 90th pct)
    b_thr = np.percentile(bsc, 90)
    t_thr = np.percentile(tsc, 90)
    foci = (bsc > b_thr) & (tsc > t_thr)
    contig = _contiguity(foci, coords, k=6)

    # also a signature-based niche (top 5% TLS score) for robustness
    tls_thr = np.percentile(tls, 95)
    niche5 = tls > tls_thr

    # ---- load method labels -------------------------------------------------
    label_files = sorted(labels_dir.glob("labels_*.npy"))
    methods = {}
    for f in label_files:
        name = f.stem.replace("labels_", "")
        methods[name] = np.load(f)
    # 10x graphclust reference (in data)
    if "tenx_graphclust" in a.obs.columns:
        gc = pd.Categorical(a.obs["tenx_graphclust"].astype(str)).codes
        methods["tenx_graphclust"] = np.asarray(gc, dtype=int)

    if not methods:
        raise SystemExit("no method labels found in " + str(labels_dir))

    # align lengths
    n = a.n_obs
    methods = {k: v for k, v in methods.items() if len(v) == n}
    names = sorted(methods)
    label_mat = np.column_stack([methods[k] for k in names])

    # consensus at k = median #domains of the panel
    ks = [len(np.unique(methods[k])) for k in names]
    k_consensus = int(np.median(ks))
    cons = consensus_labels(label_mat, k=k_consensus, seed=0)
    methods["CONSENSUS"] = cons

    # ---- recovery metrics per method vs foci + niche5 -----------------------
    rows = []
    for name, lab in methods.items():
        f1, jac, dom, touch = _best_domain_overlap(lab, foci)
        frag = _fragmentation(lab, foci)
        f1n, jacn, domn, touchn = _best_domain_overlap(lab, niche5)
        rows.append(
            {
                "method": name,
                "n_domains": int(len(np.unique(lab))),
                "foci_best_F1": round(f1, 4),
                "foci_best_Jaccard": round(jac, 4),
                "foci_frag_domains_for_80pct": frag,
                "niche5_best_F1": round(f1n, 4),
                "niche5_best_Jaccard": round(jacn, 4),
                "is_consensus": name == "CONSENSUS",
            }
        )
    res = pd.DataFrame(rows).sort_values("foci_best_F1", ascending=False)
    res.to_csv(out_dir / "recovery_metrics.csv", index=False)

    # ---- marker enrichment inside the recovered CONSENSUS domain ------------
    cons_f1, cons_j, cons_dom, cons_touch = _best_domain_overlap(cons, foci)
    in_dom = cons == cons_dom
    enrich = {}
    for panel_name, panel in [
        ("B", b_present),
        ("T", t_present),
        ("chemokine", chemo_present),
        ("immune_sig", imm_present),
    ]:
        if not panel:
            continue
        expr = np.asarray(norm[:, panel].X.mean(axis=1)).ravel()
        inside = expr[in_dom].mean()
        outside = expr[~in_dom].mean()
        enrich[panel_name] = {
            "inside_mean": float(inside),
            "outside_mean": float(outside),
            "fold": float(inside / outside) if outside > 0 else None,
        }
    # Mann-Whitney on TLS score inside vs outside recovered domain
    from scipy.stats import mannwhitneyu

    u, p = mannwhitneyu(tls[in_dom], tls[~in_dom], alternative="greater")

    summary = {
        "sample": "10x Visium FFPE Human Breast Cancer (public, CC-BY-4.0)",
        "n_spots": int(n),
        "n_genes": int(a.n_vars),
        "framing": "exploratory niche-recovery; not a validated clinical finding",
        "TLS_signature_genes_present": tls_present,
        "morans_I_TLS_signature_k6": round(moran, 4),
        "foci_definition": "B_score>90pct AND T_score>90pct",
        "n_foci_spots": int(foci.sum()),
        "foci_spatial_contiguity_k6": round(contig, 4),
        "n_niche5_spots": int(niche5.sum()),
        "methods_in_panel": names,
        "k_consensus": k_consensus,
        "consensus_recovered_domain": cons_dom,
        "consensus_foci_F1": round(cons_f1, 4),
        "consensus_foci_Jaccard": round(cons_j, 4),
        "best_single_method_foci_F1": float(res[~res.is_consensus]["foci_best_F1"].max()),
        "best_single_method": res[~res.is_consensus].iloc[0]["method"],
        "marker_enrichment_in_consensus_domain": enrich,
        "TLS_score_inside_vs_outside_MWU_p": float(p),
    }
    (out_dir / "discovery_summary.json").write_text(json.dumps(summary, indent=2))

    # persist per-spot table for plotting
    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "TLS_score": tls,
            "B_score": bsc,
            "T_score": tsc,
            "foci": foci.astype(int),
            "niche5": niche5.astype(int),
            "consensus": cons,
            "consensus_recovered": in_dom.astype(int),
        }
    )
    for name in names:
        df[f"m_{name}"] = methods[name]
    df.to_parquet(out_dir / "per_spot.parquet")

    _log(json.dumps(summary, indent=2))
    _log("\n=== recovery_metrics ===")
    _log(res.to_string(index=False))


if __name__ == "__main__":
    main()
