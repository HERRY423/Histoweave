# Statistical review layer

HistoWeave separates **scoring** (ARI / 1−RMSD / precision@k) from an independent
**statistical review** layer. Reviewers and users should not treat a point-estimate
leaderboard as a claim about significance or rank stability.

## What the layer answers

| Question | API | Output |
|----------|-----|--------|
| How uncertain is a single ARI? | `bootstrap_ari(truth, pred)` | mean + 95% CI (cell bootstrap, refit-free) |
| How stable are method ranks across datasets? | `bootstrap_rank_stability(performance)` | mean rank, rank CI, rank stability |
| What is P(method is best)? | Dirichlet–multinomial posterior over bootstrap ranks | `p_best` per method |
| Does method A beat B? | `paired_permutation_pvalue` across datasets | raw p-value |
| Are multi-method claims FDR-controlled? | `review_landscape` + `fdr_adjust` | BH / BY / Holm / Bonferroni |

```python
from histoweave.benchmark import review_landscape, run_landscape

landscape = run_landscape(datasets, methods=["kmeans", "spectral", "banksy_py"])
report = review_landscape(landscape.performance, n_boot=500, n_perm=1000)
best = report.rank_summary[0]              # best by mean rank
n_sig = report.pairwise["n_significant"]  # FDR discoveries
```

CLI (single-dataset smoke harness):

```bash
histoweave benchmark --stats --n-boot 200
```

Multi-dataset rank FDR is available via `review_landscape()` (not a single-dataset CLI).

## Oracle K is opt-in

Domain methods historically received `n_domains = domain_truth.nunique()` — an
**oracle-K leak** that does not exist in real analyses.

| Policy | Behaviour |
|--------|-----------|
| `estimate` (**default**) | Silhouette / BIC-GMM / gap on expression geometry |
| `fixed` | Caller-supplied K (`n_domains_override`) |
| `oracle` | True domain count — **requires** `allow_oracle_k=True` |

```python
from histoweave.benchmark import estimate_n_domains, run_landscape

sel = estimate_n_domains(data, method="silhouette")
k_hat, curve = sel.k, sel.scores

# Scientific default — no ground-truth K:
run_landscape(datasets, k_policy="estimate")

# Controlled ablation only:
run_landscape(datasets, k_policy="oracle", allow_oracle_k=True)
```

`TaskContract.allow_oracle_k=True` additionally requires `notes` documenting the
oracle ablation so contracts cannot silently publish leaked-K leaderboards.

## Multiple testing / FDR

Gene-level (SVG): Moran's I now writes `morans_i_pval` / `morans_i_padj` with
Benjamini–Hochberg FDR; `uns['svg']['n_significant_fdr']` reports discoveries at
α = 0.05.

Method-level: pairwise permutation p-values across datasets are adjusted with
`fdr_adjust(..., method="bh"|"by"|"holm"|"bonferroni")` inside `review_landscape`.

```python
from histoweave.benchmark import fdr_adjust, reject_nulls

q = fdr_adjust(p_values, method="bh")
sig = reject_nulls(p_values, method="bh", alpha=0.05)
```

## What this is not

* Not a replacement for a biostatistician on a specific manuscript design.
* Single-dataset leaderboards cannot support multi-method FDR across datasets —
  use a landscape with ≥3–5 datasets before claiming pairwise significance.
* Bayesian rank posteriors use a uniform Dirichlet prior over rank bins (simple,
  transparent); they are not a full Plackett–Luce fit.
