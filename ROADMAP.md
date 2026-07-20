# HistoWeave roadmap

**Status:** v0.1.0 submission freeze (2026-07-18).  
**Package:** `histoweave-spatial`  
**Headline claim:** executable evidence contracts for task-constrained method
decisions — not a new clustering algorithm and not a universal method recommender.

This roadmap is the single planning source of truth. Supporting modules (SOTA
wrappers, ISUS, digital twins, AutoML, discovery niches) remain *infrastructure*
or *evidence producers* unless a milestone below explicitly promotes them.

---

## Guiding principles

1. **One primary claim** — evidence-governed `decide()` with abstention / fallback.
2. **Negative results are first-class** — personalisation that fails the global gate
   must stay `global_default`; Oracle-K leakage must be dual-tracked.
3. **Every published ARI carries a protocol** — at minimum `k_policy`, seed policy,
   and ground-truth kind.
4. **Reference artefacts are frozen paths** — re-runs must match or bump the protocol
   version string.

---

## Current freeze (v0.1.0) — done

| Deliverable | Location |
|-------------|----------|
| Decision protocol + claim boundary | `docs/decision-protocol.md`, `histoweave.decide` |
| Validation ledger (10 scientific + 3 contract) | `release_manifest.py`, `docs/methods/validation/` |
| Protocol endpoints 1–4 | `protocol_endpoints.py`, `protocol_endpoints_results/` |
| Independent study-level personalisation (n≥15) | `independent_personalisation_results/` |
| Fail-closed SOTA wrappers | `plugins/builtin/sota_domains.py` |

---

## Near term (v0.1.x / v0.2.0) — 90 days

### PR stream A — Decision core (merge first)

- Keep `decide()` / `DecisionCard` / evidence roles on main.
- Adversarial evidence corpus: incompatible task/GT/oracle-K fixtures with
  admission rate = 0 in CI.
- Selective regret–coverage figure as a first-class report artefact.

### PR stream B — Statistical honesty for K and ISUS

- Non-oracle K estimators + dual-track reporting (`k_selection.py`).
- **Endpoint 5: Oracle-K leakage impact** (mean ARI oracle − estimate) on archived
  `non_oracle_k_sota/` long tables — *no claim that ensemble already recovers ARI*.
- ISUS: permutation p/Z + gain map; remain post-hoc only.

### PR stream C — Reference artefacts & docs

- Register frozen result directories in the validation index.
- Unify SOTA number tables with explicit `k_policy` / `oracle_k` columns.
- Restore this ROADMAP after any branch that deleted it.

### PR stream D — Release ops

- PyPI Trusted Publisher + Zenodo DOI (see `RELEASE_NOTES_v0.1.0.md`).
- Clean tag from main; avoid publishing from dirty agent branches.

**v0.2.0 success criteria**

- [ ] Main branch builds green; ROADMAP present.
- [ ] Every public SpaGCN/STAGATE/GraphST ARI row states `k_policy`.
- [ ] Protocol endpoints 1–5 re-runnable via `scripts/run_protocol_endpoints.py`.
- [ ] Independent personalisation panel ≥15 real units (already met: 17) with
      gated non-inferiority language only.

---

## Mid term (v0.3+)

| Theme | Do | Do not |
|-------|----|--------|
| Non-oracle K | Method-internal q/resolution criteria; multi-subsample stability | Market “ensemble fixes DLPFC ARI” without multi-tissue proof |
| Personalisation | Expand **real independent studies** only | Slice-LOO superiority claims |
| Interop | SpatialData/Squidpy bridges | Replacing those libraries’ data model |
| Methods | Promote wrappers only with dual-track SOTA grids | Inflating `validated` via mock-only gates |

---

## Explicit non-goals (until re-opened)

- Universal biological correctness of any domain method.
- Target-free ISUS as a pre-execution selector.
- Soft-weighting incompatible tasks into a single leaderboard.
- One-size-fits-all “best method” without a task contract.

---

## How to contribute against this roadmap

1. Open a PR that maps to **one stream** (A/B/C/D) above.
2. Cite which protocol endpoint or claim boundary the PR strengthens.
3. Prefer negative-result preservation over demo-only positive numbers.
4. See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/decision-protocol.md](docs/decision-protocol.md).
