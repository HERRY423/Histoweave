# PR streams for landing dirty work onto main

Work from agent branches should land as **four small PRs**, not one monolith.
Map each PR to [ROADMAP.md](../ROADMAP.md) streams A–D.

## Suggested order

### PR-A — Decision protocol core

**Include:** `src/histoweave/decision.py`, `benchmark/decision.py`,
`docs/decision-protocol.md`, `docs/vs-squidpy-spatialdata.md`,
`tests/test_decision.py`, related `cli` decide wiring.

**Exclude:** ISUS/K/SOTA long experiments.

**Check:** `pytest tests/test_decision.py -q`

### PR-B — Protocol endpoints + independent personalisation

**Include:** `benchmark/protocol_endpoints.py` (incl. `oracle_k_leakage_impact`),
`benchmark/independent_personalisation.py`, scripts under `scripts/run_protocol_*`,
`tests/test_protocol_endpoints.py`, `tests/test_independent_personalisation.py`,
result dirs `protocol_endpoints_results/`, `independent_personalisation_results/`.

**Check:** `pytest tests/test_protocol_endpoints.py tests/test_independent_personalisation.py -q`

### PR-C — ISUS stats + non-oracle K + reference artefacts

**Include:** `benchmark/isus.py`, `benchmark/k_selection.py`, harness/landscape
defaults, `tests/test_isus.py`, `tests/test_k_selection.py`,
`non_oracle_k_sota/`, `pareto_isus_results/` README flags,
`docs/methods/validation/*` track tables, `docs/statistical-review.md`.

**Check:** `pytest tests/test_isus.py tests/test_k_selection.py tests/test_protocol_endpoints.py -q`

### PR-D — Docs / ROADMAP / SOTA number unification

**Include:** `ROADMAP.md`, `docs/roadmap.md`, `README.md` SOTA track table,
`docs/sota-reproduction.md`, `docs/PR_STREAMS.md`, validation index reference
artefacts section.

**Check:** link paths resolve; no oracle/estimate number mixing without labels.

## Rules

1. Do not delete ROADMAP without a replacement in the same PR.
2. Every public ARI in docs must state `k_policy` or track name.
3. Prefer adding negative results over demo-only positives.
4. Open PRs against `main` from short-lived branches named
   `agent/stream-{a,b,c,d}-…`.
