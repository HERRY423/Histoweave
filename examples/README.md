# Examples

| Path | Purpose |
|------|---------|
| [`workshop_30min.ipynb`](workshop_30min.ipynb) | **One-click 30-minute workshop** (install check → demo report → method compare) |
| [`quickstart.py`](quickstart.py) | CLI-free script: pipeline + HTML report + mini leaderboard |
| [`case_study_intercepted_recommendation.py`](case_study_intercepted_recommendation.py) | **Dry-lab case study:** unjustified recommendations intercepted (`evidence_required` / `global_default` / `abstain`) |
| [`tutorial_real_visium.py`](tutorial_real_visium.py) | Visium-oriented walkthrough |
| [`tutorial_batch_correction.py`](tutorial_batch_correction.py) | Integration / batch |
| [`tutorial_custom_plugin.py`](tutorial_custom_plugin.py) | Write a plugin |

## Workshop (recommended for onboarding)

```bash
pip install "histoweave-spatial[scanpy]"
# From repo root:
jupyter notebook examples/workshop_30min.ipynb
# or
jupyter lab examples/workshop_30min.ipynb
```

Chinese guide: [`docs/zh/quickstart.md`](../docs/zh/quickstart.md).

## Headless quickstart

```bash
python examples/quickstart.py
# → quickstart_report.html
```

## Dry-lab case study — intercept bad recommendations

No tissue download. Four vignettes show the decision protocol refusing confident
but invalid promotions (missing holdout, negative external control, circular GT,
cross-task landscape pollution):

```bash
python examples/case_study_intercepted_recommendation.py
# → intercept_case_report.md
# → intercept_case_cards.json
```

Narrative: [`docs/case-study-intercepted-recommendation.md`](../docs/case-study-intercepted-recommendation.md).
