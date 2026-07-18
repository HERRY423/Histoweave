"""HTML report for digital-twin synthetic validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from jinja2 import BaseLoader, Environment, select_autoescape

from .digital_twin import DigitalTwinValidationResult

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HistoWeave digital-twin validation · {{ dataset_name }}</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 1100px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.5;
         color: #1a1a2e; background: #fff; }
  @media (prefers-color-scheme: dark) {
    body { color: #e6e6f0; background: #14141b; }
    .card, th, code { background: #1e1e28 !important; border-color: #2c2c3a !important; }
  }
  header { border-bottom: 3px solid #4C78A8; padding-bottom: .75rem; margin-bottom: 1.5rem; }
  h1 { margin: 0; font-size: 1.5rem; }
  .sub { color: #7a7a8c; font-size: .9rem; }
  h2 { font-size: 1.1rem; margin-top: 1.8rem; border-left: 4px solid #4C78A8; padding-left: .55rem; }
  .grid { display: flex; flex-wrap: wrap; gap: 1rem; }
  .card { flex: 1 1 140px; background: #f7f8fa; border: 1px solid #e3e6ea;
          border-radius: 10px; padding: .85rem 1rem; }
  .card .k { font-size: 1.35rem; font-weight: 700; }
  .card .l { font-size: .75rem; color: #7a7a8c; text-transform: uppercase; letter-spacing: .04em; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: .5rem; }
  th, td { text-align: left; padding: .4rem .55rem; border-bottom: 1px solid #e3e6ea; }
  th { background: #f2f4f7; font-weight: 600; }
  .warn { background: #fff4e5; border-left: 4px solid #F58518; padding: .6rem .8rem;
          margin: .6rem 0; border-radius: 4px; }
  .note { color: #7a7a8c; font-size: .88rem; }
  code { background: #f2f4f7; padding: .1rem .3rem; border-radius: 3px; }
  footer { margin-top: 2.5rem; font-size: .8rem; color: #9a9aa8;
           border-top: 1px solid #e3e6ea; padding-top: .8rem; }
  tr.top { background: #eaf2fb; }
</style>
</head>
<body>
<header>
  <h1>Digital-twin synthetic validation</h1>
  <div class="sub">dataset: <b>{{ dataset_name }}</b> · {{ generated }} · histoweave v{{ version }}</div>
</header>

<p class="note">
  HistoWeave built a synthetic <b>digital twin</b> that matches the real sample on
  {{ n_match_features }} target-free dimensions (sparsity, library-size stats, Moran's I,
  Hopkins tendency, effective rank, …) while planting known domain labels.
  Methods were benchmarked on the twin; the ranking below is the <b>predicted ranking</b>
  for the real sample (which has no ground truth).
</p>

<h2>Overview</h2>
<div class="grid">
  <div class="card"><div class="k">{{ twin_n_obs }}</div><div class="l">Twin cells</div></div>
  <div class="card"><div class="k">{{ twin_n_vars }}</div><div class="l">Twin genes</div></div>
  <div class="card"><div class="k">{{ n_domains }}</div><div class="l">Planted domains</div></div>
  <div class="card"><div class="k">{{ "%.3f"|format(match_cosine) }}</div><div class="l">Match cosine</div></div>
  <div class="card"><div class="k">{{ "%.3f"|format(match_l2) }}</div><div class="l">Match L2</div></div>
  <div class="card"><div class="k">{{ best_method or "—" }}</div><div class="l">Predicted best</div></div>
</div>

{% if warnings %}
<h2>Warnings</h2>
{% for w in warnings %}
<div class="warn">{{ w }}</div>
{% endfor %}
{% endif %}

<h2>Predicted method ranking (ARI on twin)</h2>
<table>
  <thead><tr><th>Rank</th><th>Method</th><th>ARI</th><th>Seconds</th><th>Notes</th></tr></thead>
  <tbody>
  {% for row in leaderboard %}
    <tr {% if row.rank == 1 %}class="top"{% endif %}>
      <td>{{ row.rank }}</td>
      <td><code>{{ row.method }}</code></td>
      <td>{{ row.score_str }}</td>
      <td>{{ row.seconds_str }}</td>
      <td>{{ row.note }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2>Feature match ({{ n_match_features }} dimensions)</h2>
<table>
  <thead><tr><th>Feature</th><th>Real</th><th>Twin</th><th>|Δ|</th><th>Rel. err</th></tr></thead>
  <tbody>
  {% for row in feature_rows %}
    <tr>
      <td>{{ row.name }}</td>
      <td>{{ row.real }}</td>
      <td>{{ row.twin }}</td>
      <td>{{ row.abs }}</td>
      <td>{{ row.rel }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2>Generator parameters</h2>
<table>
  <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
  <tbody>
  {% for k, v in generator_params.items() %}
    <tr><td><code>{{ k }}</code></td><td>{{ v }}</td></tr>
  {% endfor %}
  </tbody>
</table>

<footer>
  Protocol: histoweave.digital_twin.v{{ schema_version }}.
  Twin ARI is a proxy ranking under statistical matching — not a substitute for
  real biological validation when ground truth later becomes available.
</footer>
</body>
</html>
"""


def build_digital_twin_report(
    result: DigitalTwinValidationResult,
    output_path: str | Path,
) -> Path:
    """Render a self-contained HTML report for a digital-twin validation run."""
    from .. import __version__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    match = result.twin_result.match
    twin = result.twin_result.twin
    leaderboard_rows = []
    for row in result.leaderboard:
        score = row.get("score")
        if score is None:
            score_str = "n/a"
        else:
            try:
                score_str = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_str = "n/a"
        seconds = row.get("seconds")
        seconds_str = f"{seconds:.3f}" if isinstance(seconds, int | float) else "n/a"
        note = row.get("error") or ""
        leaderboard_rows.append(
            {
                "rank": row.get("rank", ""),
                "method": row.get("method", ""),
                "score_str": score_str,
                "seconds_str": seconds_str,
                "note": note,
            }
        )

    feature_rows = []
    for name in match.feature_order:
        feature_rows.append(
            {
                "name": name,
                "real": _fmt(match.real_features.get(name)),
                "twin": _fmt(match.twin_features.get(name)),
                "abs": _fmt(match.absolute_errors.get(name)),
                "rel": _fmt(match.relative_errors.get(name), pct=True),
            }
        )

    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html", "xml"]))
    html = env.from_string(_TEMPLATE).render(
        dataset_name=result.dataset_name,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
        twin_n_obs=twin.n_obs,
        twin_n_vars=twin.n_vars,
        n_domains=int(twin.uns.get("n_domains", 0)),
        match_cosine=match.match_cosine,
        match_l2=match.match_l2,
        best_method=result.best_method(),
        warnings=result.warnings,
        leaderboard=leaderboard_rows,
        feature_rows=feature_rows,
        n_match_features=len(match.feature_order),
        generator_params=match.generator_params,
        schema_version=result.schema_version,
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _fmt(value: object, *, pct: bool = False) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    if pct:
        return f"{100.0 * v:.1f}%"
    if abs(v) >= 1000 or (abs(v) < 1e-3 and v != 0.0):
        return f"{v:.4g}"
    return f"{v:.4f}"
