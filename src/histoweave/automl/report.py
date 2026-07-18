"""HTML report for the spatial AutoML compiler."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from jinja2 import BaseLoader, Environment, select_autoescape

from ..data import SpatialTable
from .compiler import AutoMLResult

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HistoWeave Spatial AutoML · {{ dataset_name }}</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 1140px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.5;
         color: #1a1a2e; background: #fff; }
  @media (prefers-color-scheme: dark) {
    body { color: #e6e6f0; background: #14141b; }
    .card, th, code, pre { background: #1e1e28 !important; border-color: #2c2c3a !important; }
  }
  header { border-bottom: 3px solid #54A24B; padding-bottom: .75rem; margin-bottom: 1.5rem; }
  h1 { margin: 0; font-size: 1.5rem; }
  .sub { color: #7a7a8c; font-size: .9rem; }
  h2 { font-size: 1.1rem; margin-top: 1.8rem; border-left: 4px solid #54A24B; padding-left: .55rem; }
  .grid { display: flex; flex-wrap: wrap; gap: 1rem; }
  .card { flex: 1 1 140px; background: #f7f8fa; border: 1px solid #e3e6ea;
          border-radius: 10px; padding: .85rem 1rem; }
  .card .k { font-size: 1.25rem; font-weight: 700; word-break: break-word; }
  .card .l { font-size: .75rem; color: #7a7a8c; text-transform: uppercase; letter-spacing: .04em; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: .5rem; }
  th, td { text-align: left; padding: .4rem .55rem; border-bottom: 1px solid #e3e6ea; }
  th { background: #f2f4f7; font-weight: 600; }
  tr.pareto { background: #eaf7ea; }
  .warn { background: #fff4e5; border-left: 4px solid #F58518; padding: .6rem .8rem;
          margin: .5rem 0; border-radius: 4px; }
  .note { color: #7a7a8c; font-size: .88rem; }
  code, pre { background: #f2f4f7; padding: .1rem .3rem; border-radius: 3px; }
  pre { padding: .75rem 1rem; overflow-x: auto; }
  .plots { display: flex; flex-wrap: wrap; gap: 1rem; }
  footer { margin-top: 2.5rem; font-size: .8rem; color: #9a9aa8;
           border-top: 1px solid #e3e6ea; padding-top: .8rem; }
  .badge { display: inline-block; background: #54A24B; color: #fff; font-size: .72rem;
           padding: .1rem .4rem; border-radius: 999px; margin-left: .35rem; vertical-align: middle; }
</style>
</head>
<body>
<header>
  <h1>Spatial AutoML report</h1>
  <div class="sub">dataset: <b>{{ dataset_name }}</b> · {{ generated }} · histoweave v{{ version }}</div>
</header>

<p class="note">
  Pipeline: <b>feature extraction → landscape nearest-neighbour retrieval →
  auto-run top-{{ top_k }} methods → multi-objective comparison → Pareto ranking</b>.
  Optionally guided by the natural-language compiler (<code>histoweave ask</code>).
</p>

<blockquote><b>Question:</b> {{ question }}</blockquote>

<h2>Overview</h2>
<div class="grid">
  <div class="card"><div class="k">{{ platform or "unknown" }}</div><div class="l">Platform</div></div>
  <div class="card"><div class="k">{{ task }}</div><div class="l">Task</div></div>
  <div class="card"><div class="k">{{ best_method or "—" }}</div><div class="l">Pareto-preferred</div></div>
  <div class="card"><div class="k">{{ n_neighbours }}</div><div class="l">Reference neighbours</div></div>
  <div class="card"><div class="k">{{ n_methods }}</div><div class="l">Methods executed</div></div>
  <div class="card"><div class="k">{{ n_pareto }}</div><div class="l">On Pareto front</div></div>
</div>

{% if warnings %}
<h2>Warnings</h2>
{% for w in warnings %}
<div class="warn">{{ w }}</div>
{% endfor %}
{% endif %}

<h2>Nearest reference datasets</h2>
<table>
  <thead><tr><th>#</th><th>Dataset</th><th>Similarity</th><th>Platform</th><th>Notes</th></tr></thead>
  <tbody>
  {% for n in neighbours %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><code>{{ n.name }}</code></td>
      <td>{{ "%.3f"|format(n.similarity) if n.similarity is not none else "—" }}</td>
      <td>{{ n.platform or "—" }}</td>
      <td>{{ n.detail }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2>Landscape recommendation (pre-execution)</h2>
<table>
  <thead><tr><th>Rank</th><th>Method</th><th>Score</th><th>Uncertainty</th><th>Support</th></tr></thead>
  <tbody>
  {% for m in rec_methods %}
    <tr>
      <td>{{ loop.index }}</td>
      <td><code>{{ m.method }}</code></td>
      <td>{{ "%.4f"|format(m.score) if m.score is not none else "—" }}</td>
      <td>{{ "%.3f"|format(m.uncertainty) if m.uncertainty is not none else "—" }}</td>
      <td>{{ m.support }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

<h2>Executed methods — Pareto ranking</h2>
<table>
  <thead>
    <tr>
      <th>Pareto rank</th><th>Method</th><th>Quality</th><th>Coherence</th>
      <th>Silhouette</th><th>Consensus ARI</th><th>Seconds</th><th>Domains</th>
    </tr>
  </thead>
  <tbody>
  {% for row in run_rows %}
    <tr {% if row.is_pareto %}class="pareto"{% endif %}>
      <td>{{ row.pareto_rank }}{% if row.is_pareto %}<span class="badge">front</span>{% endif %}</td>
      <td><code>{{ row.method }}</code></td>
      <td>{{ row.quality }}</td>
      <td>{{ row.coherence }}</td>
      <td>{{ row.silhouette }}</td>
      <td>{{ row.consensus }}</td>
      <td>{{ row.seconds }}</td>
      <td>{{ row.n_domains }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>

{% if plots %}
<h2>Spatial maps</h2>
<div class="plots">
  {% for svg in plots %}
    {{ svg | safe }}
  {% endfor %}
</div>
{% endif %}

{% if compiled_rationale %}
<h2>Compiler plan (advisory)</h2>
<p class="note">{{ compiled_rationale }}</p>
<pre>{{ compiled_steps }}</pre>
{% endif %}

<footer>
  Protocol: histoweave.spatial_automl.v{{ schema_version }}.
  Pareto objectives: quality (coherence + silhouette + consensus), speed, landscape recommendation score.
  Without ground truth, quality proxies guide selection — re-evaluate when labels become available.
</footer>
</body>
</html>
"""


def build_automl_report(
    result: AutoMLResult,
    output_path: str | Path,
    *,
    data: SpatialTable | None = None,
) -> Path:
    """Render a complete AutoML HTML report."""
    from .. import __version__

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pareto_by_method = {p.method: p for p in result.pareto}
    run_by_method = {r.method: r for r in result.method_runs}

    run_rows = []
    for name in result.ranked_methods:
        r = run_by_method.get(name)
        p = pareto_by_method.get(name)
        run_rows.append(
            {
                "method": name,
                "pareto_rank": p.pareto_rank if p else "—",
                "is_pareto": bool(p.is_pareto) if p else False,
                "quality": _fmt(r.quality_score if r else None),
                "coherence": _fmt(r.spatial_coherence if r else None),
                "silhouette": _fmt(r.silhouette if r else None),
                "consensus": _fmt(r.consensus_agreement if r else None),
                "seconds": _fmt(r.seconds if r else None),
                "n_domains": r.n_domains if r and r.n_domains is not None else "—",
            }
        )

    neighbours = []
    for item in result.neighbours:
        neighbours.append(
            {
                "name": item.get("name", "?"),
                "similarity": item.get("similarity"),
                "platform": item.get("platform") or item.get("assay"),
                "detail": item.get("task") or item.get("weight") or "",
            }
        )

    rec_methods = []
    for m in (result.recommendation.get("ranked_methods") or [])[:10]:
        rec_methods.append(
            {
                "method": m.get("method"),
                "score": m.get("score"),
                "uncertainty": m.get("uncertainty"),
                "support": m.get("support"),
            }
        )

    plots: list[str] = []
    if data is not None and data.spatial is not None:
        try:
            from ..report.svg import spatial_scatter_svg

            for name, col in result.label_columns.items():
                if col in data.obs.columns:
                    plots.append(
                        spatial_scatter_svg(
                            data.spatial,
                            list(data.obs[col].astype(str)),
                            title=f"Domains · {name}",
                        )
                    )
        except Exception:
            plots = []

    compiled_rationale = ""
    compiled_steps = ""
    if result.compiled_plan:
        compiled_rationale = str(result.compiled_plan.get("rationale") or "")
        steps = result.compiled_plan.get("steps") or []
        lines = []
        for i, step in enumerate(steps, 1):
            lines.append(
                f"{i}. {step.get('category')}:{step.get('method')} "
                f"{step.get('params') or {}} — {step.get('purpose') or ''}"
            )
        compiled_steps = "\n".join(lines)

    env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html", "xml"]))
    # Mark plot SVGs as already-safe HTML fragments.
    html = env.from_string(_TEMPLATE).render(
        dataset_name=result.dataset_name,
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        version=__version__,
        question=result.question,
        platform=result.platform,
        task=result.task,
        best_method=result.best_method(),
        n_neighbours=len(result.neighbours),
        n_methods=len(result.method_runs),
        n_pareto=sum(1 for p in result.pareto if p.is_pareto),
        top_k=len(result.method_runs),
        warnings=result.warnings,
        neighbours=neighbours,
        rec_methods=rec_methods,
        run_rows=run_rows,
        plots=plots,
        compiled_rationale=compiled_rationale,
        compiled_steps=compiled_steps,
        schema_version=result.schema_version,
    )
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid4().hex}")
    try:
        temporary.write_text(html, encoding="utf-8")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def _fmt(value: object) -> str:
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:.4f}"
