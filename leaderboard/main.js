// HistoWeave leaderboard — vanilla D3 + DOM. Loads data.json, renders table,
// heatmap, and small-multiples with reactive filters.
//
// data.json schema (produced by generate.py):
//   {
//     "protocol": "...",
//     "generated_at": "2026-07-14T15:00:00Z",
//     "datasets":  [{"id","platform","tissue","n_obs","n_domains","sparsity"}],
//     "methods":   [{"name","family","summary"}],
//     "records":   [{"dataset","method","seed","ari","seconds"}]
//   }

'use strict';

const state = {
  data: null,
  filterDataset: '__all__',
  filterPlatform: '__all__',
  filterTissue: '__all__',
  filterDomains: '__all__',
  filterSparsity: '__all__',
  filterFamily: '__all__',
  filterTask: '__all__',
  sortBy: 'mean_ari',
};

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

function meanArr(xs) {
  const finite = xs.filter(v => v != null && Number.isFinite(v));
  if (!finite.length) return null;
  return finite.reduce((a, b) => a + b, 0) / finite.length;
}
function stdArr(xs) {
  const finite = xs.filter(v => v != null && Number.isFinite(v));
  if (finite.length < 2) return 0;
  const m = meanArr(finite);
  const v = finite.reduce((s, x) => s + (x - m) ** 2, 0) / (finite.length - 1);
  return Math.sqrt(v);
}
function minArr(xs) {
  const finite = xs.filter(v => v != null && Number.isFinite(v));
  return finite.length ? Math.min(...finite) : null;
}
function maxArr(xs) {
  const finite = xs.filter(v => v != null && Number.isFinite(v));
  return finite.length ? Math.max(...finite) : null;
}

function visibleDatasets() {
  return state.data.datasets.filter(dataset => {
    if (state.filterDataset !== '__all__' && dataset.id !== state.filterDataset) return false;
    if (state.filterPlatform !== '__all__' && dataset.platform !== state.filterPlatform) return false;
    if (state.filterTissue !== '__all__' && dataset.tissue !== state.filterTissue) return false;
    if (state.filterDomains !== '__all__' && String(dataset.n_domains) !== state.filterDomains) return false;
    if (state.filterTask !== '__all__') {
      const task = dataset.task || state.data.task_default || 'spatial_domain';
      if (task !== state.filterTask) return false;
      // P0/P1 contract: never show self-supervised labels on the domain board.
      if (
        state.filterTask === 'spatial_domain' &&
        dataset.ground_truth_kind &&
        ['self_supervised', 'leiden', 'louvain', 'cluster_proxy'].includes(dataset.ground_truth_kind)
      ) {
        return false;
      }
    }
    const sparsity = dataset.sparsity;
    if (state.filterSparsity === 'high' && !(sparsity >= 0.90)) return false;
    if (state.filterSparsity === 'medium' && !(sparsity >= 0.75 && sparsity < 0.90)) return false;
    if (state.filterSparsity === 'low' && !(sparsity < 0.75)) return false;
    return true;
  }).map(dataset => dataset.id);
}
function visibleMethods() {
  let methods = state.data.methods;
  if (state.filterFamily !== '__all__') {
    methods = methods.filter(m => m.family === state.filterFamily);
  }
  return methods.map(m => m.name);
}

// records grouped as: {method → {dataset → [ari per seed]}}
function groupRecords(records, methods, datasets) {
  const grid = {};
  for (const m of methods) {
    grid[m] = {};
    for (const d of datasets) grid[m][d] = [];
  }
  for (const r of records) {
    if (!(r.method in grid)) continue;
    if (!(r.dataset in grid[r.method])) continue;
    grid[r.method][r.dataset].push(r.ari);
  }
  return grid;
}

// ---------------------------------------------------------------------------
// Federated evidence (additive; every function here is a no-op when the feed
// carries no `federation` block, i.e. protocol < v3).
// ---------------------------------------------------------------------------
function federationEnabled() {
  return !!(state.data && state.data.federation && state.data.federation.enabled);
}

// Map "dataset\u0000method" -> consensus cell, for quick per-record lookup.
function federationCellIndex() {
  const idx = {};
  if (!federationEnabled()) return idx;
  for (const c of (state.data.federation.cells || [])) {
    idx[`${c.dataset}\u0000${c.method}`] = c;
  }
  return idx;
}

// Roll a method's cells up to a single badge: verified > unverified > disputed
// precedence for display, plus the max number of contributing labs.
const _STATUS_RANK = { verified: 3, unverified: 2, disputed: 1 };
function methodFederationSummary(method, datasets, cellIndex) {
  let best = null, maxLabs = 0, anyDisputed = false;
  for (const d of datasets) {
    const c = cellIndex[`${d}\u0000${method}`];
    if (!c) continue;
    maxLabs = Math.max(maxLabs, c.n_labs || 0);
    if (c.verification_status === 'disputed') anyDisputed = true;
    if (best == null || (_STATUS_RANK[c.verification_status] || 0) > (_STATUS_RANK[best] || 0)) {
      best = c.verification_status;
    }
  }
  return { status: best, maxLabs, anyDisputed };
}

function fedBadge(summary) {
  if (!summary || summary.status == null) return '<span class="chip nan">local</span>';
  const label = summary.status === 'verified'
    ? `verified · ${summary.maxLabs} labs`
    : (summary.status === 'disputed' ? 'disputed' : `${summary.maxLabs} lab${summary.maxLabs === 1 ? '' : 's'}`);
  return `<span class="chip fed-${summary.status}" title="cross-lab verification status">${label}</span>`;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderStats() {
  const d = state.data;
  document.getElementById('stat-methods').textContent = d.methods.length;
  document.getElementById('stat-datasets').textContent = d.datasets.length;
  const platforms = new Set(d.datasets.map(x => x.platform));
  document.getElementById('stat-platforms').textContent = platforms.size;
  const totalCells = d.datasets.reduce((a, b) => a + (b.n_obs || 0), 0);
  document.getElementById('stat-cells').textContent = totalCells.toLocaleString();
  const proto = d.protocol || '—';
  document.getElementById('build-protocol').textContent = proto;
  document.getElementById('build-date').textContent =
    d.generated_at ? new Date(d.generated_at).toLocaleString() : '—';
  renderFederationBanner();
}

// Additive: a one-line summary of the federated evidence network, shown only
// when the feed is v3 (has a `federation` block). The container is hidden by
// default in index.html, so pre-federation feeds show nothing new.
function renderFederationBanner() {
  const el = document.getElementById('federation-banner');
  if (!el) return;
  if (!federationEnabled()) {
    el.hidden = true;
    return;
  }
  const s = state.data.federation.summary || {};
  const tol = state.data.federation.tolerance;
  const tolStr = tol != null ? ` · tolerance |Δ| ≤ ${tol}` : '';
  el.hidden = false;
  el.innerHTML = `
    <span class="fed-title">Federated evidence network</span>
    <span class="chip fed-verified">${s.n_verified || 0} verified</span>
    <span class="chip fed-unverified">${s.n_unverified || 0} unverified</span>
    <span class="chip fed-disputed">${s.n_disputed || 0} disputed</span>
    <span class="fed-meta">${s.n_nodes || 0} contributing labs · ${s.n_cells || 0} cells${tolStr}</span>
  `;
}

function addOptions(id, values) {
  const select = document.getElementById(id);
  for (const value of [...new Set(values)].filter(value => value != null).sort()) {
    const option = document.createElement('option');
    option.value = String(value);
    option.textContent = String(value);
    select.appendChild(option);
  }
}

function renderFilters() {
  const dsSel = document.getElementById('filter-dataset');
  for (const ds of state.data.datasets) {
    const opt = document.createElement('option');
    opt.value = ds.id;
    opt.textContent = `${ds.id} · ${ds.platform}`;
    dsSel.appendChild(opt);
  }
  addOptions('filter-platform', state.data.datasets.map(dataset => dataset.platform));
  addOptions('filter-tissue', state.data.datasets.map(dataset => dataset.tissue));
  addOptions('filter-domains', state.data.datasets.map(dataset => dataset.n_domains));
}

function bindControls() {
  document.getElementById('filter-dataset').addEventListener('change', e => {
    state.filterDataset = e.target.value; rerender();
  });
  for (const [id, key] of [
    ['filter-platform', 'filterPlatform'],
    ['filter-tissue', 'filterTissue'],
    ['filter-domains', 'filterDomains'],
    ['filter-sparsity', 'filterSparsity'],
  ]) {
    document.getElementById(id).addEventListener('change', event => {
      state[key] = event.target.value;
      rerender();
    });
  }
  document.getElementById('filter-family').addEventListener('change', e => {
    state.filterFamily = e.target.value; rerender();
  });
  const taskFilter = document.getElementById('filter-task');
  if (taskFilter) {
    taskFilter.addEventListener('change', e => {
      state.filterTask = e.target.value; rerender();
    });
  }
  document.getElementById('sort-by').addEventListener('change', e => {
    state.sortBy = e.target.value; rerender();
  });
}

function renderMethodList() {
  const container = document.getElementById('method-list');
  container.innerHTML = '';
  for (const m of state.data.methods) {
    const card = document.createElement('div');
    card.className = 'method-card';
    card.innerHTML = `
      <div class="name">${m.name}</div>
      <div class="desc">${m.summary || ''}</div>
      <div class="fam"><span class="chip ${m.family}">${m.family}</span></div>
    `;
    container.appendChild(card);
  }
}

function renderTable() {
  const datasets = visibleDatasets();
  const methods = visibleMethods();
  const grid = groupRecords(state.data.records, methods, datasets);
  const seconds = groupRecords(
    state.data.records.map(r => ({...r, ari: r.seconds})),
    methods, datasets,
  );

  const stats = methods.map(m => {
    const perDs = datasets.map(d => meanArr(grid[m][d]));
    const secs = datasets.map(d => meanArr(seconds[m][d]));
    const mean = meanArr(perDs);
    const max = maxArr(perDs);
    const seconds_mean = meanArr(secs);
    return { method: m, perDs, mean, max, seconds_mean, secs };
  });

  // Win count per method: how many datasets does this method have the best ARI on?
  const winCount = Object.fromEntries(methods.map(m => [m, 0]));
  datasets.forEach((d, i) => {
    let best = -Infinity, bestM = null;
    for (const s of stats) {
      const v = s.perDs[i];
      if (v != null && Number.isFinite(v) && v > best) { best = v; bestM = s.method; }
    }
    if (bestM) winCount[bestM] += 1;
  });
  stats.forEach(s => { s.wins = winCount[s.method]; });

  // Sort
  const key = state.sortBy;
  stats.sort((a, b) => {
    const sortAsc = (k) => (a[k] ?? -Infinity) - (b[k] ?? -Infinity);
    const sortDesc = (k) => (b[k] ?? -Infinity) - (a[k] ?? -Infinity);
    if (key === 'mean_ari') return sortDesc('mean');
    if (key === 'max_ari') return sortDesc('max');
    if (key === 'win_count') return sortDesc('wins') || sortDesc('mean');
    if (key === 'seconds') return sortAsc('seconds_mean');
    if (key === 'name') return a.method.localeCompare(b.method);
    return 0;
  });

  // Best method per dataset (visible slice)
  const winIdx = datasets.map((d, i) => {
    let best = -Infinity, bestI = -1;
    stats.forEach((s, si) => {
      const v = s.perDs[i];
      if (v != null && Number.isFinite(v) && v > best) { best = v; bestI = si; }
    });
    return bestI;
  });

  // Federation rollup per method (additive; empty object when disabled).
  const fedEnabled = federationEnabled();
  const cellIndex = federationCellIndex();
  const fedByMethod = {};
  if (fedEnabled) {
    for (const s of stats) {
      fedByMethod[s.method] = methodFederationSummary(s.method, datasets, cellIndex);
    }
  }

  // Head — the Evidence column is only present when federation is enabled, so
  // the table is byte-identical to the pre-federation layout otherwise.
  const thead = document.getElementById('lb-head');
  thead.innerHTML = `
    <th>Method</th>
    <th>Family</th>
    ${fedEnabled ? '<th title="Cross-lab verification status">Evidence</th>' : ''}
    <th>Mean ARI</th>
    <th>Best ARI</th>
    <th>Wins</th>
    <th>~ Runtime (s)</th>
    ${datasets.map(d => `<th title="${d}">${d}</th>`).join('')}
  `;

  // Body
  const tbody = document.getElementById('lb-body');
  tbody.innerHTML = '';
  const fmt = (v) => (v == null || !Number.isFinite(v)) ? '<span class="chip nan">n/a</span>' : v.toFixed(3);
  const fmtSec = (v) => (v == null || !Number.isFinite(v)) ? '—' : v.toFixed(1);
  for (const s of stats) {
    const fam = state.data.methods.find(m => m.name === s.method).family;
    const fedCell = fedEnabled ? `<td>${fedBadge(fedByMethod[s.method])}</td>` : '';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${s.method}</td>
      <td><span class="chip ${fam}">${fam}</span></td>
      ${fedCell}
      <td>${fmt(s.mean)}</td>
      <td>${fmt(s.max)}</td>
      <td>${s.wins}</td>
      <td>${fmtSec(s.seconds_mean)}</td>
      ${datasets.map((d, i) => {
        const v = s.perDs[i];
        const isWin = stats.indexOf(s) === winIdx[i] && v != null && Number.isFinite(v);
        const cls = isWin ? ' class="cell-win"' : '';
        const tag = isWin ? ' <span class="chip win">win</span>' : '';
        return `<td${cls}>${fmt(v)}${tag}</td>`;
      }).join('')}
    `;
    tbody.appendChild(row);
  }
}

function renderHeatmap() {
  const datasets = visibleDatasets();
  const methods = visibleMethods();
  const grid = groupRecords(state.data.records, methods, datasets);

  const cellW = 90, cellH = 22;
  const padL = 190, padT = 60, padR = 20, padB = 30;
  const width = padL + cellW * datasets.length + padR;
  const height = padT + cellH * methods.length + padB;

  // Rank methods by mean ARI (desc) so top rows are strongest.
  const ordered = methods.slice().sort((a, b) => {
    const A = meanArr(datasets.map(d => meanArr(grid[a][d]))) ?? -Infinity;
    const B = meanArr(datasets.map(d => meanArr(grid[b][d]))) ?? -Infinity;
    return B - A;
  });

  const svg = d3.select('#heatmap').html('')
    .append('svg')
    .attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', width)
    .attr('height', height);

  // Determine max ARI for colour scale (visible slice only).
  let vmax = 0;
  for (const m of methods) for (const d of datasets) {
    const v = meanArr(grid[m][d]);
    if (v != null && Number.isFinite(v)) vmax = Math.max(vmax, v);
  }
  vmax = Math.max(0.05, vmax);   // guard against all-zero slices
  const colour = d3.scaleLinear()
    .domain([0, vmax * 0.5, vmax])
    // paper -> lime -> orange (Phylo palette)
    .range(['#FAF9F3', '#E9ED4C', '#FF9400']);

  // Column headers (dataset IDs)
  svg.append('g').selectAll('text')
    .data(datasets)
    .join('text')
      .attr('x', (_, i) => padL + i * cellW + cellW / 2)
      .attr('y', padT - 14)
      .attr('text-anchor', 'middle')
      .attr('font-size', 12)
      .attr('font-family', 'Liberation Sans, Arimo, DejaVu Sans, sans-serif')
      .attr('fill', '#000')
      .text(d => d);

  // Row labels (method names, right-aligned in the left gutter)
  svg.append('g').selectAll('text')
    .data(ordered)
    .join('text')
      .attr('x', padL - 12)
      .attr('y', (_, i) => padT + i * cellH + cellH / 2 + 4)
      .attr('text-anchor', 'end')
      .attr('font-size', 12)
      .attr('font-family', 'Liberation Sans, Arimo, DejaVu Sans, sans-serif')
      .attr('fill', '#000')
      .text(d => d);

  // Cells
  const cellData = [];
  ordered.forEach((m, mi) => datasets.forEach((d, di) => {
    cellData.push({ m, d, mi, di, v: meanArr(grid[m][d]) });
  }));
  svg.append('g').selectAll('rect')
    .data(cellData)
    .join('rect')
      .attr('x', c => padL + c.di * cellW + 1)
      .attr('y', c => padT + c.mi * cellH + 1)
      .attr('width', cellW - 2)
      .attr('height', cellH - 2)
      .attr('fill', c => (c.v == null || !Number.isFinite(c.v)) ? '#eee' : colour(c.v))
      .attr('stroke', '#000')
      .attr('stroke-opacity', 0.05)
      .append('title')
      .text(c => `${c.m} @ ${c.d}: ARI=${(c.v ?? NaN).toFixed(3)}`);

  // Numeric text overlay
  svg.append('g').selectAll('text.v')
    .data(cellData)
    .join('text')
      .attr('class', 'v')
      .attr('x', c => padL + c.di * cellW + cellW / 2)
      .attr('y', c => padT + c.mi * cellH + cellH / 2 + 4)
      .attr('text-anchor', 'middle')
      .attr('font-size', 11)
      .attr('font-family', 'Liberation Sans, Arimo, DejaVu Sans, sans-serif')
      .attr('fill', c => {
        if (c.v == null || !Number.isFinite(c.v)) return '#999';
        // Dark ink on light cells, ink stays legible on lime/orange too.
        return '#000';
      })
      .text(c => (c.v == null || !Number.isFinite(c.v)) ? 'n/a' : c.v.toFixed(2));
}

function renderSmallMultiples() {
  const datasets = visibleDatasets();
  const methods = visibleMethods();
  const grid = groupRecords(state.data.records, methods, datasets);

  // Global x-domain across visible datasets, y-domain: fixed 0..1 for ARI.
  const container = document.getElementById('small-multiples');
  container.innerHTML = '';
  const w = 220, h = 90, pad = { l: 30, r: 8, t: 6, b: 22 };

  const x = d3.scaleBand()
    .domain(datasets)
    .range([pad.l, w - pad.r])
    .padding(0.25);
  const y = d3.scaleLinear().domain([0, 1]).range([h - pad.b, pad.t]);

  for (const m of methods) {
    const panel = document.createElement('div');
    panel.className = 'sm-panel';
    const fam = state.data.methods.find(x => x.name === m).family;
    panel.innerHTML = `
      <h3>${m}</h3>
      <div class="sm-sub"><span class="chip ${fam}">${fam}</span></div>
    `;
    container.appendChild(panel);

    const svg = d3.select(panel).append('svg')
      .attr('viewBox', `0 0 ${w} ${h}`)
      .attr('width', w).attr('height', h);

    // Y grid at 0.25, 0.5, 0.75
    [0.25, 0.5, 0.75].forEach(yv => {
      svg.append('line')
        .attr('x1', pad.l).attr('x2', w - pad.r)
        .attr('y1', y(yv)).attr('y2', y(yv))
        .attr('stroke', '#ECE9E2').attr('stroke-width', 1);
    });

    // Per-dataset whisker + point
    for (const d of datasets) {
      const vs = grid[m][d];
      const mn = minArr(vs), mx = maxArr(vs), mean = meanArr(vs);
      const cx = x(d) + x.bandwidth() / 2;
      if (mn != null && mx != null && mx !== mn) {
        svg.append('line')
          .attr('x1', cx).attr('x2', cx)
          .attr('y1', y(mn)).attr('y2', y(mx))
          .attr('stroke', '#0279EE').attr('stroke-width', 2);
      }
      if (mean != null) {
        svg.append('circle')
          .attr('cx', cx).attr('cy', y(mean))
          .attr('r', 3)
          .attr('fill', fam === 'spatial_aware' ? '#75A025' : '#0279EE')
          .attr('stroke', '#000').attr('stroke-opacity', 0.15)
          .append('title')
          .text(`${m} @ ${d}: mean=${mean.toFixed(3)}, min=${(mn ?? NaN).toFixed(3)}, max=${(mx ?? NaN).toFixed(3)}`);
      }
    }

    // X labels (small, rotated only if long)
    svg.append('g').selectAll('text')
      .data(datasets)
      .join('text')
        .attr('x', d => x(d) + x.bandwidth() / 2)
        .attr('y', h - 6)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .attr('font-family', 'Liberation Sans, Arimo, DejaVu Sans, sans-serif')
        .attr('fill', '#444')
        .text(d => d);

    // Y axis labels
    svg.append('text')
      .attr('x', pad.l - 4).attr('y', y(0) + 3)
      .attr('text-anchor', 'end').attr('font-size', 9).attr('fill', '#666').text('0');
    svg.append('text')
      .attr('x', pad.l - 4).attr('y', y(1) + 3)
      .attr('text-anchor', 'end').attr('font-size', 9).attr('fill', '#666').text('1');
    svg.append('text')
      .attr('x', 4).attr('y', pad.t + 10)
      .attr('font-size', 9).attr('fill', '#666').text('ARI');
  }
}

function rerender() {
  renderTable();
  renderHeatmap();
  renderSmallMultiples();
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

async function boot() {
  const res = await fetch('data.json', { cache: 'no-store' });
  if (!res.ok) {
    document.body.innerHTML =
      `<main class="wrap"><section class="panel">
       <h2>Data unavailable</h2>
       <p>Failed to load <code>data.json</code>: HTTP ${res.status}. If this is a
       local check-out, run <code>python leaderboard/generate.py</code> to build
       the feed from the CSV artefacts, then reload.</p>
       </section></main>`;
    return;
  }
  state.data = await res.json();
  renderStats();
  renderFilters();
  renderMethodList();
  bindControls();
  rerender();
}
document.addEventListener('DOMContentLoaded', boot);
