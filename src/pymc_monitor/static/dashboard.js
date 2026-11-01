(function () {
  'use strict';
  const config = JSON.parse(document.getElementById('dashboard-config').dataset.config);
  const $ = id => document.getElementById(id);
  const overview = $('overview'), detail = $('detail'), headerStop = $('stop-all');
  const requested = new Set();
  let stopAllRequested = false, current = null, stopError = '', charts = [];
  const element = (tag, value) => {
    const node = document.createElement(tag);
    node.textContent = String(value ?? 'unavailable');
    return node;
  };
  const appendRow = (table, values) => {
    const row = document.createElement('tr');
    for (const value of values) row.append(element('td', value));
    table.append(row);
  };
  const number = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const cleanName = value => String(value ?? '').replace(/^.*?::/, '');
  const complete = state => Boolean(state && state.complete);
  const canStop = state => Boolean(config.stop_url && state && !state.complete &&
    !state.stop_requested && state.can_stop !== false);
  function status(state) {
    if (!state) return 'Waiting for sampling to start';
    if (state.status === 'error') return 'Run error';
    if (state.status === 'cached' || state.cached) return 'Cached result';
    if (state.status === 'stopped') return 'Run stopped';
    if (state.stop_requested) return 'Stop requested';
    if (state.complete) return 'Run complete';
    return 'Sampling in progress';
  }
  function normalize(data) {
    if (data && Array.isArray(data.models)) {
      return {models: data.models, runs: data.runs || {}, multi: true};
    }
    return {models: [{label: 'progress', name: 'pymc-monitor'}],
      runs: {progress: data || null}, multi: false};
  }
  function urlWithParams(base, label) {
    const url = new URL(base, location.href);
    if (config.token) url.searchParams.append('token', config.token);
    if (label !== null) url.searchParams.append('model', label);
    return url;
  }
  async function requestStop(label) {
    if (!config.stop_url || !window.confirm(label === null
      ? "Stop all models early? You'll keep the draws collected so far."
      : "Stop this model early? You'll keep the draws collected so far.")) return;
    const target = label === null ? headerStop : document.querySelector('.model-stop');
    if (target) target.disabled = true;
    try {
      const response = await fetch(urlWithParams(config.stop_url, label),
        {method: 'POST', credentials: 'same-origin'});
      let acknowledged = response.status === 204;
      if (response.ok && !acknowledged) {
        try { acknowledged = (await response.json()).stopped === true; } catch (_) { /* no ack */ }
      }
      if (!acknowledged) throw new Error(`HTTP ${response.status}: stop not acknowledged`);
      stopError = '';
      if (label === null) {
        stopAllRequested = true;
        for (const model of current?.models || []) {
          if (canStop(current.runs[model.label])) requested.add(model.label);
        }
      } else requested.add(label);
    } catch (error) {
      stopError = `Stop failed: ${error.message}`;
    }
    if (current) render(current);
  }
  headerStop.addEventListener('click', () => requestStop(null));
  $('back').addEventListener('click', event => {
    event.preventDefault();
    location.hash = '';
    if (current) render(current);
  });
  window.addEventListener('hashchange', () => { if (current) render(current); });
  function addSection(parent, heading) {
    const section = element('section', '');
    section.append(element('h3', heading));
    parent.append(section);
    return section;
  }
  function renderTraces(parent, traces) {
    const entries = Object.entries(traces || {}).slice(0, 20);
    if (!entries.length) return;
    const section = addSection(parent, 'Trace previews');
    for (const [name, chains] of entries) {
      if (!Array.isArray(chains)) continue;
      const series = chains.filter(Array.isArray).map(chain => chain.slice(-500).map(value =>
        typeof value === 'number' && Number.isFinite(value) ? value : null));
      if (!series.length || !series.some(chain => chain.length)) continue;
      const figure = element('figure', '');
      figure.append(element('figcaption', cleanName(name)));
      const plots = element('div', '');
      plots.className = 'trace-plots';
      const density = element('div', ''), trace = element('div', '');
      plots.append(density, trace);
      figure.append(plots);
      section.append(figure);
      const colors = ['#60a5fa', '#fbbf24', '#34d399', '#f87171'];
      const length = Math.max(...series.map(chain => chain.length));
      const x = Array.from({length}, (_, index) => index);
      const traceData = [x, ...series.map(chain => x.map((_, index) => chain[index] ?? null))];
      const bins = 60;
      const finite = series.flat().filter(value => value !== null);
      if (!finite.length) continue;
      const lo = Math.min(...finite), hi = Math.max(...finite);
      const width = hi - lo || 1;
      const densityX = Array.from({length: bins}, (_, index) => lo + width * index / (bins - 1));
      const densityData = [densityX, ...series.map(chain => {
        const values = chain.filter(value => value !== null);
        const bandwidth = Math.max(width / 40, 1e-9);
        return densityX.map(point => values.reduce((sum, value) => {
          const z = (point - value) / bandwidth;
          return sum + Math.exp(-z * z / 2);
        }, 0) / (values.length * bandwidth * Math.sqrt(2 * Math.PI) || 1));
      })];
      for (const [container, data, axis] of [[density, densityData, 'Value'], [trace, traceData, 'Draw']]) {
        const plot = new uPlot({width: Math.max(240, Math.min(450, section.clientWidth / 2 - 20)),
          height: 160, legend: {show: false}, series: [{label: axis}, ...series.map((_, i) =>
            ({label: `Chain ${i}`, stroke: colors[i % colors.length], width: 1}))]}, data, container);
        charts.push(plot);
      }
    }
  }
  function runDetails(state) {
    const root = element('div', '');
    if (!state) { root.append(element('p', 'Waiting for sampling to start…')); return root; }
    const phases = addSection(root, 'Phases');
    const list = element('ul', '');
    for (const phase of state.phases || []) list.append(element('li',
      `${phase.label}: ${phase.detail} (${phase.status})`));
    phases.append(list);
    const chains = addSection(root, 'Chains');
    const chainTable = element('table', '');
    appendRow(chainTable, ['Chain', 'Phase', 'Draw', 'Divergences']);
    for (const [id, chain] of Object.entries(state.sampling?.chains || {}))
      appendRow(chainTable, [id, chain.phase, `${chain.draw}/${chain.total}`, chain.divergences]);
    chains.append(chainTable);
    const convergence = addSection(root, 'Convergence');
    const diagnostic = state.convergence || {};
    const verdict = diagnostic.verdict || {};
    convergence.append(element('p', verdict.decision
      ? `${verdict.decision} at ${diagnostic.draws} draws (${verdict.final ? 'final' : 'checkpoint'}); ${verdict.hint || ''}`
      : 'Awaiting diagnostics'));
    const table = element('table', '');
    appendRow(table, ['Variable', 'R-hat', 'Bulk ESS', 'Tail ESS']);
    for (const [name, param] of Object.entries(diagnostic.params || {}))
      appendRow(table, [cleanName(name), param.rhat, param.ess_bulk, param.ess_tail]);
    convergence.append(table);
    const checks = element('ul', '');
    for (const [name, check] of Object.entries(verdict.checks || {}))
      checks.append(element('li', `${name}: ${check.value ?? 'unavailable'}; target ${check.threshold ?? 'none'} (${check.ok === true ? 'pass' : check.ok === false ? 'fail' : 'not gating'})`));
    convergence.append(checks);
    renderTraces(root, state.traces);
    return root;
  }
  function render(view) {
    current = view;
    for (const chart of charts) chart.destroy();
    charts = [];
    overview.replaceChildren();
    detail.replaceChildren();
    const models = view.models;
    const allDone = models.length > 0 && models.every(model => complete(view.runs[model.label]));
    const selection = new URLSearchParams(location.hash.replace(/^#/, '')).get('model');
    const selected = view.multi && models.some(model => model.label === selection) ? selection : null;
    const showOverview = view.multi && selected === null;
    $('back').hidden = !view.multi || showOverview;
    overview.hidden = !showOverview;
    detail.hidden = showOverview;
    headerStop.hidden = !config.stop_url || (view.multi && !showOverview);
    const eligible = models.some(model => canStop(view.runs[model.label]) &&
      !requested.has(model.label));
    headerStop.textContent = allDone ? 'Done' : stopAllRequested && !eligible ? 'Stopping…' :
      view.multi ? 'Stop all' : 'Stop sampling';
    headerStop.disabled = !eligible;
    if (showOverview) {
      for (const model of models) {
        const state = view.runs[model.label];
        const card = element('article', '');
        card.className = 'model-card';
        card.tabIndex = 0;
        card.append(element('h2', cleanName(model.name || model.label)), element('p', status(state)));
        const sampling = state?.sampling;
        if (sampling) card.append(element('p', `${number(sampling.total_draws)} draws; ${number(sampling.total_divergences)} divergences`));
        if (state?.convergence?.verdict) card.append(element('p', state.convergence.verdict.decision || 'Awaiting diagnostics'));
        if (config.stop_url && canStop(state) && !requested.has(model.label)) {
          const button = element('button', 'Stop');
          button.type = 'button';
          button.addEventListener('click', event => { event.stopPropagation(); requestStop(model.label); });
          card.append(button);
        }
        const open = () => { location.hash = `model=${encodeURIComponent(model.label)}`; };
        card.addEventListener('click', open);
        card.addEventListener('keydown', event => { if (event.key === 'Enter') open(); });
        overview.append(card);
      }
    } else {
      const model = models.find(item => item.label === selected) || models[0];
      if (model) {
        const state = view.runs[model.label];
        const heading = element('div', '');
        heading.className = 'card-head';
        heading.append(element('h2', cleanName(model.name || model.label)));
        if (view.multi && config.stop_url) {
          const button = element('button', requested.has(model.label) || state?.stop_requested ? 'Stopping…' : 'Stop');
          button.type = 'button';
          button.className = 'model-stop';
          button.disabled = !canStop(state) || requested.has(model.label);
          button.addEventListener('click', () => requestStop(model.label));
          heading.append(button);
        }
        detail.append(heading, element('p', status(state)), runDetails(state));
      }
    }
    $('status').textContent = stopError || (allDone ? 'All runs complete' :
      view.multi ? 'Models in progress' : status(view.runs.progress));
  }
  async function poll() {
    try {
      const response = await fetch(urlWithParams(config.progress_url, null),
        {credentials: 'same-origin', cache: 'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const view = normalize(await response.json());
      render(view);
      if (!view.models.length || view.models.some(model => !complete(view.runs[model.label])))
        setTimeout(poll, 500);
    } catch (error) {
      $('status').textContent = `Dashboard unavailable: ${error.message}`;
      setTimeout(poll, 1000);
    }
  }
  poll();
})();
