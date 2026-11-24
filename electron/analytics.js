'use strict';

window.InferaStudioAnalytics = (() => {
  const $ = (id) => document.getElementById(id);
  const measured = (value) => Number.isFinite(value) && value >= 0;
  const smallPercent = new Intl.NumberFormat(undefined, { style: 'percent', maximumSignificantDigits: 1 });
  const axisNumber = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 });
  const rateTick = (value) => axisNumber.format(value) + ' tok/s';
  const percentTick = (value) => axisNumber.format(value) + '%';
  const percentValue = (value) => value > 0 && value < .01 ? percentTick(value) : value.toFixed(2) + '%';
  const memoryTick = (value) => axisNumber.format(value) + ' MiB';

  const duration = (value) => {
    if (!measured(value)) return '—';
    if (value > 0 && value < .001) return '<1 ms';
    return value < 1 ? Math.round(value * 1000) + ' ms' : value.toFixed(2) + ' s';
  };
  const timeTick = (value) => {
    if (value === 0) return '0 s';
    if (value < .001) return axisNumber.format(value * 1e6) + ' µs';
    if (value < 1) return axisNumber.format(value * 1000) + ' ms';
    return axisNumber.format(value) + ' s';
  };
  const mint = '--accent';
  const blue = '--analytics-blue';
  const amber = '--analytics-amber';
  const throughput = [
    { key: 'generation_per_second', label: 'Generation', tone: mint },
    { key: 'prompt_per_second', label: 'Prompt processing', tone: blue },
  ];
  const tokenCounts = [
    { key: 'context_tokens', label: 'Context used', tone: blue },
    { key: 'generated_tokens', label: 'Generated', tone: mint },
  ];

  function element(tag, className = '', text = '') {
    const node = document.createElement(tag);
    node.className = className;
    node.textContent = text;
    return node;
  }

  function button(label, action) {
    const node = element('button', 'quiet analytics-link', label);
    node.type = 'button';
    node.addEventListener('click', action);
    return node;
  }

  function card(title) {
    const panel = element('section', 'analytics-card');
    panel.append(element('h3', '', title));
    return panel;
  }

  function placeholder(text) {
    return element('p', 'analytics-placeholder muted', text);
  }

  function graph(title, options) {
    const panel = card(title);
    panel.classList.add('analytics-chart');
    const legend = element('ul', 'analytics-chart-legend');
    for (const { label, tone } of options.series) {
      const entry = element('li', '', label);
      entry.style.setProperty('--series-color', 'var(' + tone + ')');
      legend.append(entry);
    }
    legend.hidden = options.series.length < 2;
    const canvas = element('canvas', 'analytics-line-chart');
    canvas.tabIndex = 0;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', title + '. Use arrow keys to inspect measured values.');
    const reading = element('output', 'analytics-chart-reading');
    const empty = placeholder('');
    const line = window.InferaStudioLineChart.create(canvas, reading, options);
    panel.append(legend, canvas, reading, empty);
    return {
      panel,
      draw: line.draw,
      update(data, emptyMessage) {
        const hasValues = data.some((point) => options.series.some(({ key }) => measured(point[key])));
        canvas.hidden = reading.hidden = !hasValues;
        empty.hidden = hasValues;
        if (!hasValues) empty.textContent = emptyMessage;
        line.update(data);
      },
    };
  }

  function create({ requestContext, processedPromptTokens, rate, percentOf, renderPerformanceDetails }) {
    const dialog = $('inference-analytics');
    const overview = $('analytics-overview');
    const recent = $('analytics-recent');
    const tabs = [$('analytics-overview-tab'), $('analytics-recent-tab')];
    let snapshot;
    let history = [];
    let selected;
    let currentTab = 0;
    let view;
    let recentPlots = [];

    function drawPlots() {
      if (currentTab === 1) {
        for (const plot of recentPlots) plot.draw();
      } else if (view) {
        for (const plot of view.plots) plot.draw();
        if (view.details.open) {
          for (const plot of view.detailPlots) plot.draw();
        }
      }
    }
    new ResizeObserver(drawPlots).observe(dialog);

    function overviewView() {
      const header = element('div', 'analytics-section-heading');
      const title = element('h3');
      const stage = element('span', 'analytics-live');
      header.append(title, stage);
      if (selected) {
        header.append(button('Back to latest request', () => {
          selected = null;
          renderOverview();
          overview.focus();
        }));
      }
      const kpis = element('dl', 'analytics-kpis');
      const values = ['Generation speed', 'Time to first token', 'Context used', 'Request total'].map((label) => {
        const item = element('div');
        const reading = element('dd');
        item.append(element('dt', '', label), reading);
        kpis.append(item);
        return reading;
      });
      values.at(-1).id = 'analytics-elapsed';
      const plots = [
        graph('Token throughput over time', { series: throughput, formatX: timeTick, formatY: rateTick }),
        graph('Token counts over time', { series: tokenCounts, formatX: timeTick, formatY: axisNumber.format }),
      ];
      const grid = element('div', 'analytics-grid');
      grid.append(...plots.map(({ panel }) => panel));
      const detailPlots = [
        graph('Prompt tokens over time', {
          series: [
            { key: 'prompt_tokens', label: 'Prompt tokens', tone: blue },
            { key: 'processed_prompt_tokens', label: 'Newly processed', tone: mint },
            { key: 'cached_prompt_tokens', label: 'Reused from KV cache', tone: amber },
          ],
          formatX: timeTick, formatY: axisNumber.format, formatValue: (value) => value.toLocaleString(),
        }),
        graph('Context utilization over time', {
          series: [{ key: 'context_percent', label: 'Context used', tone: blue }],
          formatX: timeTick, formatY: percentTick, formatValue: percentValue,
        }),
        graph('KV prompt reuse over time', {
          series: [{ key: 'reuse_percent', label: 'Prompt reused', tone: amber }],
          formatX: timeTick, formatY: percentTick, formatValue: percentValue,
        }),
        graph('KV buffer allocation over time', {
          series: [{ key: 'kv_allocated_mib', label: 'Allocated', tone: mint }],
          formatX: timeTick, formatY: memoryTick, formatValue: (value) => value.toFixed(1) + ' MiB',
        }),
      ];
      const detailGrid = element('div', 'analytics-grid analytics-detail-charts');
      detailGrid.append(...detailPlots.map(({ panel }) => panel));
      const details = element('details', 'analytics-card analytics-measurements');
      const body = element('div', 'performance-details');
      details.append(element('summary', '', 'Timing, context and KV cache details'), body, detailGrid);
      details.addEventListener('toggle', drawPlots);
      overview.replaceChildren(header, kpis, grid, details);
      return { title, stage, values, plots, details, body, detailPlots };
    }

    function detailPoints(points, metrics, used) {
      const elapsed = metrics.request_seconds;
      // Older responses may only have these measurements in the request summary.
      // Use the elapsed time of the snapshot so earlier gaps remain visible.
      if (measured(elapsed) && elapsed >= (points.at(-1)?.x ?? 0)) {
        points = points.filter(({ x }) => x < elapsed);
        points.push({
          ...metrics,
          x: elapsed,
          prompt_tokens: metrics.prompt_tokens ?? (metrics.budget_estimated ? null : metrics.budget_prompt),
          context_tokens: used?.prefix === '' ? used.used : metrics.prompt_progress_tokens,
        });
      }
      return points.map((point) => ({
        ...point,
        processed_prompt_tokens: processedPromptTokens(point),
        reuse_percent: point.prompt_tokens > 0 && measured(point.cached_prompt_tokens)
          ? 100 * point.cached_prompt_tokens / point.prompt_tokens : null,
        context_percent: metrics.context > 0 && measured(point.context_tokens)
          ? 100 * point.context_tokens / metrics.context : null,
      }));
    }

    function refreshOverview() {
      const metrics = selected?.metrics ?? snapshot.metrics;
      const live = !selected && snapshot.live;
      if (!metrics) {
        overview.replaceChildren(placeholder(live ? 'Waiting for inference measurements…'
          : 'Send a message in this conversation to see inference measurements.'));
        view = null;
        return;
      }
      const samples = metrics.samples ?? [];
      if (!view) {
        view = overviewView();
        view.details.open = !samples.length && !live;
      }
      view.title.textContent = selected ? 'Request ' + selected.number : live ? 'Current request' : 'Latest request';
      view.stage.hidden = !live;
      view.stage.textContent = live ? snapshot.stage || 'Preparing request' : '';
      const used = requestContext(metrics);
      const fraction = used && used.used / metrics.context;
      const contextPercent = used ? used.prefix + (fraction > 0 && fraction < .001
        ? smallPercent.format(fraction) : percentOf(used.used, metrics.context)) : '—';
      const readings = [
        rate(metrics.generation_per_second), duration(metrics.first_token_seconds),
        contextPercent, duration(metrics.request_seconds),
      ];
      for (const [index, reading] of readings.entries()) view.values[index].textContent = reading;
      const points = samples.map((sample) => ({ ...sample, x: sample.elapsed_seconds }));
      const missing = live ? 'Waiting for measurements…' : 'No time samples were recorded for this request.';
      for (const plot of view.plots) plot.update(points, missing);
      const details = detailPoints(points, metrics, used);
      const missingDetail = live ? 'Waiting for measurements…' : 'No measurement was recorded for this request.';
      for (const plot of view.detailPlots) plot.update(details, missingDetail);
      renderPerformanceDetails(view.body, metrics, { showContextMeter: false, formatTiming: duration });
    }

    function renderOverview() {
      view = null;
      refreshOverview();
    }

    function renderRecent() {
      const requests = history.slice(-12);
      if (!requests.length) {
        recentPlots = [];
        recent.replaceChildren(placeholder('Recorded responses will appear here after generation.'));
        return;
      }
      const charts = element('div', 'analytics-grid');
      const formatX = (number) => '#' + number;
      recentPlots = [
        graph('Generation speed', {
          series: [{ key: 'generation_per_second', label: 'Generation', tone: mint }],
          formatX, formatY: rateTick, discrete: true,
        }),
        graph('Time to first token', {
          series: [{ key: 'first_token_seconds', label: 'First token', tone: blue }],
          formatX, formatY: timeTick, discrete: true,
        }),
      ];
      charts.append(...recentPlots.map(({ panel }) => panel));
      const summary = card('Recorded requests');
      const table = element('table', 'analytics-table');
      table.append(element('caption', 'muted', 'Up to 12 recent responses in this conversation'));
      const heading = table.createTHead().insertRow();
      for (const label of ['Request', 'Generation speed', 'First token', 'Request total']) {
        const cell = element('th', '', label);
        cell.scope = 'col';
        heading.append(cell);
      }
      const body = table.createTBody();
      for (const request of requests.slice().reverse()) {
        const { metrics, number } = request;
        const entry = body.insertRow();
        entry.insertCell().append(button('Request ' + number, () => {
          selected = request;
          showTab(0);
          overview.focus();
        }));
        for (const value of [rate(metrics.generation_per_second), duration(metrics.first_token_seconds),
          duration(metrics.request_seconds)]) entry.insertCell().textContent = value;
      }
      summary.append(table);
      recent.replaceChildren(charts, summary);
      const points = requests.map(({ metrics, number }) => ({ ...metrics, x: number }));
      for (const plot of recentPlots) plot.update(points, 'No recorded measurements.');
    }

    function showTab(index) {
      currentTab = index;
      for (const [i, tab] of tabs.entries()) {
        tab.tabIndex = i === index ? 0 : -1;
        tab.setAttribute('aria-selected', String(i === index));
      }
      overview.hidden = index !== 0;
      recent.hidden = index !== 1;
      (index === 0 ? renderOverview : renderRecent)();
    }

    for (const [index, tab] of tabs.entries()) {
      tab.addEventListener('click', () => showTab(index));
      tab.addEventListener('keydown', (event) => {
        const next = { ArrowLeft: 1 - index, ArrowRight: 1 - index, Home: 0, End: 1 }[event.key];
        if (next == null) return;
        event.preventDefault();
        showTab(next);
        tabs[next].focus();
      });
    }
    $('analytics-close').addEventListener('click', () => dialog.close());
    dialog.addEventListener('close', () => {
      if (dialog.open) return;
      snapshot = selected = view = null;
      history = recentPlots = [];
      overview.replaceChildren();
      recent.replaceChildren();
    });

    function receive(data) {
      const changed = data.messages !== snapshot?.messages;
      snapshot = data;
      if (changed) {
        history = data.messages.filter(({ role, state, performance }) =>
          role === 'assistant' && state !== 'streaming' && performance)
          .map(({ id, performance }, index) => ({ id, metrics: performance, number: index + 1 }));
      }
      return changed;
    }

    return {
      dialog,
      open(data, trigger, messageId) {
        receive(data);
        selected = history.find(({ id }) => id === messageId);
        $('analytics-conversation').textContent = data.title;
        showTab(0);
        overview.scrollTop = recent.scrollTop = 0;
        trigger.focus();
        dialog.showModal();
        drawPlots();
      },
      update(data) {
        if (!dialog.open) return;
        const changed = receive(data);
        if (currentTab === 0 && !selected) refreshOverview();
        if (currentTab === 1 && changed) {
          const focusedRequest = recent.contains(document.activeElement) && document.activeElement.matches('button');
          renderRecent();
          if (focusedRequest) recent.focus();
        }
      },
      elapsed(value) {
        if (!dialog.open || currentTab !== 0 || selected || !snapshot.live) return;
        const reading = $('analytics-elapsed');
        if (reading) reading.textContent = duration(value);
      },
    };
  }

  return { create };
})();
