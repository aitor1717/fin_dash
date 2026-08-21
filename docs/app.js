// Ported from an early option-a.html + refined-2-shared.js mockup (the
// settled editorial design) plus the composite.js helpers that mockup
// relied on, and four analytics charts from an earlier "precision dark"
// app.js that have no slot in the ported layout (rendered below-fold
// instead -- see index.html).
//
// Colors and glow intensity are read from CSS custom properties at runtime,
// so a pure-CSS edit is enough to retheme.
const CS = getComputedStyle(document.documentElement);
function cv(name, fallback) { const v = CS.getPropertyValue(name).trim(); return v || fallback; }
function hexToRgb(hex) {
  const m = hex.replace('#', '').match(/.{1,2}/g);
  if (!m || m.length < 3) return '43,232,168';
  return m.slice(0, 3).map(h => parseInt(h, 16)).join(',');
}
const GREEN = cv('--green', '#2be8a8'), RED = cv('--red', '#ff5470'), GRAY = cv('--gray', '#6b7690'), BLUE = cv('--blue', '#22e5ff');
const MUTED = cv('--muted', '#7a8096'), GRID = cv('--grid', 'rgba(255,255,255,0.07)');
const GREEN_RGB = hexToRgb(GREEN), RED_RGB = hexToRgb(RED), BLUE_RGB = hexToRgb(BLUE);
const GLOW = parseFloat(cv('--glow-strength', '1')) || 0;
// d.historic.dates is trading-day-aligned (build.py reindexes onto the
// benchmark's own index), so these are trading-day counts, not calendar
// days -- ~21/month, ~252/year.
const DAY_PERIODS = { '1d': 1, '1m': 21, '6m': 126, '1y': 252, all: Infinity };
// Beta/alpha are a regression coefficient: meaningless, or wildly noisy,
// from a handful of points. Unlike return/Sharpe/drawdown (well-defined
// for any window length), the regression always uses at least this many
// trailing days. It expands backward past the selected period when that
// period is shorter (e.g. 1D). See computeKPIsForRange.
const MIN_REGRESSION_DAYS = 30;
const BENCH_LABELS = { SPY: 'SPY', QQQ: 'NASDAQ', DIA: 'DOW' };
const PERIOD_DISPLAY = { '1d': '1D', '1m': '1M', '6m': '6M', '1y': '1Y', all: 'All', custom: 'Custom' };
const PLOTLY_CONFIG = { displayModeBar: false, responsive: true };

let GLOBAL_D = null, currentPeriod = '6m', customStart = null, customEnd = null;

function fmtPct(v, d = 2) {
  if (v == null || Number.isNaN(v)) return '—';
  const s = v.toFixed(d) + '%';
  return v > 0 ? '+' + s : s;
}
function fmtNum(v, d = 2) { return v == null ? '—' : v.toFixed(d); }

// Linear regression (beta=slope, alpha=intercept), same math as Python's
// alpha_beta(). Applied client-side since only the full-period version is
// precomputed server-side.
function regress(x, y) {
  const n = x.length;
  if (n < 2) return { beta: 0, alpha: 0 };
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (x[i] - mx) * (y[i] - my); den += (x[i] - mx) ** 2; }
  const beta = den ? num / den : 0;
  const alpha = my - beta * mx;
  return { beta, alpha };
}

function pctChange(arr) {
  const out = [];
  for (let i = 1; i < arr.length; i++) out.push(arr[i] / arr[i - 1] - 1);
  return out;
}

function isWinning(p) {
  // unrealized_pct is signed for P&L (positive = profit) regardless of
  // side -- see build.py's open_positions loop.
  return p.unrealized_pct >= 0;
}

function getWindowIndices(d, period) {
  const dates = d.historic.dates, n = dates.length;
  if (period === 'custom') {
    let i0 = dates.findIndex(x => x >= customStart);
    if (i0 === -1) i0 = 0;
    let i1 = n - 1;
    for (let i = n - 1; i >= 0; i--) { if (dates[i] <= customEnd) { i1 = i; break; } }
    if (i1 < i0) i1 = i0;
    return [i0, i1];
  }
  if (period === 'all') return [0, n - 1];
  const days = DAY_PERIODS[period];
  return [Math.max(0, n - 1 - days), n - 1];
}

function computeKPIsForRange(d, i0, i1, period) {
  const s = d.summary;
  if (i0 === 0 && i1 === d.historic.dates.length - 1) {
    const spy = s.alpha_beta.SPY || {};
    return { returnPct: s.cumulative_return_pct, sharpe: s.sharpe_ratio, sortino: s.sortino_ratio, alpha: spy.alpha_annualized_pct, beta: spy.beta, maxDD: s.max_drawdown.max_drawdown_pct, winRate: s.win_rate_pct, regressionWindowExpanded: false, maxDDIsIntraday: false };
  }
  const eq = d.historic.equity_strategy_dollars.slice(i0, i1 + 1);
  const todayPortfolio = (d.today.series || {}).portfolio;

  // Sharpe/Sortino/alpha/beta all need enough return observations to be
  // meaningful -- a variance estimate or regression coefficient from a
  // couple of points is noise, not signal. All four share the same
  // floored window: MIN_REGRESSION_DAYS trailing days, expanding
  // backward past [i0,i1] only when that period is shorter. Periods
  // already at or above the minimum (6M/1Y/All) are unaffected.
  const regressI0 = Math.min(i0, Math.max(0, i1 - MIN_REGRESSION_DAYS + 1));
  const eqForRegress = d.historic.equity_strategy_dollars.slice(regressI0, i1 + 1);
  const stratRetForRegress = pctChange(eqForRegress);
  const mean = stratRetForRegress.reduce((a, b) => a + b, 0) / (stratRetForRegress.length || 1);
  const std = Math.sqrt(stratRetForRegress.reduce((a, b) => a + (b - mean) ** 2, 0) / (stratRetForRegress.length || 1));
  const sharpe = std ? (mean / std) * Math.sqrt(252) : 0;
  // Same len>=2 guard as pipeline/metrics.py's sortino_ratio. A
  // single-point downside sample gives 0/0 = NaN, not 0, from a plain
  // std formula.
  const downside = stratRetForRegress.filter(r => r < 0);
  let sortino = 0;
  if (downside.length >= 2) {
    const dMean = downside.reduce((a, b) => a + b, 0) / downside.length;
    const dStd = Math.sqrt(downside.reduce((a, b) => a + (b - dMean) ** 2, 0) / downside.length);
    sortino = dStd ? (mean / dStd) * Math.sqrt(252) : 0;
  }
  const benchForRegress = d.historic.equity_normalized.QQQ ? d.historic.equity_normalized.QQQ.slice(regressI0, i1 + 1) : null;
  let alpha = null, beta = null;
  if (benchForRegress && benchForRegress.length > 2) {
    const r = regress(pctChange(benchForRegress), stratRetForRegress);
    beta = r.beta; alpha = r.alpha * 252 * 100;
  }

  // Max drawdown IS well-defined for a single day on its own (today's own
  // peak-to-trough), unlike Sharpe/Sortino/alpha/beta. So 1D reads the
  // real intraday series instead of borrowing a longer window. The daily
  // historic series has no new realized close on a day with no closed
  // trades, so its own "1D" slice is always a flat 0%, regardless of how
  // today's book is actually moving intraday.
  let maxDD;
  const maxDDIsIntraday = period === '1d' && todayPortfolio && todayPortfolio.length;
  if (maxDDIsIntraday) {
    let peak = -Infinity, dd = 0;
    todayPortfolio.forEach(v => { if (v == null) return; peak = Math.max(peak, v); dd = Math.min(dd, v - peak); });
    maxDD = dd; // already percentage points, same units as today.series itself
  } else {
    let peak = eq[0] ?? 0, dd = 0;
    eq.forEach(v => { peak = Math.max(peak, v); dd = Math.min(dd, (v - peak) / peak); });
    maxDD = dd * 100;
  }

  // Today's actual move, from the 5-minute intraday series, not the
  // 2-point daily-close-to-daily-close change. Matches what the main
  // chart shows for this period (see renderMainChartIntraday).
  let returnPct = eq.length > 1 ? 100 * (eq[eq.length - 1] / eq[0] - 1) : 0;
  if (period === '1d' && todayPortfolio && todayPortfolio.length) {
    returnPct = todayPortfolio[todayPortfolio.length - 1];
  }

  return { returnPct, sharpe, sortino, alpha, beta, maxDD, winRate: s.win_rate_pct, regressionWindowExpanded: regressI0 < i0, maxDDIsIntraday: !!maxDDIsIntraday };
}

function mainChartLayout() {
  return {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'inherit', color: MUTED, size: 10 },
    margin: { l: 42, r: 16, t: 6, b: 20 },
    xaxis: { gridcolor: GRID, showspikes: false }, yaxis: { gridcolor: GRID, showspikes: false },
    legend: { orientation: 'h', y: -0.12, font: { size: 9, color: MUTED } },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: 'rgba(19,18,29,0.92)', bordercolor: 'rgba(255,255,255,0.15)', font: { size: 10 } },
  };
}

function renderMainChart(d, i0, i1, period) {
  if (period === '1d') { renderMainChartIntraday(d); return; }
  const dates = d.historic.dates.slice(i0, i1 + 1);
  // equity_normalized is rebased to 100 at the full history's own start,
  // not the displayed window's. Slicing it directly would start the two
  // lines at whatever value each had at i0 -- usually different for
  // strategy vs. benchmark. Re-rebasing the slice to 0% at its own first
  // point (same idea as renderBenchRow) makes both lines start together
  // for any period.
  const rebase = (arr) => (arr.length ? arr.map(v => 100 * (v / arr[0] - 1)) : arr);
  const eq = rebase(d.historic.equity_normalized.strategy.slice(i0, i1 + 1));
  // QQQ isn't guaranteed to exist. A config that omits it, or a fetch
  // failure that drops the column (see build.py's all-NaN-column-drop),
  // would otherwise crash the chart. Guarded the same way
  // computeKPIsForRange guards its own QQQ read.
  const bench = d.historic.equity_normalized.QQQ ? rebase(d.historic.equity_normalized.QQQ.slice(i0, i1 + 1)) : null;
  // Red when the displayed period is a net loss. Same up/down-by-sign
  // convention as renderSelectedChart, applied to the period's own
  // start/end.
  const color = eq.length > 1 && eq[eq.length - 1] < eq[0] ? RED : GREEN;
  const traces = [];
  if (bench) {
    traces.push({ x: dates, y: bench, type: 'scatter', mode: 'lines', name: 'NASDAQ',
      line: { color: GRAY, width: 1, dash: 'dot', shape: 'spline', smoothing: 0.6 },
      hovertemplate: 'NASDAQ %{y:.1f}%<extra></extra>' });
  }
  if (GLOW > 0) {
    traces.push({ x: dates, y: eq, type: 'scatter', mode: 'lines', name: 'Portfolio (glow)',
      line: { color, width: 11, shape: 'spline', smoothing: 0.7 }, opacity: 0.16 * GLOW,
      hoverinfo: 'skip', showlegend: false });
  }
  traces.push({ x: dates, y: eq, type: 'scatter', mode: 'lines', name: 'Portfolio',
    line: { color, width: 3, shape: 'spline', smoothing: 0.7 },
    hovertemplate: 'Portfolio %{y:.1f}%<extra></extra>' });
  Plotly.newPlot('chart-main', traces, {
    ...mainChartLayout(),
    yaxis: { gridcolor: GRID, showspikes: false, title: { text: '% since period start', font: { size: 9 } } },
  }, PLOTLY_CONFIG);
}

// 1D: the daily-resolution historic series has only ~1-2 points for
// "today" -- too coarse for an intraday view. The pipeline already
// fetches real 5-minute data for benchmarks and the notional-weighted
// open book (market_data.fetch_intraday_today, in today.series). Use
// that instead of a daily-granularity series.
function renderMainChartIntraday(d) {
  const timestamps = d.today.timestamps || [];
  const series = d.today.series || {};
  if (!timestamps.length || (!series.portfolio && !series.QQQ)) {
    document.getElementById('chart-main').innerHTML = '<div class="empty-note">No intraday data yet today (market may be closed).</div>';
    return;
  }
  const t = timestamps.map(x => new Date(x));
  const traces = [];
  if (series.QQQ) {
    traces.push({ x: t, y: series.QQQ, type: 'scatter', mode: 'lines', name: 'NASDAQ',
      line: { color: GRAY, width: 1, dash: 'dot', shape: 'spline', smoothing: 0.6 },
      hovertemplate: 'NASDAQ %{y:.1f}%<extra></extra>' });
  }
  if (series.portfolio) {
    // Same up/down-by-sign convention as renderSelectedChart: first vs.
    // last point of the series shown, not just "is the latest value
    // negative". A session that dipped and fully recovered should read
    // green.
    const firstVal = series.portfolio.find(v => v != null);
    const lastVal = [...series.portfolio].reverse().find(v => v != null);
    const color = firstVal != null && lastVal != null && lastVal < firstVal ? RED : GREEN;
    if (GLOW > 0) {
      traces.push({ x: t, y: series.portfolio, type: 'scatter', mode: 'lines', name: 'Portfolio (glow)',
        line: { color, width: 11, shape: 'spline', smoothing: 0.7 }, opacity: 0.16 * GLOW,
        hoverinfo: 'skip', showlegend: false });
    }
    traces.push({ x: t, y: series.portfolio, type: 'scatter', mode: 'lines', name: 'Portfolio',
      line: { color, width: 3, shape: 'spline', smoothing: 0.7 },
      hovertemplate: 'Portfolio %{y:.1f}%<extra></extra>' });
  }
  Plotly.newPlot('chart-main', traces, {
    ...mainChartLayout(),
    yaxis: { gridcolor: GRID, showspikes: false, title: { text: '% since open', font: { size: 9 } } },
  }, PLOTLY_CONFIG);
}

function renderKPIs(k) {
  const ret = document.getElementById('cq-return');
  ret.textContent = fmtPct(k.returnPct); ret.className = 'v ' + (k.returnPct >= 0 ? 'good' : 'bad');
  const sh = document.getElementById('cq-sharpe');
  sh.textContent = fmtNum(k.sharpe); sh.className = 'v neutral-blue';
  const al = document.getElementById('cq-alpha');
  al.textContent = k.alpha != null ? fmtPct(k.alpha) : '—'; al.className = 'v ' + (k.alpha >= 0 ? 'good' : 'bad');
  const be = document.getElementById('cq-beta');
  be.textContent = k.beta != null ? fmtNum(k.beta) : '—';
  be.className = 'v ' + ((k.beta != null && k.beta >= 0 && k.beta <= 1.5) ? 'neutral-blue' : 'neutral-red');
  const dd = document.getElementById('cq-dd');
  dd.textContent = fmtPct(k.maxDD); dd.className = 'v neutral-blue';
  const win = document.getElementById('cq-win');
  win.textContent = k.winRate.toFixed(1); win.className = 'v neutral-blue';
  const so = document.getElementById('cq-sortino');
  so.textContent = fmtNum(k.sortino); so.className = 'v neutral-blue';

  // Hover-only, not always-visible text. Which window a reading came
  // from is secondary detail -- same "identity moves to the tooltip"
  // pattern used throughout (calendar/position tiles, gauge dots).
  const windowNote = 'Uses a trailing 30-day window since the selected period is too short for a reliable calculation.';
  const windowTitle = k.regressionWindowExpanded ? windowNote : '';
  document.getElementById('cq-alpha-label').title = windowTitle;
  document.getElementById('cq-beta-label').title = windowTitle;
  document.getElementById('cq-sharpe-label').title = windowTitle;
  document.getElementById('cq-sortino-label').title = windowTitle;
  document.getElementById('cq-dd-label').title = k.maxDDIsIntraday
    ? "Today's own intraday peak-to-trough, not the multi-day drawdown reading."
    : '';
}

function update() {
  const [i0, i1] = getWindowIndices(GLOBAL_D, currentPeriod);
  renderMainChart(GLOBAL_D, i0, i1, currentPeriod);
  const kpis = computeKPIsForRange(GLOBAL_D, i0, i1, currentPeriod);
  renderKPIs(kpis);
  renderBenchRow(GLOBAL_D, i0, i1, currentPeriod);
  const betaPeriodLabel = kpis.regressionWindowExpanded ? '30d' : (PERIOD_DISPLAY[currentPeriod] || currentPeriod);
  renderGauges(GLOBAL_D, kpis.beta, betaPeriodLabel);
}

// Plotly's gauge `bar` always fills from the axis minimum to the value;
// there's no native "grows from center" mode. Faked here: hide the bar
// (transparent) and build the fill from `steps` instead -- a dim track
// on either side, one bright step from zero to the value. A beta of
// 0.22 then shows as a short bright arc right of center, not a bar from
// -2. No gap at zero: the bright step touches it directly.
function renderBetaGauge(value, range, color, glowRgb, allTimeBeta, periodLabel) {
  const [lo, hi] = range;
  const dim = 'rgba(127,127,127,0.12)';
  let steps;
  if (value >= 0) {
    const fillEnd = Math.min(hi, value);
    steps = [
      { range: [lo, 0], color: dim },
      { range: [0, fillEnd], color },
      { range: [fillEnd, hi], color: dim },
    ];
  } else {
    const fillStart = Math.max(lo, value);
    steps = [
      { range: [lo, fillStart], color: dim },
      { range: [fillStart, 0], color },
      { range: [0, hi], color: dim },
    ];
  }
  document.getElementById('beta-value').textContent = fmtNum(value, 2);
  // Hover-only, not always-visible text. Which window this reading uses
  // is secondary detail -- same "identity moves to the tooltip" pattern
  // used for the calendar/position tiles and the reference dots.
  const betaLabelEl = document.getElementById('beta-label');
  if (betaLabelEl) betaLabelEl.title = periodLabel ? `Showing: ${periodLabel}` : '';
  // mode:'gauge' only, no built-in Plotly number. Its position isn't
  // controllable -- it lands right at the seam, on top of the balance
  // bar. The custom .gauge-label in the HTML replaces it, placed and
  // colored by CSS/JS instead.
  return Plotly.newPlot('gauge-beta', [{
    type: 'indicator', mode: 'gauge', value,
    gauge: {
      axis: { range, showticklabels: false, ticks: '', tickcolor: 'rgba(0,0,0,0)' },
      bar: { color: 'rgba(0,0,0,0)', thickness: 0.75 }, bgcolor: 'transparent', borderwidth: 0,
      steps,
    },
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: { l: 22, r: 22, t: 16, b: 6 },
  }, PLOTLY_CONFIG).then(() => {
    document.getElementById('gauge-beta').style.filter = GLOW > 0 ? `drop-shadow(0 0 ${7 * GLOW}px rgba(${glowRgb},0.85))` : 'none';
    // Beta's fill tracks the selected period. The all-time value is kept
    // as a reference dot -- blue, distinct from the fill's own green/red
    // status coloring, same pattern as the capital gauge's average-
    // utilization dot.
    if (allTimeBeta != null) positionGaugeDot('gauge-beta', allTimeBeta, range, BLUE, `All-time beta: ${fmtNum(allTimeBeta, 2)}`);
  });
}

const GAUGE_MARGIN = { l: 22, r: 22, t: 16, b: 6 };

// Fill and track are both built from `steps` (same technique as
// renderBetaGauge), not Plotly's `bar` element -- `bar` and `steps` render
// at different radial thicknesses, so a bar-based fill wouldn't reach the
// same width as the steps-based track behind it.
function renderCapitalGauge(value, range, color, glowRgb, avgValue) {
  const [lo, hi] = range;
  const dim = 'rgba(127,127,127,0.12)';
  const fillEnd = Math.max(lo, Math.min(hi, value));
  document.getElementById('cash-value').textContent = fmtNum(value, 1) + '%';
  // mode:'gauge' only -- see renderBetaGauge. Also avoids counter-
  // mirroring Plotly's own number after the scaleY(-1) CSS flip that
  // turns this half into the "U".
  return Plotly.newPlot('gauge-cash', [{
    type: 'indicator', mode: 'gauge', value,
    gauge: {
      axis: { range, showticklabels: false, ticks: '', tickcolor: 'rgba(0,0,0,0)' },
      bar: { color: 'rgba(0,0,0,0)', thickness: 0.75 }, bgcolor: 'transparent', borderwidth: 0,
      steps: [
        { range: [lo, fillEnd], color },
        { range: [fillEnd, hi], color: dim },
      ],
    },
  }], {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    margin: GAUGE_MARGIN,
  }, PLOTLY_CONFIG).then(() => {
    const el = document.getElementById('gauge-cash');
    el.style.filter = GLOW > 0 ? `drop-shadow(0 0 ${7 * GLOW}px rgba(${glowRgb},0.85))` : 'none';
    if (avgValue != null) positionGaugeDot('gauge-cash', avgValue, range, GREEN, `All-time avg utilization: ${fmtNum(avgValue, 1)}%`);
  });
}

// Reads the rendered background arc's own SVG path data instead of
// assuming a thickness fraction. d3-style annulus/ring paths encode both
// radii as explicit numbers after "A" commands, so reading them from the
// path is exact, not a guess. Center and outer radius come from the
// path's bounding box (bottom-center, half-width); inner radius comes
// from the path data itself.
function positionGaugeDot(divId, value, range, color, label) {
  const el = document.getElementById(divId);
  const bgPaths = Array.from(el.querySelectorAll('g.bg-arc path'));
  if (!bgPaths.length) return;
  let bgPath = bgPaths[0], bestW = -1;
  bgPaths.forEach(p => {
    try { const w = p.getBBox().width; if (w > bestW) { bestW = w; bgPath = p; } } catch (e) { /* ignore */ }
  });
  const svg = bgPath.ownerSVGElement;
  const ctm = bgPath.getScreenCTM();
  if (!svg || !ctm) return;
  const bbox = bgPath.getBBox();

  const d = bgPath.getAttribute('d') || '';
  const radii = [...d.matchAll(/A\s*([\d.]+)[, ]/g)].map(m => parseFloat(m[1])).filter(n => !isNaN(n));
  const outerRadiusLocal = radii.length ? Math.max(...radii) : bbox.width / 2;
  const innerRadiusLocal = radii.length > 1 ? Math.min(...radii) : outerRadiusLocal * 0.25;

  function toScreen(x, y) {
    const pt = svg.createSVGPoint(); pt.x = x; pt.y = y;
    return pt.matrixTransform(ctm);
  }

  const centerLocal = { x: bbox.x + bbox.width / 2, y: bbox.y + bbox.height };
  const frac = Math.min(1, Math.max(0, (value - range[0]) / (range[1] - range[0])));
  const angle = Math.PI * (1 - frac); // pi (left, min) -> 0 (right, max)
  const dotLocal = {
    x: centerLocal.x + innerRadiusLocal * Math.cos(angle),
    y: centerLocal.y - innerRadiusLocal * Math.sin(angle),
  };
  const screenPt = toScreen(dotLocal.x, dotLocal.y);

  let dot = document.getElementById(divId + '-dot');
  if (!dot) {
    dot = document.createElement('div');
    dot.id = divId + '-dot';
    dot.style.position = 'absolute';
    dot.style.borderRadius = '50%';
    dot.style.pointerEvents = 'none';
    dot.style.zIndex = '5';
    document.body.appendChild(dot);
  }
  const size = 6;
  dot.style.width = size + 'px';
  dot.style.height = size + 'px';
  dot.style.left = (screenPt.x + window.scrollX - size / 2) + 'px';
  dot.style.top = (screenPt.y + window.scrollY - size / 2) + 'px';
  dot.style.background = color;
  dot.style.boxShadow = `0 0 0 3px ${cv('--card', '#1a1b1f')}`;
  dot.style.pointerEvents = label ? 'auto' : 'none';
  if (label) dot.title = label;
}

// Same bg-arc lookup as positionGaugeDot. centerLocal (bottom-center of
// the arc's bounding box) is the flat "diameter" edge of the half-donut
// in its own authored coordinates, before the CSS flip. getScreenCTM
// gives its true on-screen position either way -- the CTM already
// includes the cash gauge's scaleY(-1). Also returns the arc's rendered
// width (via getBoundingClientRect) so the seam-cover bar below can
// match it exactly.
function measureGaugeArc(divId) {
  const el = document.getElementById(divId);
  const bgPaths = Array.from(el.querySelectorAll('g.bg-arc path'));
  if (!bgPaths.length) return null;
  let bgPath = bgPaths[0], bestW = -1;
  bgPaths.forEach(p => {
    try { const w = p.getBBox().width; if (w > bestW) { bestW = w; bgPath = p; } } catch (e) { /* ignore */ }
  });
  const svg = bgPath.ownerSVGElement;
  const ctm = bgPath.getScreenCTM();
  if (!svg || !ctm) return null;
  const bbox = bgPath.getBBox();
  const pt = svg.createSVGPoint();
  pt.x = bbox.x + bbox.width / 2;
  pt.y = bbox.y + bbox.height;
  return { seamY: pt.matrixTransform(ctm).y, width: bgPath.getBoundingClientRect().width };
}

// The dome and the U should meet exactly at the seam. Plotly centers
// each half-donut within its own margin-adjusted plot box rather than
// pinning it to one edge, so the CSS 50% split between the two
// gauge-mini divs doesn't reliably land on the real seam. Measured
// directly off both arcs' rendered paths and averaged. The balance bar
// is pinned to that exact pixel and widened to the arcs' measured
// diameter (see .ls-bar), so it also covers the seam instead of
// floating over a gap.
//
// The FILL (ls-bar), not the block's overall midpoint, needs to sit on
// the seam -- .ls-block also holds the "Balance" title above the bar.
// .ls-block is position:absolute, so it's offsetParent for its
// children: bar.offsetTop plus half its height is the bar's distance
// from the block's own top edge. Subtracting that out targets the bar.
function positionBalanceBar() {
  const beta = measureGaugeArc('gauge-beta');
  const cash = measureGaugeArc('gauge-cash');
  if (!beta || !cash) return;
  const block = document.querySelector('.ls-block');
  const bar = document.getElementById('ls-bar');
  const container = document.querySelector('.gauge-circle');
  if (!block || !bar || !container) return;
  bar.style.width = Math.max(beta.width, cash.width) + 'px';
  const seamY = (beta.seamY + cash.seamY) / 2;
  const barCenterOffset = bar.offsetTop + bar.offsetHeight / 2;
  block.style.top = (seamY - container.getBoundingClientRect().top - barCenterOffset) + 'px';
}

// Fills grow outward from the center: green rightward with the account
// fraction allocated long, red leftward with the fraction allocated
// short, each capped at half the bar's width. 100% allocated to one
// side fills that whole half, edge to seam. Both start at the center,
// so they always meet with no gap. Based on capital committed
// (notional), not headcount -- two very differently sized positions
// shouldn't count the same.
function renderLongShortBar(d) {
  const bar = document.getElementById('ls-bar');
  bar.innerHTML = '';
  const rows = d.open_positions;
  const accountSize = d.config.account_size;
  if (!rows.length || !accountSize) {
    const seg = document.createElement('div'); seg.style.flex = '1'; seg.style.background = BLUE;
    bar.appendChild(seg);
    return;
  }
  const longNotional = rows.filter(p => p.side === 'long').reduce((a, p) => a + p.notional_dollars, 0);
  const shortNotional = rows.filter(p => p.side === 'short').reduce((a, p) => a + p.notional_dollars, 0);
  const longFrac = Math.min(1, longNotional / accountSize);
  const shortFrac = Math.min(1, shortNotional / accountSize);
  const mk = (w, color) => {
    const e = document.createElement('div');
    e.style.flex = String(Math.max(w, 0.0001));
    if (color) e.style.background = color;
    return e;
  };
  bar.appendChild(mk(0.5 * (1 - shortFrac)));       // empty, far left
  bar.appendChild(mk(0.5 * shortFrac, RED));         // red, grows leftward from center
  bar.appendChild(mk(0.5 * longFrac, GREEN));        // green, grows rightward from center
  bar.appendChild(mk(0.5 * (1 - longFrac)));         // empty, far right
}

// periodBeta: the selected period's beta, same value as the top KPI row
// (including its MIN_REGRESSION_DAYS floor for short periods). The
// gauge's fill tracks this instead of the all-time figure, so the gauge
// and KPI row never disagree. Capital utilization is a "right now"
// reading, not period-dependent, and is untouched.
function renderGauges(d, periodBeta, periodLabel) {
  // Force a layout reflow before either gauge renders. Without this,
  // beta (rendered first) gets measured by Plotly before the flex
  // column computes its 50/50 split. Capital (rendered second) then
  // measures correctly, since beta's own render forced a reflow as a
  // side effect, leaving beta stuck tiny. Reading offsetHeight forces
  // the browser to settle layout first.
  const betaWrap = document.getElementById('gauge-beta-wrap');
  const cashWrap = document.getElementById('gauge-cash-wrap');
  if (betaWrap) void betaWrap.offsetHeight;
  if (cashWrap) void cashWrap.offsetHeight;

  const s = d.summary;
  const allTimeBeta = (s.alpha_beta.SPY || {}).beta ?? null;
  const beta = periodBeta ?? allTimeBeta ?? 0;
  const betaOk = beta >= 0 && beta <= 1.5;
  const betaP = renderBetaGauge(beta, [-2, 2], betaOk ? GREEN : RED, betaOk ? GREEN_RGB : RED_RGB, allTimeBeta, periodLabel);
  renderLongShortBar(d);

  const utilSeries = d.historic.capital_utilization_pct || [];
  const cash = utilSeries[utilSeries.length - 1] || 0;
  const avgUtil = utilSeries.length ? utilSeries.reduce((a, b) => a + b, 0) / utilSeries.length : null;
  // Unlike beta and the long/short balance bar, capital utilization has
  // no green/red status coding. It always fills blue, regardless of level.
  // Range is sized to the data, not hardcoded to [0, 100] -- this repo
  // models no margin/leverage, but a value can still exceed 100% between
  // consecutive util samples, and a hardcoded ceiling would clamp the
  // avg-utilization dot to the rightmost edge instead of pinning it at
  // its real position. 100 stays the floor so a normal (<=100%) book
  // still gets the full gauge width; headroom rounds up to the nearest 10.
  const utilCeiling = Math.max(100, cash, avgUtil || 0);
  const utilMax = Math.ceil(utilCeiling / 10) * 10;
  const cashP = renderCapitalGauge(cash, [0, utilMax], BLUE, BLUE_RGB, avgUtil);

  Promise.all([betaP, cashP]).then(positionBalanceBar);
}

function renderBubbleChart(d) {
  const rows = d.open_positions;
  const el = document.getElementById('chart-bubble');
  if (!rows.length) { el.innerHTML = '<div class="empty-note">No open positions</div>'; return; }
  const accountSize = d.config.account_size;
  const xs = rows.map(p => p.unrealized_dollars);
  const ys = rows.map(p => 100 * p.unrealized_dollars / accountSize);
  const sizes = rows.map(p => p.notional_dollars);
  const colors = rows.map(p => isWinning(p) ? GREEN : RED);
  const text = rows.map(p => p.ticker);

  const sumX = xs.reduce((a, b) => a + b, 0);
  const sumY = ys.reduce((a, b) => a + b, 0);
  const sumSize = sizes.reduce((a, b) => a + b, 0);

  const xMin = Math.min(...xs, sumX, 0), xMax = Math.max(...xs, sumX, 0);
  const yMin = Math.min(...ys, sumY, 0), yMax = Math.max(...ys, sumY, 0);
  const xSpan = (xMax - xMin) || 1, ySpan = (yMax - yMin) || 1;
  const xRange = [xMin - 0.35 * xSpan, xMax + 0.35 * xSpan];
  const yRange = [yMin - 0.35 * ySpan, yMax + 0.35 * ySpan];

  const sizeref = 2 * Math.max(...sizes, 1) / (46 ** 2);
  const sumSizeref = 2 * Math.max(sumSize, 1) / (58 ** 2);
  const traces = [];
  if (GLOW > 0) {
    traces.push({ x: xs, y: ys, mode: 'markers', type: 'scatter',
      marker: { size: sizes, sizemode: 'area', sizeref, sizemin: 16, color: 'rgba(0,0,0,0)', line: { color: colors, width: 9 }, opacity: 0.2 * GLOW },
      hoverinfo: 'skip', showlegend: false });
  }
  traces.push({ x: xs, y: ys, mode: 'markers', type: 'scatter', text,
    marker: { size: sizes, sizemode: 'area', sizeref, sizemin: 16,
      color: 'rgba(0,0,0,0)', line: { color: colors, width: 2.2 } },
    hovertemplate: '%{text}<br>%{y:.2f}% of account<extra></extra>', showlegend: false });
  // The "all positions" marker gets its own slight glow and fill, always
  // on regardless of the page's global glow setting. It's the one spot
  // meant to read as slightly lifted off the flat aesthetic elsewhere.
  traces.push({ x: [sumX], y: [sumY], mode: 'markers', type: 'scatter',
    marker: { size: [sumSize], sizemode: 'area', sizeref: sumSizeref, sizemin: 20, color: 'rgba(0,0,0,0)', line: { color: BLUE, width: 9 }, opacity: 0.1 },
    hoverinfo: 'skip', showlegend: false });
  traces.push({ x: [sumX], y: [sumY], mode: 'markers', type: 'scatter', text: ['All positions'],
    marker: { size: [sumSize], sizemode: 'area', sizeref: sumSizeref, sizemin: 20, color: `rgba(${BLUE_RGB},0.05)`, line: { color: BLUE, width: 2.6 } },
    hovertemplate: '%{text}<br>%{y:.2f}% of account<extra></extra>', showlegend: false });

  Plotly.newPlot('chart-bubble', traces, {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', showlegend: false,
    font: { family: 'inherit', color: MUTED, size: 10 },
    margin: { l: 40, r: 14, t: 10, b: 30 },
    xaxis: { gridcolor: GRID, range: xRange, zeroline: true, zerolinecolor: 'rgba(150,150,150,0.35)', zerolinewidth: 1, title: { text: 'Growth', font: { size: 9 } } },
    yaxis: { gridcolor: GRID, range: yRange, zeroline: true, zerolinecolor: 'rgba(150,150,150,0.35)', zerolinewidth: 1, title: { text: 'Growth (% of account)', font: { size: 9 } } },
  }, PLOTLY_CONFIG);
}

// No visible text on the cells themselves. At this size (60 in a row)
// text would just be noise; the date/%/position-count detail lives in
// the native hover tooltip (el.title). Flat/zero days are left empty,
// not colored -- this palette has no "flat" hue, just green/red
// intensity or nothing.
function renderCalendarTiles(container, d, n) {
  const dates = d.historic.dates, eq = d.historic.equity_strategy_dollars;
  const accountSize = d.config.account_size;
  const countArr = d.historic.open_position_count || [];
  if (!dates.length) { container.innerHTML = '<div class="empty-note">No data</div>'; return; }

  const startIdx = Math.max(1, eq.length - n);
  const shown = [];
  for (let i = startIdx; i < eq.length; i++) shown.push(100 * (eq[i] - eq[i - 1]) / accountSize);
  const maxAbs = Math.max(...shown.map(Math.abs), 0.01);

  container.innerHTML = '';
  for (let i = startIdx; i < eq.length; i++) {
    const pnl = eq[i] - eq[i - 1];
    const pnlPct = 100 * pnl / accountSize;
    const nPos = countArr[i] || 0;
    const el = document.createElement('div'); el.className = 'cal-cell';
    const t = Math.min(1, Math.abs(pnlPct) / maxAbs);
    const alpha = 0.45 + 0.55 * t;
    let bg, glow;
    if (pnl > 0) { bg = `rgba(${GREEN_RGB},${alpha})`; glow = `rgba(${GREEN_RGB},${alpha * 0.85 * GLOW})`; }
    else if (pnl < 0) { bg = `rgba(${RED_RGB},${alpha})`; glow = `rgba(${RED_RGB},${alpha * 0.85 * GLOW})`; }
    else { bg = 'transparent'; glow = 'transparent'; }
    el.style.background = bg;
    el.style.boxShadow = (GLOW > 0 && pnl !== 0) ? `0 0 ${14 * GLOW}px -1px ${glow}` : 'none';
    const dateTag = new Date(dates[i] + 'T00:00:00Z').toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric', timeZone: 'UTC' });
    el.title = `${dateTag} -- ${(pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(1)}% -- ${nPos} positions`;
    container.appendChild(el);
  }
}

function renderPositionTiles(container, d, n) {
  const rows = [...d.open_positions].sort((a, b) => Math.abs(b.unrealized_pct) - Math.abs(a.unrealized_pct)).slice(0, n);
  if (!rows.length) { container.innerHTML = '<div class="empty-note">No open positions</div>'; return; }
  const maxAbs = Math.max(...rows.map(p => Math.abs(p.unrealized_pct)), 0.01);
  container.innerHTML = '';
  rows.forEach(p => {
    const el = document.createElement('div'); el.className = 'pos-tile'; el.dataset.ticker = p.ticker;
    const good = isWinning(p);
    const t = Math.min(1, Math.abs(p.unrealized_pct) / maxAbs);
    const alpha = 0.4 + 0.55 * t;
    const color = good ? `rgba(${GREEN_RGB},${alpha})` : `rgba(${RED_RGB},${alpha})`;
    const glow = good ? `rgba(${GREEN_RGB},${alpha * 0.8 * GLOW})` : `rgba(${RED_RGB},${alpha * 0.8 * GLOW})`;
    el.style.borderColor = color;
    el.style.boxShadow = GLOW > 0 ? `0 0 12px -2px ${glow}` : 'none';
    el.innerHTML = `<div class="pt-ticker">${p.ticker}</div><div class="pt-pct">${fmtPct(p.unrealized_pct, 1)}</div>`;
    el.title = `${p.ticker} (${p.side}) ${fmtPct(p.unrealized_pct, 1)} -- click for intraday`;
    el.addEventListener('click', () => selectTicker(p.ticker));
    container.appendChild(el);
  });
}

function renderBenchRow(d, i0, i1, period) {
  const row = document.getElementById('bench-row');
  row.innerHTML = '';
  ['QQQ', 'SPY', 'DIA'].forEach(key => {
    const el = document.createElement('div'); el.className = 'bench-mini'; el.dataset.ticker = key;
    let pct = null;
    if (period === '1d') {
      const series = (d.today.series || {})[key];
      if (series && series.length) pct = series[series.length - 1];
    } else {
      const series = d.historic.equity_normalized[key];
      if (series && series.length) {
        const windowed = series.slice(i0, i1 + 1);
        pct = windowed.length > 1 ? 100 * (windowed[windowed.length - 1] / windowed[0] - 1) : 0;
      }
    }
    if (pct == null) { el.innerHTML = `<div class="l">${BENCH_LABELS[key]}</div><div class="v">—</div>`; row.appendChild(el); return; }
    el.innerHTML = `<div class="l">${BENCH_LABELS[key]}</div><div class="v ${pct >= 0 ? 'good' : 'bad'}">${fmtPct(pct)}</div>`;
    el.addEventListener('click', () => selectTicker(key));
    row.appendChild(el);
  });
}

function selectTicker(key) {
  renderSelectedChart(GLOBAL_D, key);
}

function renderSelectedChart(d, key) {
  const label = BENCH_LABELS[key] || key;
  document.getElementById('selected-title').textContent = label + ' today';
  const series = (d.today.series || {})[key];
  const timestamps = d.today.timestamps || [];
  const container = document.getElementById('chart-selected');
  if (!series || !series.length) { container.innerHTML = '<div class="empty-note">No intraday data for ' + label + '</div>'; return; }
  const t = timestamps.map(x => new Date(x));
  const firstVal = series.find(v => v != null);
  const lastVal = [...series].reverse().find(v => v != null);
  const up = firstVal != null && lastVal != null ? lastVal >= firstVal : true;
  const color = up ? GREEN : RED;
  const traces = [];
  if (GLOW > 0) {
    traces.push({ x: t, y: series, type: 'scatter', mode: 'lines',
      line: { color, width: 6, shape: 'spline', smoothing: 0.5 }, opacity: 0.16 * GLOW, hoverinfo: 'skip', showlegend: false });
  }
  traces.push({ x: t, y: series, type: 'scatter', mode: 'lines',
    line: { color, width: 2, shape: 'spline', smoothing: 0.5 }, showlegend: false,
    hovertemplate: '%{x|%H:%M}: %{y:.1f}<extra></extra>' });

  // Sits exactly on the line, not at the raw entry_price (usually from a
  // prior day, rarely matching where today's line sits). If open_datetime
  // falls inside today's window, mark the closest real point on the
  // line; otherwise use the line's first point. Only rendered for an
  // open position, never a benchmark. Always blue -- the line is already
  // green/red for up/down, so a colored dot on top would read as
  // ambiguous (long/short, or the current move?). Blue keeps it a plain
  // "here's the entry" marker.
  const pos = (d.open_positions || []).find(p => p.ticker === key);
  if (pos && t.length) {
    let idx = 0;
    const openTime = pos.open_datetime ? new Date(pos.open_datetime).getTime() : null;
    if (openTime != null) {
      let bestDiff = Infinity;
      t.forEach((ts, i) => {
        const diff = Math.abs(ts.getTime() - openTime);
        if (diff < bestDiff) { bestDiff = diff; idx = i; }
      });
    }
    const yAtIdx = series[idx];
    if (yAtIdx != null) {
      traces.push({ x: [t[idx]], y: [yAtIdx], mode: 'markers', type: 'scatter', text: ['Entry'],
        marker: { size: 8, color: BLUE, line: { color: cv('--card', '#1a1b1f'), width: 3 } },
        hovertemplate: 'Entry: %{y:.1f}<extra></extra>', showlegend: false });
    }
  }

  Plotly.newPlot('chart-selected', traces, {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'inherit', color: MUTED, size: 9 },
    margin: { l: 34, r: 10, t: 4, b: 22 },
    xaxis: { gridcolor: GRID }, yaxis: { gridcolor: GRID },
  }, PLOTLY_CONFIG);
}

// --- Below-fold analytics -----------------------------------------------
// No slot in the ported viewport layout above; kept simple, reusing the
// same theme constants. See index.html's .below-fold section.

function belowFoldLayout(extra = {}) {
  return {
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'inherit', color: MUTED, size: 10 },
    margin: { l: 40, r: 14, t: 6, b: 26 },
    xaxis: { gridcolor: GRID }, yaxis: { gridcolor: GRID },
    showlegend: false,
    ...extra,
  };
}

function renderRollingSharpeChart(d) {
  const rs = d.historic.rolling_sharpe_60d;
  if (!rs.dates.length) {
    document.getElementById('chart-rolling-sharpe').innerHTML = '<div class="empty-note">Not enough daily return history yet for a 60-day window.</div>';
    return;
  }
  // A light trailing average on top of the already-60-day-windowed
  // values. Spline shape alone only smooths the curve between points; it
  // doesn't reduce day-to-day jitter in the values themselves. Cranking
  // spline smoothing up further on noisy data risks overshoot -- the
  // curve bulging past the real data range.
  const smoothedValues = movingAverage(rs.values, 5);
  Plotly.newPlot('chart-rolling-sharpe',
    [{ x: rs.dates, y: smoothedValues, type: 'scatter', mode: 'lines',
       line: { color: BLUE, width: 2, shape: 'spline', smoothing: 0.6 } }],
    belowFoldLayout(), PLOTLY_CONFIG);
}

// Trailing simple moving average, window-sized. Used for the smoothed
// overlay line. Partial (shorter) window at the very start of the
// series, instead of leaving it undefined there.
function movingAverage(arr, window) {
  const out = [];
  for (let i = 0; i < arr.length; i++) {
    const start = Math.max(0, i - window + 1);
    let sum = 0;
    for (let j = start; j <= i; j++) sum += arr[j];
    out.push(sum / (i - start + 1));
  }
  return out;
}

function renderUtilizationChart(d) {
  const util = d.historic.capital_utilization_pct;
  // A wide trailing window (16 days) for a genuinely calmer overlay, not
  // just a rendering trick. Utilization only samples at each new
  // admission (see sizing.py) and is naturally spiky day to day.
  const smoothed = movingAverage(util, 16);
  Plotly.newPlot('chart-utilization', [
    { x: d.historic.dates, y: util, type: 'scatter', mode: 'lines',
      line: { color: GREEN, width: 1.5, shape: 'spline', smoothing: 0.75 },
      fill: 'tozeroy', fillcolor: `rgba(${GREEN_RGB},0.12)` },
    { x: d.historic.dates, y: smoothed, type: 'scatter', mode: 'lines',
      line: { color: BLUE, width: 1, shape: 'spline', smoothing: 0.75 } },
  ], belowFoldLayout({ yaxis: { gridcolor: GRID, title: { text: '%', font: { size: 9 } } } }), PLOTLY_CONFIG);
}

function renderHistogram(d) {
  const returns = d.trade_returns_pct;
  const wins = returns.filter(v => v > 0);
  const losses = returns.filter(v => v <= 0);
  Plotly.newPlot('chart-histogram',
    [
      { x: losses, type: 'histogram', name: 'Loss', marker: { color: RED }, xbins: { size: 2 } },
      { x: wins, type: 'histogram', name: 'Win', marker: { color: GREEN }, xbins: { size: 2 } },
    ],
    belowFoldLayout({ barmode: 'overlay', xaxis: { gridcolor: GRID, title: { text: '% return', font: { size: 9 } }, range: [-40, 40] }, bargap: 0.05 }),
    PLOTLY_CONFIG);
}

function renderConcentrationChart(d) {
  const rows = d.ticker_concentration;
  if (!rows.length) {
    document.getElementById('chart-concentration').innerHTML = '<div class="empty-note">No admitted closed trades yet.</div>';
    return;
  }
  Plotly.newPlot('chart-concentration',
    [{
      x: rows.map(r => r.ticker), y: rows.map(r => r.share_of_total_pct),
      type: 'bar', width: 0.4, marker: { color: BLUE },
    }],
    belowFoldLayout({
      bargap: 0.4, margin: { l: 40, r: 14, t: 6, b: 30 },
      yaxis: { gridcolor: GRID, title: { text: '% of total realized P&L', font: { size: 9 } } },
    }),
    PLOTLY_CONFIG);
}

// --- Load -----------------------------------------------------------------

// The published default: fully synthetic demo data (see
// pipeline/generate_sample_feed.py). Real trade history is gitignored and
// never shipped -- point this at your own config's output_dir to run your
// own book instead (see README.md).
const DATA_PATH = 'data-sample/dashboard.json';

async function loadAndRenderAll(path) {
  const res = await fetch(path, { cache: 'no-store' });
  const d = await res.json();
  GLOBAL_D = d;

  const lastDateStr = d.historic.dates[d.historic.dates.length - 1];
  const lastDate = new Date(lastDateStr);
  const updatedTag = document.getElementById('updated-tag');
  if (updatedTag) {
    const monthYear = lastDate.toLocaleDateString(undefined, { month: 'long', year: 'numeric', timeZone: 'UTC' });
    updatedTag.textContent = `Last updated ${monthYear}`;
  }
  const defStart = new Date(lastDate); defStart.setDate(defStart.getDate() - 15);
  customStart = defStart.toISOString().slice(0, 10);
  customEnd = lastDateStr;
  document.getElementById('date-start').value = customStart;
  document.getElementById('date-end').value = customEnd;
  document.getElementById('date-start').max = lastDateStr;
  document.getElementById('date-end').max = lastDateStr;

  renderCalendarTiles(document.getElementById('calendar-grid'), d, 60);
  renderPositionTiles(document.getElementById('positions-grid'), d, 15);
  update(); // also renders the gauges, with the current period's beta
  renderBubbleChart(d);
  renderRollingSharpeChart(d);
  renderUtilizationChart(d);
  renderHistogram(d);
  renderConcentrationChart(d);

  requestAnimationFrame(() => {
    ['chart-main', 'gauge-beta', 'gauge-cash', 'chart-bubble', 'chart-selected',
      'chart-rolling-sharpe', 'chart-utilization', 'chart-histogram', 'chart-concentration'].forEach(id => {
      const el = document.getElementById(id);
      if (el && el.data) Plotly.Plots.resize(el);
    });
  });

  const defaultKey = d.open_positions.length ? d.open_positions[0].ticker : 'QQQ';
  selectTicker(defaultKey);
}

async function main() {
  // The "Sample data" tag is only true for the published default -- any
  // other DATA_PATH (a real book, a different scenario) means we can't
  // claim it's sample data, so drop the tag instead of showing a stale one.
  if (DATA_PATH !== 'data-sample/dashboard.json') {
    document.getElementById('sample-tag')?.remove();
    document.getElementById('sample-tag-sep')?.remove();
  }

  document.querySelectorAll('.periods button[data-p]').forEach(b => b.addEventListener('click', () => {
    document.querySelectorAll('.periods button[data-p]').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    currentPeriod = b.dataset.p;
    document.getElementById('periods').classList.toggle('custom-active', currentPeriod === 'custom');
    update();
  }));
  document.getElementById('date-start').addEventListener('change', e => { customStart = e.target.value; if (currentPeriod === 'custom') update(); });
  document.getElementById('date-end').addEventListener('change', e => { customEnd = e.target.value; if (currentPeriod === 'custom') update(); });
  window.addEventListener('resize', () => { if (GLOBAL_D) update(); });

  await loadAndRenderAll(DATA_PATH);
}

main().catch((err) => {
  document.body.innerHTML =
    `<p style="color:${RED};padding:40px;font-family:var(--font)">Failed to load dashboard data: ${err}. Run <code>python pipeline/build.py</code> first.</p>`;
  console.error(err);
});
