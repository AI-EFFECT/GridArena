/**
 * New Synthetic Dataset (/pf-datagen/ui/new): eight step boxes, as in the
 * digital twin's creation wizard. Each step validates before moving on; the
 * last one reviews the request and starts it. One grid is a single
 * POST /pf-datagen/mv/{grid_id}/run; several start as a group
 * (POST /pf-datagen/group/run). Either way the page then opens the new run
 * or group. "Download standalone package" sends the same request to
 * .../package instead and saves the zip it returns (pf_datagen/package.py).
 */
(function () {

  var TOTAL_STEPS = 8;
  var currentStep = 1;
  var furthestStep = 1;

  function $(id) { return document.getElementById(id); }

  // ── Grid picker ──

  window.pfFilterGridChecklist = function () {
    var q = $('pfGridSearch').value.toLowerCase();
    document.querySelectorAll('#pfGridChecklist .pf-grid-row').forEach(function (row) {
      var id = row.getAttribute('data-grid-id') || '';
      row.style.display = id.indexOf(q) !== -1 ? '' : 'none';
    });
  };

  window.pfSelectAllVisible = function () {
    document.querySelectorAll('#pfGridChecklist .pf-grid-row').forEach(function (row) {
      if (row.style.display !== 'none') {
        var cb = row.querySelector('.pfGridCheckbox');
        if (cb) cb.checked = true;
      }
    });
    pfUpdateSelectedCount();
  };

  window.pfClearSelection = function () {
    document.querySelectorAll('.pfGridCheckbox').forEach(function (cb) { cb.checked = false; });
    pfUpdateSelectedCount();
  };

  window.pfUpdateSelectedCount = function () {
    var n = document.querySelectorAll('.pfGridCheckbox:checked').length;
    $('pfSelectedCount').textContent = n + ' grid' + (n === 1 ? '' : 's') + ' selected' +
      (n > 1 ? ' — they will run as a group' : '');
  };

  function pfSelectedGridIds() {
    return Array.prototype.slice.call(document.querySelectorAll('.pfGridCheckbox:checked'))
      .map(function (cb) { return cb.value; });
  }

  // ── Data source (Historical database) picker ──

  var pfDatabases = [];
  var pfDatabasesLoading = true;
  var pfDatabasesFetchError = null;

  (async function pfLoadHistoricalDatabases() {
    var select = $('pfHistoricalDatabase');
    var info = $('pfHistoricalDatabaseInfo');
    try {
      var res = await fetch('/historical/databases');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      var databases = await res.json();
      pfDatabases = databases;

      // DER step: the generation databases (any type but load) with series.
      pfGenDbs = databases.filter(function (d) { return d.data_type && d.data_type !== 'load' && d.n_series > 0; });

      select.innerHTML = '';
      if (!databases.length) {
        select.innerHTML = '<option value="">No historical databases yet</option>';
        info.textContent = 'Upload one on the Historical page first.';
        return;
      }
      // A load source: one grid level, with series, and load data (not PV or other generation).
      function isGeneration(d) { return d.data_type && d.data_type !== 'load'; }
      function isUsable(d) { return d.grid_level !== 'Mixed' && d.grid_level !== 'Empty' && !isGeneration(d); }
      var usable = databases.filter(isUsable);
      if (!usable.length) {
        info.textContent = 'No usable databases -- every one mixes grid levels, has no series or holds generation data. Upload a homogeneous load database on the Historical page.';
      }
      databases.forEach(function (d) {
        var opt = document.createElement('option');
        opt.value = d.database_id;
        var disabled = !isUsable(d);
        opt.disabled = disabled;
        opt.textContent = d.database_id + (d.name ? ' (' + d.name + ')' : '') + ' -- ' + d.grid_level
          + (isGeneration(d) ? ' [' + d.data_type + ' generation, not a load source]' : (disabled ? ' [not usable]' : ''))
          + ' -- ' + d.n_series + ' series';
        select.appendChild(opt);
      });
      select.addEventListener('change', function () {
        var d = databases.find(function (x) { return x.database_id === select.value; });
        info.textContent = d ? (d.n_series + ' series, ' + d.n_timestamps.toLocaleString('en-GB')
          + (d.n_timestamps === 1 ? ' timestamp' : ' timestamps')
          + (d.first_ts ? ' (' + d.first_ts + ' → ' + d.last_ts + ')' : '')
          + ', ' + d.n_records.toLocaleString('en-GB') + ' records, detected level: ' + d.grid_level) : '';
      });
      if (usable.length) {
        select.value = usable[0].database_id;
        select.dispatchEvent(new Event('change'));
      }
    } catch (err) {
      pfDatabasesFetchError = err.message;
      select.innerHTML = '<option value="">Failed to load databases</option>';
      info.textContent = 'Error: ' + err.message;
    } finally {
      pfDatabasesLoading = false;
      pfInitGroups();
    }
  })();

  // ── Field toggles ──

  window.pfDatagenToggleTopologyFields = function () {
    var type = $('pfTopologyType').value;
    $('pfTopologyFields').style.display = type === 'none' ? 'none' : 'block';
    $('pfNVariantsField').style.display = type === 'random' ? 'flex' : 'none';
  };

  // Load level: show its fields only when on, with the mode's default range as placeholders.
  var PF_LEVEL_DEFAULTS = { multiplier: [0.5, 1.5], loadability: [0.1, 1.2] };
  window.pfLoadLevelModeChanged = function () {
    var mode = $('pfLoadLevelMode').value;
    $('pfLoadLevelFields').style.display = mode === 'off' ? 'none' : 'block';
    var d = PF_LEVEL_DEFAULTS[mode];
    if (d) {
      $('pfLoadLevelMin').placeholder = d[0];
      $('pfLoadLevelMax').placeholder = d[1];
    }
  };

  // ── DER step: plant groups, time matching, preview ──

  var pfGenDbs = [];          // generation databases (any type but load) with series
  var pfGroups = [];          // [{database_id, interpretation, count_min, count_max, capacity_mw, placement, control}]
  var pfPreviewData = null;   // last /pf-datagen/generation-preview answer
  var PF_PLACEMENTS = [['random', 'Random node'], ['load', 'Near load'], ['clustered', 'Clustered'],
                       ['feeder_end', 'Feeder end']];
  var PF_CONTROLS = [['mixed', 'Mixed (type default)'], ['fixed_pf', 'Fixed cos φ'], ['q_v', 'Q(V)'],
                     ['p_v', 'P(V)'], ['voltage', 'Voltage']];
  var PF_TYPE_COLORS = { pv: 'var(--color-solar)', wind: 'var(--color-accent)', chp: 'var(--color-energy)',
                         hydro: '#5b8def', other: '#e07a5f', generation: '#c9a227' };

  function pfEsc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function pfFmtMw(w) { return (w / 1e6).toLocaleString('en-GB', { maximumFractionDigits: 2 }) + ' MW'; }

  function pfGenInfo(dbId) {
    return pfPreviewData && (pfPreviewData.generation || []).find(function (g) { return g.database_id === dbId; });
  }

  // Called once the databases are listed: one PV group (or the first generation database) to start from.
  function pfInitGroups() {
    if (!pfGroups.length && pfGenDbs.length) {
      var pv = pfGenDbs.find(function (d) { return d.data_type === 'pv'; }) || pfGenDbs[0];
      pfGroups.push({ database_id: pv.database_id, interpretation: 'aggregated', count_min: 3, count_max: 8,
                      capacity_mw: 1, placement: 'random', control: 'mixed' });
    }
    pfRenderGroups();
  }

  window.pfAddGroup = function () {
    var last = pfGroups[pfGroups.length - 1];
    pfGroups.push(last ? Object.assign({}, last) : {
      database_id: pfGenDbs.length ? pfGenDbs[0].database_id : '', interpretation: 'aggregated', count_min: 1,
      count_max: 1, capacity_mw: 1, placement: 'random', control: 'mixed' });
    pfRenderGroups();
    pfPreviewSoon();
  };

  function pfOptions(pairs, value) {
    return pairs.map(function (p) {
      return '<option value="' + pfEsc(p[0]) + '"' + (p[0] === value ? ' selected' : '') + '>' + pfEsc(p[1]) + '</option>';
    }).join('');
  }

  function pfRenderGroups() {
    var body = $('pfDerGroups');
    if (!body) return;
    if (!pfGenDbs.length) {
      body.innerHTML = '<tr><td colspan="7" class="muted">' + (pfDatabasesLoading ? 'Loading the databases…'
        : 'No generation databases yet — upload one (PV, wind, …) on the <a href="/historical/ui">Historical</a> page.') + '</td></tr>';
      pfUpdateTotal();
      return;
    }
    var dbPairs = pfGenDbs.map(function (d) { return [d.database_id, d.database_id + ' (' + d.data_type + ')']; });
    body.innerHTML = pfGroups.map(function (g, k) {
      var ind = g.interpretation === 'individual';
      var info = pfGenInfo(g.database_id);
      var fromData = info ? info.n_producing + ' series · ' + pfFmtMw(info.ratings_w_total) : 'ratings from the data';
      return '<tr data-k="' + k + '">' +
        (ind ? '<td colspan="2" class="pf-fromdata">' + pfEsc(fromData) + '</td>'
             : '<td class="pf-count-range"><input class="input" type="number" min="0" max="1000" step="1" data-f="count_min" value="' + pfEsc(g.count_min) + '" aria-label="Fewest plants per grid">' +
               '<span>–</span><input class="input" type="number" min="0" max="1000" step="1" data-f="count_max" value="' + pfEsc(g.count_max) + '" aria-label="Most plants per grid"></td>' +
               '<td class="pf-num"><input class="input" type="number" min="0.001" step="0.1" data-f="capacity_mw" value="' + pfEsc(g.capacity_mw) + '" aria-label="Capacity of each plant (MW)"></td>') +
        '<td><select class="input" data-f="database_id" aria-label="Generation database">' + pfOptions(dbPairs, g.database_id) + '</select></td>' +
        '<td><select class="input" data-f="interpretation" aria-label="Data are">' +
          pfOptions([['aggregated', 'Aggregated'], ['individual', 'Individual']], g.interpretation) + '</select></td>' +
        '<td><select class="input" data-f="placement" aria-label="Placement">' + pfOptions(PF_PLACEMENTS, g.placement) + '</select></td>' +
        '<td><select class="input" data-f="control" aria-label="Control">' + pfOptions(PF_CONTROLS, g.control) + '</select></td>' +
        '<td><button type="button" class="btn pf-remove" data-remove="' + k + '" title="Remove this group" aria-label="Remove this group"' +
          (pfGroups.length === 1 ? ' disabled' : '') + '>✕</button></td></tr>';
    }).join('');
    pfUpdateTotal();
  }

  (function bindGroupTable() {
    var body = $('pfDerGroups');
    if (!body) return;
    body.addEventListener('input', function (ev) {
      var f = ev.target.getAttribute('data-f');
      var row = ev.target.closest('tr');
      if (!f || !row) return;
      var g = pfGroups[parseInt(row.getAttribute('data-k'), 10)];
      if (f === 'count_min' || f === 'count_max') g[f] = parseInt(ev.target.value, 10);
      else if (f === 'capacity_mw') g.capacity_mw = parseFloat(ev.target.value);
      else return;
      pfUpdateTotal();
    });
    body.addEventListener('change', function (ev) {
      var f = ev.target.getAttribute('data-f');
      var row = ev.target.closest('tr');
      if (!f || !row || f === 'count_min' || f === 'count_max' || f === 'capacity_mw') return;
      pfGroups[parseInt(row.getAttribute('data-k'), 10)][f] = ev.target.value;
      pfRenderGroups();
      if (f === 'database_id') pfPreviewSoon();
    });
    body.addEventListener('click', function (ev) {
      var btn = ev.target.closest('[data-remove]');
      if (!btn || pfGroups.length < 2) return;
      pfGroups.splice(parseInt(btn.getAttribute('data-remove'), 10), 1);
      pfRenderGroups();
      pfPreviewSoon();
    });
  })();

  function pfUpdateTotal() {
    var el = $('pfDerTotal');
    if (!el) return;
    var w = [0, 0], n = [0, 0], unknown = false;
    pfGroups.forEach(function (g) {
      if (g.interpretation === 'individual') {
        var info = pfGenInfo(g.database_id);
        if (info) { [0, 1].forEach(function (k) { w[k] += info.ratings_w_total; n[k] += info.n_producing; }); }
        else unknown = true;
      } else if (g.count_min >= 0 && g.count_max >= g.count_min && g.capacity_mw > 0) {
        n[0] += g.count_min; n[1] += g.count_max;
        w[0] += g.count_min * g.capacity_mw * 1e6; w[1] += g.count_max * g.capacity_mw * 1e6;
      }
    });
    var span = function (a, b, f) { return a === b ? f(a) : f(a) + ' – ' + f(b); };
    el.innerHTML = 'Per grid: ' + span(n[0], n[1], String) + ' plant' + (n[1] === 1 ? '' : 's') + ', <strong>' +
      span(w[0], w[1], pfFmtMw) + '</strong> installed' + (unknown ? ' (+ individual plants, sizes pending)' : '') +
      (n[0] !== n[1] ? '; each grid draws its own number' : '');
  }

  // DER: which parts apply to the source chosen.
  window.pfDerToggle = function () {
    var src = $('pfDerSource').value;
    $('pfDerFields').style.display = src !== 'grid' ? 'block' : 'none';
    $('pfGridDerCard').style.display = src === 'replace' ? 'none' : '';
    if (src !== 'grid') pfPreviewSoon();
  };

  // Placement follows the timeline: a consecutive window keeps the plants where they are.
  function pfLayoutFollowTimeline() {
    var sel = $('pfDerLayout');
    var windowed = $('pfTimeline').value === 'window';
    if (windowed) sel.value = 'per_run';
    sel.disabled = windowed;
    $('pfDerLayoutNote').textContent = windowed ? 'Fixed: the load timeline is a consecutive window (step 2).' : '';
  }

  function pfSummaries() {
    $('pfVarSummary').textContent = '— performance ' + range(pfNum('pfDerHistPerfMin', 0.9), pfNum('pfDerHistPerfMax', 1)) +
      ' · ρ ' + pfNum('pfDerHistRho', 0.85) + ' · σ ' + pfNum('pfDerHistSigma', 0.08) +
      ' · reference year ' + ($('pfDerHistRefYear').value || 'auto');
    $('pfCtrlSummary').textContent = '— cos φ ' + range(pfNum('pfDerCosMin', 0.95), pfNum('pfDerCosMax', 1)) +
      ' · V set ' + range(pfNum('pfDerVsetMin', 0.99), pfNum('pfDerVsetMax', 1.03)) + ' p.u.';
  }
  ['pfDerHistPerfMin', 'pfDerHistPerfMax', 'pfDerHistRho', 'pfDerHistSigma', 'pfDerHistRefYear',
   'pfDerCosMin', 'pfDerCosMax', 'pfDerVsetMin', 'pfDerVsetMax'].forEach(function (id) {
    $(id).addEventListener('input', pfSummaries);
  });
  pfSummaries();

  // Generation preview: the load's first days, and each database over the window a run would read.
  var pfPreviewTimer = null, pfPreviewSeq = 0;
  window.pfPreviewSoon = function () {
    clearTimeout(pfPreviewTimer);
    pfPreviewTimer = setTimeout(pfLoadPreview, 350);
  };

  async function pfLoadPreview() {
    if ($('pfDerSource').value === 'grid' || currentStep !== 3) return;
    var ids = pfGroups.map(function (g) { return g.database_id; }).filter(function (d, i, a) { return d && a.indexOf(d) === i; });
    var info = $('pfGenInfo');
    if (!ids.length) { info.textContent = ''; $('pfGenChart').innerHTML = ''; return; }
    var q = new URLSearchParams({ database_ids: ids.join(','), keep_time_of_day: $('pfDerHistKeepTod').checked,
                                  timezone: $('pfDerHistTz').value.trim() || 'UTC', days: 3 });
    if ($('pfHistoricalDatabase').value) q.set('load_database_id', $('pfHistoricalDatabase').value);
    if ($('pfTimeline').value === 'window' && $('pfTimelineStart').value) q.set('load_start', $('pfTimelineStart').value);
    if ($('pfDerHistStart').value) q.set('generation_start', $('pfDerHistStart').value);
    if ($('pfLoadAggregation').value) q.set('load_aggregation_s', $('pfLoadAggregation').value);
    var seq = ++pfPreviewSeq;
    info.textContent = 'Loading the preview…';
    try {
      var res = await fetch('/pf-datagen/generation-preview?' + q.toString());
      var data = await res.json();
      if (seq !== pfPreviewSeq) return;
      if (!res.ok) throw new Error(detailText(data.detail));
      pfPreviewData = data;
      pfRenderPreview(data);
      pfRenderGroups();
    } catch (err) {
      if (seq !== pfPreviewSeq) return;
      info.innerHTML = '<span class="pf-warn">Preview unavailable: ' + pfEsc(err.message) + '</span>';
      $('pfGenChart').innerHTML = '';
    }
  }

  function pfStepLabel(s) { return s >= 3600 ? (s / 3600) + ' h' : s >= 60 ? (s / 60) + ' min' : s + ' s'; }
  function pfMs(iso) { return Date.parse(iso + 'Z'); }   // naive clock times, compared as such

  function pfRenderPreview(d) {
    var shift = d.shift_days;
    var shiftText = shift === 0 ? 'same instants as the load' : (shift > 0 ? '+' : '') +
      (Math.round(shift * 100) / 100) + ' days from the load';
    var lines = ['Load window from <strong>' + pfEsc(d.load_start.replace('T', ' ').slice(0, 16)) + '</strong>' +
      ($('pfTimeline').value === 'window' ? '' : ' (spread samples: the earliest one)') + '; generation read at ' + shiftText + '.'];
    d.generation.forEach(function (g) {
      lines.push('<strong>' + pfEsc(g.database_id) + '</strong> · ' + pfEsc(g.data_type) + ' · ' + pfStepLabel(g.step_s) +
        ' · ' + pfEsc(g.first.slice(0, 10)) + ' → ' + pfEsc(g.last.slice(0, 10)) + ' · clock ' +
        pfEsc($('pfDerHistTz').value.trim() || 'UTC') +
        (g.times.length ? '' : ' · <span class="pf-warn">no data in this window: a run borrows the nearest day with data (up to 31 days), or fails</span>'));
    });
    // Which series gets interpolated, and how to avoid it.
    var agg = parseInt($('pfLoadAggregation').value, 10) || null;
    var loadStep = d.load && d.load.step_s;
    var genSteps = d.generation.map(function (g) { return g.step_s; });
    if (agg) {
      var finer = genSteps.some(function (s) { return s < 0.99 * agg; });
      var coarser = genSteps.some(function (s) { return s > 1.01 * agg; });
      lines.push('Load averaged over ' + pfStepLabel(agg) + ' intervals (step 2)' +
        (finer ? '; finer generation averaged over the same intervals' : '') +
        (coarser ? '; <span class="pf-warn">generation coarser than ' + pfStepLabel(agg) + ' still interpolated</span>' : '') + '.');
    } else if (loadStep && genSteps.some(function (s) { return s > 1.01 * loadStep; })) {
      lines.push('The load steps at ' + pfStepLabel(loadStep) + ': generation with a coarser step is interpolated ' +
        'linearly between its points. To average the load instead, set <em>Load resolution</em> in step 2 to ' +
        pfStepLabel(Math.max.apply(null, genSteps)) + '.');
    }
    $('pfGenInfo').innerHTML = lines.join('<br>');
    pfDrawPreview(d);
  }

  function pfSvg(tag, attrs) {
    var el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }

  function pfPath(xs, ys, x, y) {
    var out = '';
    for (var k = 0; k < xs.length; k++) out += (k ? 'L' : 'M') + x(xs[k]).toFixed(1) + ',' + y(ys[k]).toFixed(1);
    return out;
  }

  function pfDrawPreview(d) {
    var host = $('pfGenChart');
    host.innerHTML = '';
    var W = Math.max(320, host.clientWidth || 640), H = 190, L = 34, R = 10, T = 8, B = 24;
    var x0 = pfMs(d.load_start), x1 = x0 + d.days * 864e5;
    // A net load can export (negative): the axis reaches down to it.
    var vmin = Math.min(0, Math.min.apply(null, [0].concat((d.load ? d.load.values : []),
      d.generation.reduce(function (a, g) { return a.concat(g.values); }, []))));
    var x = function (t) { return L + (t - x0) / (x1 - x0) * (W - L - R); };
    var y = function (v) { return T + (1 - v) / (1 - vmin) * (H - T - B); };
    var svg = pfSvg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img',
                             'aria-label': 'Load and generation over the first days, normalised' });
    (vmin < -0.05 ? [Math.ceil(vmin * 2) / 2, 0, 0.5, 1] : [0, 0.5, 1]).forEach(function (v) {
      svg.appendChild(pfSvg('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'pf-chart-grid' }));
      var t = pfSvg('text', { x: L - 6, y: y(v) + 3, 'text-anchor': 'end', class: 'pf-chart-axis' });
      t.textContent = v.toFixed(1);
      svg.appendChild(t);
    });
    for (var day = Math.ceil(x0 / 864e5) * 864e5; day <= x1; day += 864e5) {
      svg.appendChild(pfSvg('line', { x1: x(day), x2: x(day), y1: T, y2: H - B, class: 'pf-chart-grid' }));
      if (x1 - day < 0.3 * 864e5) continue;          // no room for the last label
      var lab = pfSvg('text', { x: x(day) + 3, y: H - 8, class: 'pf-chart-axis' });
      lab.textContent = new Date(day).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' });
      svg.appendChild(lab);
    }
    var series = [];
    if (d.load && d.load.times.length) series.push({ name: 'Load (' + d.load.database_id + ')', cls: 'pf-chart-load',
      color: 'var(--color-text-muted)', t: d.load.times.map(pfMs), v: d.load.values });
    d.generation.forEach(function (g, k) {
      if (!g.times.length) return;
      series.push({ name: g.database_id, cls: 'pf-chart-line', color: PF_TYPE_COLORS[g.der_type] || 'var(--color-accent)',
                    dash: k > 0 && d.generation.slice(0, k).some(function (o) { return o.der_type === g.der_type; }),
                    t: g.times.map(pfMs), v: g.values });
    });
    series.forEach(function (s) {
      var p = pfSvg('path', { d: pfPath(s.t, s.v, x, y), class: s.cls, stroke: s.color });
      if (s.dash) p.setAttribute('stroke-dasharray', '5 3');
      svg.appendChild(p);
    });
    var cross = pfSvg('line', { x1: 0, x2: 0, y1: T, y2: H - B, class: 'pf-chart-cross', visibility: 'hidden' });
    svg.appendChild(cross);
    var hit = pfSvg('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
    svg.appendChild(hit);
    var readout = $('pfGenReadout');
    readout.textContent = '';
    hit.addEventListener('mousemove', function (ev) {
      var box = svg.getBoundingClientRect();
      var px = (ev.clientX - box.left) / box.width * W;
      var t = x0 + (px - L) / (W - L - R) * (x1 - x0);
      cross.setAttribute('x1', px); cross.setAttribute('x2', px); cross.setAttribute('visibility', 'visible');
      var parts = [new Date(t).toISOString().slice(0, 16).replace('T', ' ')];
      series.forEach(function (s) {
        var k = 0, best = Infinity;
        for (var j = 0; j < s.t.length; j++) { var dd = Math.abs(s.t[j] - t); if (dd < best) { best = dd; k = j; } }
        parts.push(s.name + ' ' + (best < 36e5 ? s.v[k].toFixed(2) : '—'));
      });
      readout.textContent = parts.join(' · ');
    });
    hit.addEventListener('mouseleave', function () { cross.setAttribute('visibility', 'hidden'); readout.textContent = ''; });
    host.appendChild(svg);
    var legend = document.createElement('div');
    legend.className = 'pf-legend';
    legend.innerHTML = series.map(function (s) {
      return '<span><i style="background:' + s.color + '"></i>' + pfEsc(s.name) + '</span>';
    }).join('') + '<span class="muted">each normalised by its maximum in the window; local clock of the load</span>';
    host.appendChild(legend);
  }

  // ── Grid devices: the DER card and the Set-points counts ──

  var pfDevicesKey = null;
  async function pfLoadDevices() {
    var ids = pfSelectedGridIds();
    var key = ids.join(',');
    if (!ids.length || key === pfDevicesKey) return;
    pfDevicesKey = key;
    var list = $('pfGridDerList');
    list.innerHTML = '<li class="muted">Loading…</li>';
    try {
      var res = await fetch('/pf-datagen/devices?grids=' + encodeURIComponent(key));
      var data = await res.json();
      if (!res.ok) throw new Error(detailText(data.detail));
      if (key !== pfDevicesKey) return;
      pfShowDevices(data);
    } catch (err) {
      pfDevicesKey = null;
      list.innerHTML = '<li class="muted">Could not read the grids: ' + pfEsc(err.message) + '</li>';
    }
  }

  function pfShowDevices(data) {
    var grids = data.grids.filter(function (g) { return !g.error; });
    var more = data.n_requested > data.n_summarised ? ' (first ' + data.n_summarised + ' of ' + data.n_requested + ' grids)' : '';
    var typeText = function (types) {
      return Object.keys(types).sort().map(function (t) { return (types[t] > 1 ? types[t] + ' ' : '') + t.toUpperCase(); }).join(', ');
    };
    var rows;
    if (grids.length <= 6) {
      rows = grids.map(function (g) {
        return '<li><strong>' + pfEsc(g.mv_grid_id) + '</strong> — ' + (g.ders.count
          ? g.ders.count + ' DER · ' + g.ders.mw.toFixed(2) + ' MW · ' + pfEsc(typeText(g.ders.types)) : 'none') + '</li>';
      });
    } else {
      var types = {}, n = 0, mw = 0, without = 0;
      grids.forEach(function (g) {
        n += g.ders.count; mw += g.ders.mw;
        if (!g.ders.count) without += 1;
        Object.keys(g.ders.types).forEach(function (t) { types[t] = (types[t] || 0) + g.ders.types[t]; });
      });
      rows = ['<li>' + grids.length + ' grids: ' + n + ' DER · ' + mw.toFixed(2) + ' MW · ' + pfEsc(typeText(types)) +
              (without ? '; ' + without + ' grid' + (without === 1 ? '' : 's') + ' without any' : '') + '</li>'];
    }
    data.grids.filter(function (g) { return g.error; }).forEach(function (g) {
      rows.push('<li class="muted">' + pfEsc(g.mv_grid_id) + ': ' + pfEsc(g.error) + '</li>');
    });
    $('pfGridDerList').innerHTML = rows.join('') + (more ? '<li class="muted">' + pfEsc(more) + '</li>' : '');
    ['substations', 'capacitor_banks', 'regulators', 'transformers'].forEach(function (k) {
      var total = grids.reduce(function (s, g) { return s + g[k]; }, 0);
      var card = document.querySelector('.pf-card[data-device="' + k + '"]');
      var count = card.querySelector('[data-count]');
      count.textContent = total ? total.toLocaleString('en-GB') + (grids.length > 1 ? ' in ' + grids.length + ' grids' : '') + more
                                : 'none in the selected grids';
      card.classList.toggle('pf-card--none', total === 0);
    });
  }

  // ── Set-points step ──

  window.pfShuntChanged = function () {
    $('pfShuntBandField').style.display = $('pfShuntSteps').value === 'voltage' ? 'flex' : 'none';
  };

  // The busbar set-point against the load level (0 = no load, 1 = P_max).
  window.pfDrawSlack = function () {
    var host = $('pfSlackChart');
    if (!host) return;
    var mode = $('pfSlackMode').value;
    var lo, hi, line;
    if (mode === 'uniform') { lo = pfNum('pfSlackMin', 0.98); hi = pfNum('pfSlackMax', 1.04); }
    else if (mode === 'load_dependent') {
      var v0 = pfNum('pfSlackV0', 1.0), v1 = pfNum('pfSlackV1', 1.04), nz = pfNum('pfSlackNoise', 0.01);
      line = [v0, v1]; lo = Math.min(v0, v1) - nz; hi = Math.max(v0, v1) + nz;
    } else { line = [1.0, 1.0]; lo = hi = 1.0; }
    var ymin = Math.floor((Math.min(lo, 1.0) - 0.005) * 100) / 100, ymax = Math.ceil((Math.max(hi, 1.0) + 0.005) * 100) / 100;
    var W = Math.max(280, host.clientWidth || 320), H = 120, L = 40, R = 10, T = 6, B = 22;
    var x = function (lv) { return L + lv * (W - L - R); };
    var y = function (v) { return T + (ymax - v) / (ymax - ymin) * (H - T - B); };
    var svg = pfSvg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Busbar set-point against load level' });
    [ymin, 1.0, ymax].filter(function (v, k, a) { return a.indexOf(v) === k; }).forEach(function (v) {
      svg.appendChild(pfSvg('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), class: 'pf-chart-grid' }));
      var t = pfSvg('text', { x: L - 6, y: y(v) + 3, 'text-anchor': 'end', class: 'pf-chart-axis' });
      t.textContent = v.toFixed(2);
      svg.appendChild(t);
    });
    [[0, 'no load'], [1, 'P_max']].forEach(function (a) {
      var t = pfSvg('text', { x: x(a[0]), y: H - 6, 'text-anchor': a[0] ? 'end' : 'start', class: 'pf-chart-axis' });
      t.textContent = a[1];
      svg.appendChild(t);
    });
    var color = 'var(--color-accent)';
    if (mode === 'uniform') {
      svg.appendChild(pfSvg('rect', { x: x(0), y: y(hi), width: x(1) - x(0), height: Math.max(1, y(lo) - y(hi)),
                                      class: 'pf-chart-band', fill: color }));
    }
    if (mode === 'load_dependent' && nz > 0) {
      svg.appendChild(pfSvg('path', { d: 'M' + x(0) + ',' + y(line[0] + nz) + 'L' + x(1) + ',' + y(line[1] + nz) +
        'L' + x(1) + ',' + y(line[1] - nz) + 'L' + x(0) + ',' + y(line[0] - nz) + 'Z', class: 'pf-chart-band', fill: color }));
    }
    if (line) svg.appendChild(pfSvg('path', { d: pfPath([0, 1], line, x, y), class: 'pf-chart-line', stroke: color }));
    host.innerHTML = '';
    host.appendChild(svg);
  };

  // The usual utilisation differs: ~0.4 of the average, ~0.8 of the peak. Swap the default only.
  window.pfSizeByChanged = function () {
    var u = $('pfUtilisation'), peak = $('pfSizeBy').value === 'peak';
    if (peak && parseFloat(u.value) === 0.4) u.value = '0.8';
    if (!peak && parseFloat(u.value) === 0.8) u.value = '0.4';
  };

  window.pfTimelineChanged = function () {
    $('pfTimelineStart').style.display = $('pfTimeline').value === 'window' ? '' : 'none';
    pfLayoutFollowTimeline();
  };

  window.pfSlackModeChanged = function () {
    var mode = $('pfSlackMode').value;
    $('pfSlackRangeField').style.display = mode === 'uniform' ? 'flex' : 'none';
    $('pfSlackLoadField').style.display = mode === 'load_dependent' ? '' : 'none';
    pfDrawSlack();
  };

  // Elements with data-show-when="<checkbox id>" show only while that box is ticked.
  function bindCheckboxToggles() {
    var targets = document.querySelectorAll('[data-show-when]');
    var boxes = {};
    targets.forEach(function (el) { boxes[el.getAttribute('data-show-when')] = true; });
    Object.keys(boxes).forEach(function (id) {
      var box = $(id);
      var apply = function () {
        document.querySelectorAll('[data-show-when="' + id + '"]').forEach(function (el) {
          el.style.display = box.checked ? '' : 'none';
        });
      };
      box.addEventListener('change', apply);
      apply();
    });
  }
  bindCheckboxToggles();

  // ── Request ──

  function pfNum(id, fallback) {
    var v = parseFloat($(id).value);
    return isNaN(v) ? fallback : v;
  }

  function pfNumberOrNull(id) {
    var v = $(id).value;
    return v === '' ? null : parseFloat(v);
  }

  function pfInt(id, fallback) {
    var v = parseInt($(id).value, 10);
    return isNaN(v) ? fallback : v;
  }

  // Extra cable types for swaps (axis 5): a JSON list, or [] when no file is chosen.
  async function pfReadCatalogue() {
    var input = $('pfParCatalogue');
    if (!input || !input.files || !input.files.length) return [];
    var data = JSON.parse(await input.files[0].text());
    if (!Array.isArray(data)) throw new Error('Extra cable types must be a JSON list.');
    return data;
  }

  async function buildPayload() {
    var seedVal = $('pfSeed').value;
    var payload = {
      historical_database_id: $('pfHistoricalDatabase').value,
      reassignment_period_timesteps: pfInt('pfReassignmentPeriod', 100) || 100,
      scenario_count: pfInt('pfScenarioCount', 1000) || 1000,
      timeline: $('pfTimeline').value,
      timeline_start: $('pfTimeline').value === 'window' && $('pfTimelineStart').value ? $('pfTimelineStart').value : null,
      load_aggregation_s: parseInt($('pfLoadAggregation').value, 10) || null,
      load_noise_sigma: pfNum('pfLoadNoiseSigma', 0),
      load_level: {
        mode: $('pfLoadLevelMode').value,
        min: pfNumberOrNull('pfLoadLevelMin'),
        max: pfNumberOrNull('pfLoadLevelMax'),
        sampling: $('pfLoadLevelSampling').value,
        bins: pfInt('pfLoadLevelBins', 10) || 10
      },
      load_shape: {
        point_sizing: $('pfPointSizing').value,
        utilisation: pfNum('pfUtilisation', 0.4) || 0.4,
        size_by: $('pfSizeBy').value,
        feeders_per_point: $('pfFeedersPerPoint').value,
        reactive: $('pfReactive').value,
        pf_min: pfNum('pfPfMin', 0.9) || 0.9,
        pf_max: pfNum('pfPfMax', 0.99) || 0.99,
        spatial_sigma: pfNum('pfSpatialSigma', 0),
        correlation_km: pfNum('pfCorrelationKm', 2) || 2,
        q_sigma: pfNum('pfQSigma', 0)
      },
      der: {
        source: $('pfDerSource').value,
        groups: $('pfDerSource').value !== 'grid' ? pfGroups.map(function (g) {
          var ind = g.interpretation === 'individual';
          return { database_id: g.database_id, interpretation: g.interpretation,
                   count_min: ind ? 1 : g.count_min, count_max: ind ? 1 : g.count_max,
                   capacity_mw: ind ? null : g.capacity_mw,
                   placement: g.placement, control: g.control };
        }) : [],
        generation: {
          generation_start: $('pfDerHistStart').value || null,
          keep_time_of_day: $('pfDerHistKeepTod').checked,
          timezone: $('pfDerHistTz').value.trim() || 'UTC',
          reference_year: pfInt('pfDerHistRefYear', 0) || null,
          performance_min: pfNum('pfDerHistPerfMin', 0.9),
          performance_max: pfNum('pfDerHistPerfMax', 1.0),
          rho: pfNum('pfDerHistRho', 0.85),
          sigma: pfNum('pfDerHistSigma', 0.08)
        },
        cos_phi_min: pfNum('pfDerCosMin', 0.95),
        cos_phi_max: pfNum('pfDerCosMax', 1.0),
        v_set_min: pfNum('pfDerVsetMin', 0.99),
        v_set_max: pfNum('pfDerVsetMax', 1.03),
        resample_grid_settings: $('pfDerResample').checked,
        layout: $('pfTimeline').value === 'window' ? 'per_run' : $('pfDerLayout').value
      },
      setpoints: {
        slack_mode: $('pfSlackMode').value || null,
        slack_v0: pfNum('pfSlackV0', 1.0),
        slack_v1: pfNum('pfSlackV1', 1.04),
        slack_noise: pfNum('pfSlackNoise', 0.01),
        per_substation: $('pfSlackPerSub').checked,
        shunt_steps: $('pfShuntSteps').value,
        shunt_band_min: pfNum('pfShuntBandMin', 0.97),
        shunt_band_max: pfNum('pfShuntBandMax', 1.03),
        regulators_resample: $('pfRegResample').checked,
        reg_v_set_min: pfNum('pfRegVMin', 0.98),
        reg_v_set_max: pfNum('pfRegVMax', 1.03),
        reg_deadband_min: pfNum('pfRegDbMin', 0.015),
        reg_deadband_max: pfNum('pfRegDbMax', 0.025),
        transformer_taps_resample: $('pfTapResample').checked,
        transformer_tap_steps: $('pfTapResample').checked && $('pfTapSteps').value !== '' ? pfInt('pfTapSteps', 0) : null
      },
      topology_perturbation: {
        type: $('pfTopologyType').value,
        k: pfInt('pfTopologyK', 1) || 1,
        n_variants: pfInt('pfNVariants', 10) || 10
      },
      reconfiguration: {
        enabled: $('pfReconfEnabled').checked,
        n_configs: pfInt('pfReconfN', 10) || 10,
        switchable: $('pfReconfSwitchable').value,
        allow_meshed: $('pfReconfMeshed').checked,
        extra_closed: pfInt('pfReconfExtra', 1) || 1
      },
      outages: {
        transformers: $('pfOutTransformers').checked,
        der_trip_p: pfNum('pfOutDer', 0),
        shunt_out_p: pfNum('pfOutShunt', 0),
        regulator_bypass_p: pfNum('pfOutReg', 0)
      },
      variants_per_sample: pfInt('pfVarPerSample', 0) || 'all',
      solver: $('pfSolver').value,
      admittance_perturbation: {
        enabled: $('pfAdmittanceEnabled').checked,
        sigma: pfNum('pfAdmittanceSigma', 0.2) || 0.2
      },
      design: { mode: $('pfDesignMode').value },
      measurements: {
        enabled: $('pfMeasEnabled').checked,
        cp_coverage: pfNum('pfMeasCp', 0.3),
        cable_coverage: pfNum('pfMeasCable', 0.1),
        der_min_mva: pfNum('pfMeasDer', 1),
        profile: $('pfMeasProfile').value,
        substation_profile: $('pfMeasSubProfile').value,
        missing_p: pfNum('pfMeasMissing', 0.02),
        pseudo_sigma: pfNum('pfMeasPseudo', 0.3)
      },
      target_mix: {
        min_violation_share: $('pfTargetEnabled').checked ? pfNum('pfTargetShare', 0.15) : null,
        violations: $('pfTargetViolations').value,
        pilot_share: pfNum('pfTargetPilot', 0.2)
      },
      parameters: {
        length_sigma: $('pfParLength').checked ? pfNum('pfParLengthSigma', 0.05) : 0,
        temperature: $('pfParTemp').checked
          ? { min: pfNum('pfParTempMin', 20), max: pfNum('pfParTempMax', 70), spread: pfNum('pfParTempSpread', 5) }
          : null,
        swap_p: $('pfParSwap').checked ? pfNum('pfParSwapP', 0.1) : 0,
        transformer_uk_sigma: $('pfParTrafo').checked ? pfNum('pfParUk', 0.1) : 0,
        transformer_xr_sigma: $('pfParTrafo').checked ? pfNum('pfParXr', 0.2) : 0,
        draw: $('pfParDraw').value,
        catalogue: await pfReadCatalogue()
      },
      chunk_commit_size: pfInt('pfChunkCommitSize', 100) || 100,
      seed: seedVal ? parseInt(seedVal, 10) : null
    };
    // A drawn busbar set-point needs its range.
    if (payload.setpoints.slack_mode === 'uniform') {
      payload.slack_voltage_pu = { min: pfNum('pfSlackMin', 0.98), max: pfNum('pfSlackMax', 1.04) };
    }
    return payload;
  }

  // ── Validation: problems per step ──

  function problemsOf(step) {
    var out = [];
    if (step === 1 && pfSelectedGridIds().length === 0) {
      out.push('Select at least one MV grid.');
    }
    if (step === 2) {
      var dbSelect = $('pfHistoricalDatabase');
      if (pfDatabasesLoading) {
        out.push('Still loading the historical databases -- wait a moment.');
      } else if (pfDatabasesFetchError) {
        out.push('Could not load the historical databases (' + pfDatabasesFetchError + ') -- reload the page.');
      } else if (!pfDatabases.length) {
        out.push('No historical databases exist yet -- upload one on the Historical page first.');
      } else if (!dbSelect.value || !dbSelect.options[dbSelect.selectedIndex] ||
                 dbSelect.options[dbSelect.selectedIndex].disabled) {
        out.push('Select a usable historical database (load data, one grid level, with series).');
      }
      if (!(pfInt('pfScenarioCount', 0) >= 1)) out.push('Total timesteps must be at least 1.');
      var lo = pfNumberOrNull('pfLoadLevelMin'), hi = pfNumberOrNull('pfLoadLevelMax');
      if ($('pfLoadLevelMode').value !== 'off' && lo !== null && hi !== null && lo > hi) {
        out.push('The load level range is reversed (min above max).');
      }
      if (pfNum('pfPfMin', 0.9) > pfNum('pfPfMax', 0.99)) out.push('The power factor range is reversed.');
    }
    if (step === 3 && $('pfDerSource').value !== 'grid') {
      if (!pfGenDbs.length) out.push('Generated DER follow a generation database: upload one (PV, wind, ...) on the Historical page.');
      pfGroups.forEach(function (g, k) {
        var name = 'Group ' + (k + 1) + ': ';
        if (!g.database_id) out.push(name + 'choose a generation database.');
        if (g.interpretation === 'aggregated') {
          var whole = function (v) { return v >= 0 && v <= 1000 && Math.floor(v) === v; };
          if (!whole(g.count_min) || !whole(g.count_max)) out.push(name + 'the numbers of plants must be whole numbers from 0 to 1000.');
          else if (g.count_min > g.count_max) out.push(name + 'the fewest plants must not exceed the most.');
          else if (g.count_max === 0) out.push(name + 'the group never has a plant (most = 0).');
          if (!(g.capacity_mw > 0)) out.push(name + 'each plant needs a capacity above 0 MW.');
        }
      });
      if (pfNum('pfDerHistPerfMin', 0.9) > pfNum('pfDerHistPerfMax', 1)) out.push('The performance factor range is reversed.');
      if (pfNum('pfDerCosMin', 0.95) > pfNum('pfDerCosMax', 1)) out.push('The cos φ range is reversed.');
      if (pfNum('pfDerVsetMin', 0.99) > pfNum('pfDerVsetMax', 1.03)) out.push('The voltage set-point range is reversed.');
    }
    if (step === 4) {
      if ($('pfSlackMode').value === 'uniform' && pfNum('pfSlackMin', 0.98) >= pfNum('pfSlackMax', 1.04)) {
        out.push('The busbar voltage range is reversed.');
      }
      if ($('pfShuntSteps').value === 'voltage' && pfNum('pfShuntBandMin', 0.97) >= pfNum('pfShuntBandMax', 1.03)) {
        out.push('The capacitor band is reversed.');
      }
      if ($('pfRegResample').checked) {
        if (pfNum('pfRegVMin', 0.98) >= pfNum('pfRegVMax', 1.03)) out.push('The regulator set-point range is reversed.');
        if (pfNum('pfRegDbMin', 0.015) >= pfNum('pfRegDbMax', 0.025)) out.push('The regulator deadband range is reversed.');
      }
    }
    if (step === 5 && $('pfReconfEnabled').checked && $('pfReconfMeshed').checked && $('pfSolver').value !== 'newton') {
      out.push('Meshed reconfiguration closes loops -- choose the Newton-Raphson solver.');
    }
    if (step === 6 && $('pfTargetEnabled').checked && $('pfLoadLevelMode').value === 'off') {
      out.push('A target mix needs a load level (step 2).');
    }
    return out;
  }

  function showStepProblems(step) {
    var problems = problemsOf(step);
    var el = $('stepMsg' + step);
    if (el) el.textContent = problems.join(' ');
    return problems.length === 0;
  }

  // ── Step navigation ──

  window.goStep = function (n) {
    if (n < 1 || n > TOTAL_STEPS || n === currentStep) return;
    if (n > currentStep) {
      for (var s = currentStep; s < n; s++) {
        if (!showStepProblems(s)) {
          if (s !== currentStep) showStep(s);
          return;
        }
      }
    }
    showStep(n);
  };

  function showStep(n) {
    $('step' + currentStep).style.display = 'none';
    $('step' + n).style.display = '';
    currentStep = n;
    furthestStep = Math.max(furthestStep, n);
    updateProgress();
    if (n === 3 || n === 4) pfLoadDevices();
    if (n === 3) { pfLayoutFollowTimeline(); pfPreviewSoon(); }
    if (n === 4) { pfShuntChanged(); pfDrawSlack(); }
    if (n === TOTAL_STEPS) updateReview();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function updateProgress() {
    document.querySelectorAll('#progressBar .dt-progress__step').forEach(function (el) {
      var s = parseInt(el.getAttribute('data-step'), 10);
      el.classList.remove('dt-progress__step--active', 'dt-progress__step--done', 'pf-reachable');
      if (s === currentStep) el.classList.add('dt-progress__step--active');
      else if (s < currentStep) el.classList.add('dt-progress__step--done');
      if (s <= furthestStep && s !== currentStep) el.classList.add('pf-reachable');
    });
  }

  document.querySelectorAll('#progressBar .dt-progress__step').forEach(function (el) {
    el.addEventListener('click', function () {
      var s = parseInt(el.getAttribute('data-step'), 10);
      if (s <= furthestStep) goStep(s);
    });
  });

  // ── Review ──

  function range(a, b) { return a === b ? String(a) : a + '–' + b; }

  function summarise(p, gridIds) {
    var rows = [];
    var add = function (k, v) { rows.push([k, v]); };
    add('Grids', gridIds.length > 3 ? gridIds.length + ' grids (a group): ' + gridIds.slice(0, 3).join(', ') + ', …'
                                    : gridIds.join(', ') + (gridIds.length > 1 ? ' (a group)' : ''));
    add('Data source', p.historical_database_id || '—');
    add('Samples', p.scenario_count + ' timesteps, ' + (p.timeline === 'window'
      ? 'consecutive window' + (p.timeline_start ? ' from ' + p.timeline_start : ' (start from the seed)')
      : 'spread over the history') + (p.load_aggregation_s ? '; load averaged over ' + pfStepLabel(p.load_aggregation_s) : '') +
      '; new assignment every ' + p.reassignment_period_timesteps);
    var lvl = p.load_level;
    add('Load level', lvl.mode === 'off' ? 'off (historical load)' :
      (lvl.mode === 'loadability' ? 'fraction of P_max ' : 'multiplier ') +
      (lvl.min !== null || lvl.max !== null ? range(lvl.min, lvl.max) : '(default range)') + ', ' + lvl.sampling +
      (lvl.sampling === 'stratified' ? ' in ' + lvl.bins + ' bins' : ''));
    var ls = p.load_shape;
    var noise = [];
    if (p.load_noise_sigma > 0) noise.push('local ' + p.load_noise_sigma);
    if (ls.spatial_sigma > 0) noise.push('spatial ' + ls.spatial_sigma + ' over ' + ls.correlation_km + ' km');
    if (ls.q_sigma > 0) noise.push('Q ' + ls.q_sigma);
    add('Load shape', ls.point_sizing + ' sizing' + (ls.point_sizing === 'series' ? '' : ', utilisation ' + ls.utilisation +
        (ls.size_by === 'peak' ? ' of the peak (P99)' : ' of the average')) +
        (ls.feeders_per_point === 'by_kva' ? ', feeders per point by kVA (if a Feeder database)' : '') +
        ', pf ' + range(ls.pf_min, ls.pf_max) +
        (ls.reactive === 'pf' ? ' (always)' : '') + '; noise ' + (noise.length ? noise.join(', ') : 'off'));
    var d = p.der;
    var label = function (pairs, v) { var m = pairs.find(function (q) { return q[0] === v; }); return m ? m[1] : v; };
    var groups = d.groups.map(function (g) {
      return (g.interpretation === 'individual' ? 'every series of ' + g.database_id + ' (as recorded)'
              : range(g.count_min, g.count_max) + ' × ' + g.capacity_mw + ' MW from ' + g.database_id +
                (g.count_min !== g.count_max ? ' (drawn per grid)' : '')) +
        ', ' + label(PF_PLACEMENTS, g.placement).toLowerCase() + ', ' + label(PF_CONTROLS, g.control);
    });
    var gen = d.generation;
    add('DER', d.source === 'grid' ? "grid's own" + (d.resample_grid_settings ? ', settings redrawn' : '') :
      (d.source === 'add' ? "added to the grid's: " : 'generated only: ') + groups.join('; ') +
      (d.layout === 'per_run' ? '; placed once per run' : '; placed anew per sample') +
      (gen.generation_start ? '; window from ' + gen.generation_start.replace('T', ' ') +
        (gen.keep_time_of_day ? ' (whole days)' : '') : '; same instants as the load') + ', clock ' + gen.timezone +
      (d.source === 'add' && d.resample_grid_settings ? "; grid's settings redrawn" : ''));
    var sp = p.setpoints, spParts = [];
    spParts.push(sp.slack_mode === 'uniform' ? 'busbar ' + range(p.slack_voltage_pu.min, p.slack_voltage_pu.max) + ' p.u.'
      : sp.slack_mode === 'load_dependent' ? 'busbar load-dependent ' + sp.slack_v0 + '→' + sp.slack_v1 + ' ± ' + sp.slack_noise + ' p.u.'
      : 'busbar 1.0 p.u.');
    if (sp.per_substation) spParts.push('per substation');
    if (sp.shunt_steps !== 'fixed') spParts.push('capacitor steps ' + sp.shunt_steps +
      (sp.shunt_steps === 'voltage' ? ' (' + range(sp.shunt_band_min, sp.shunt_band_max) + ' p.u.)' : ''));
    if (sp.regulators_resample) spParts.push('regulators redrawn (set-point ' + range(sp.reg_v_set_min, sp.reg_v_set_max) +
      ', deadband ' + range(sp.reg_deadband_min, sp.reg_deadband_max) + ')');
    if (sp.transformer_taps_resample) spParts.push('transformer taps redrawn' +
      (sp.transformer_tap_steps !== null ? ' within ±' + sp.transformer_tap_steps + ' steps' : ''));
    add('Set-points', spParts.join(', '));
    var t = p.topology_perturbation, topo = [];
    if (t.type === 'n_minus_k') topo.push('N-' + t.k + ' outages');
    if (t.type === 'random') topo.push(t.n_variants + ' random outage sets of up to ' + t.k);
    if (p.outages.transformers) topo.push('transformers included');
    if (p.reconfiguration.enabled) topo.push(p.reconfiguration.n_configs + ' reconfigurations (' +
      (p.reconfiguration.switchable === 'all_lines' ? 'all lines' : 'switches') +
      (p.reconfiguration.allow_meshed ? ', meshed +' + p.reconfiguration.extra_closed : '') + ')');
    if (p.variants_per_sample !== 'all') topo.push(p.variants_per_sample + ' variant(s) per sample');
    add('Topology', (topo.length ? topo.join(', ') : 'base topology only') + '; ' +
        (p.solver === 'newton' ? 'Newton-Raphson' : 'sweep') + ' solver');
    var o = p.outages;
    add('Per-sample outages', o.der_trip_p || o.shunt_out_p || o.regulator_bypass_p
      ? 'DER ' + o.der_trip_p + ', banks ' + o.shunt_out_p + ', regulators ' + o.regulator_bypass_p : 'off');
    var par = p.parameters, pp = [];
    if (par.length_sigma) pp.push('length ±' + par.length_sigma);
    if (par.temperature) pp.push('temperature ' + range(par.temperature.min, par.temperature.max) + ' °C');
    if (par.swap_p) pp.push('cable swaps ' + par.swap_p);
    if (par.transformer_uk_sigma || par.transformer_xr_sigma) pp.push('transformers uk ±' + par.transformer_uk_sigma +
      ', X/R ±' + par.transformer_xr_sigma);
    if (p.admittance_perturbation.enabled) pp.push('admittance ±' + p.admittance_perturbation.sigma);
    if (par.catalogue.length) pp.push(par.catalogue.length + ' extra cable types');
    add('Parameters', pp.length ? pp.join(', ') + (pp.length && par.draw === 'per_sample' ? ' (per sample)' : '') : 'off');
    var tm = p.target_mix;
    add('Design', p.design.mode + (tm.min_violation_share !== null
      ? '; target ' + Math.round(100 * tm.min_violation_share) + ' % ' + tm.violations + ' violations, pilot ' +
        Math.round(100 * tm.pilot_share) + ' %' : '; no target mix'));
    var m = p.measurements;
    add('Measurements', m.enabled ? Math.round(100 * m.cp_coverage) + ' % of points, ' +
      Math.round(100 * m.cable_coverage) + ' % of cables, DER from ' + m.der_min_mva + ' MVA; noise ' + m.profile +
      ' / substation ' + m.substation_profile + '; ' + Math.round(100 * m.missing_p) + ' % missing' : 'off');
    return rows;
  }

  async function updateReview() {
    var grid = $('reviewGrid');
    var list = $('reviewProblems');
    grid.innerHTML = '';
    list.innerHTML = '';
    var problems = [];
    for (var s = 1; s < TOTAL_STEPS; s++) {
      problemsOf(s).forEach(function (text) { problems.push([s, text]); });
    }
    problems.forEach(function (pr) {
      var li = document.createElement('li');
      var a = document.createElement('a');
      a.textContent = 'Step ' + pr[0];
      a.addEventListener('click', function () { goStep(pr[0]); });
      li.appendChild(a);
      li.appendChild(document.createTextNode(': ' + pr[1]));
      list.appendChild(li);
    });
    try {
      summarise(await buildPayload(), pfSelectedGridIds()).forEach(function (row) {
        var dt = document.createElement('dt');
        dt.textContent = row[0];
        var dd = document.createElement('dd');
        dd.textContent = row[1];
        grid.appendChild(dt);
        grid.appendChild(dd);
      });
    } catch (err) {
      var li = document.createElement('li');
      li.textContent = err.message;
      list.appendChild(li);
    }
    $('pfStartBtn').disabled = list.children.length > 0;
  }

  // ── Start ──

  function pfSetMsg(text, isError) {
    var el = $('pfDatagenMsg');
    el.textContent = text;
    el.style.color = isError ? 'var(--color-danger)' : '';
    el.style.fontWeight = isError ? '600' : '';
  }

  function detailText(d) {
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map(function (e) { return (e.loc || []).slice(1).join('.') + ': ' + e.msg; }).join('; ');
    if (d && d.message) {
      return d.message + ' ' + (d.skipped || []).map(function (s) { return s.mv_grid_id + ' (' + s.reason + ')'; }).join(', ');
    }
    return JSON.stringify(d);
  }

  async function pfReadyToSend() {
    await updateReview();
    if ($('reviewProblems').children.length) {
      pfSetMsg('Fix the problems listed above first.', true);
      return false;
    }
    return true;
  }

  // Standalone package: the same request, built here and downloaded as a zip
  // (POST /pf-datagen/mv/{grid_id}/package or /pf-datagen/group/package).
  $('pfPackageBtn').addEventListener('click', async function () {
    var btn = $('pfPackageBtn');
    try {
      if (!(await pfReadyToSend())) return;
      var gridIds = pfSelectedGridIds();
      var payload = await buildPayload();
      btn.disabled = true;
      $('pfStartBtn').disabled = true;
      pfSetMsg('Building the package (grid' + (gridIds.length > 1 ? 's' : '') + ' and load scenarios)...', false);
      var res = gridIds.length > 1
        ? await fetch('/pf-datagen/group/package', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mv_grid_ids: gridIds, request: payload })
          })
        : await fetch('/pf-datagen/mv/' + encodeURIComponent(gridIds[0]) + '/package', {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
          });
      if (!res.ok) {
        var err = await res.json().catch(function () { return { detail: res.statusText }; });
        throw new Error(detailText(err.detail));
      }
      var blob = await res.blob();
      var m = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') || '');
      var name = m ? m[1] : 'gridarena_dataset.zip';
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 10000);
      pfSetMsg('Downloaded ' + name + ' (' + (blob.size / 1048576).toFixed(1) + ' MB). Unzip it and follow how_to_run.md.', false);
    } catch (err) {
      pfSetMsg('Could not build the package: ' + err.message, true);
    } finally {
      btn.disabled = false;
      $('pfStartBtn').disabled = false;
    }
  });

  $('pfDatagenForm').addEventListener('submit', async function (e) {
    e.preventDefault();
    var btn = $('pfStartBtn');
    try {
      if (!(await pfReadyToSend())) return;
      var gridIds = pfSelectedGridIds();
      var payload = await buildPayload();
      btn.disabled = true;
      pfSetMsg('Starting...', false);
      var res, data;
      if (gridIds.length > 1) {
        res = await fetch('/pf-datagen/group/run', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mv_grid_ids: gridIds, request: payload })
        });
        data = await res.json();
        if (!res.ok) throw new Error(detailText(data.detail));
        pfSetMsg(data.message + (data.skipped.length ? ' Skipped: ' + data.skipped.map(function (s) {
          return s.mv_grid_id + ' (' + s.reason + ')'; }).join(', ') + '.' : '') + ' Opening the group...', false);
        location.href = '/pf-datagen/ui/groups/' + data.group_id;
        return;
      }
      res = await fetch('/pf-datagen/mv/' + encodeURIComponent(gridIds[0]) + '/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      data = await res.json();
      if (!res.ok) throw new Error(detailText(data.detail));
      pfSetMsg(data.message + ' Opening the run...', false);
      location.href = '/pf-datagen/ui/runs/' + data.run_id;
    } catch (err) {
      pfSetMsg('Could not start: ' + err.message, true);
      btn.disabled = false;
    }
  });
})();
