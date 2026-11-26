/**
 * Run and group pages of the Synthetic Data section
 * (/pf-datagen/ui/runs/{run_id}, /pf-datagen/ui/groups/{group_id}): live
 * progress while a run is going, then its summary, diversity report, charts,
 * exports and split suggestions; for a group, the combined report and a row
 * per grid.
 */
(function () {

  // Plotly loads on first use (the MV graph script has its own copy of this helper).
  function ensurePlotly(callback) {
    if (window.Plotly) {
      callback();
    } else {
      var s = document.createElement('script');
      s.src = 'https://cdn.plot.ly/plotly-2.26.0.min.js';
      s.onload = callback;
      document.head.appendChild(s);
    }
  }

  function samplesText(r) {
    if (!r.total_scenarios) return r.status === 'queued' ? 'waiting to start' : 'preparing';
    return r.completed_scenarios + (r.failed_scenarios ? ' + ' + r.failed_scenarios + ' failed' : '') + ' / ' +
      r.total_scenarios;
  }

  // ── Group view ──

  function pfPct(x) { return x == null ? '—' : (100 * x).toFixed(1) + ' %'; }

  async function loadGroupReport(groupId) {
    var info = document.getElementById('pfGroupInfo');
    try {
      var res = await fetch('/pf-datagen/groups/' + groupId + '/report');
      var rep = await res.json();
      if (!res.ok) throw new Error(rep.detail || 'Failed to load the group');
      var sres = await fetch('/pf-datagen/groups/' + groupId + '/splits');
      var splits = sres.ok ? await sres.json() : { grids: {} };
      var statusText = Object.keys(rep.status_counts).map(function (k) { return rep.status_counts[k] + ' ' + k; }).join(', ');
      var vmin = rep.per_sample && rep.per_sample.vm_pu_min, load = rep.per_sample && rep.per_sample.max_loading_pct;
      info.textContent = rep.n_runs + ' runs (' + statusText + ')' +
        (rep.skipped.length ? ' | skipped: ' + rep.skipped.map(function (s) { return s.mv_grid_id + ' (' + s.reason + ')'; }).join(', ') : '') +
        ' | outputs ' + rep.outcomes.converged + '/' + rep.outcomes.total + ' converged' +
        (vmin ? ' | lowest voltage per sample ' + vmin.p5.toFixed(3) + '–' + vmin.p95.toFixed(3) + ' p.u. (5–95 %)' : '') +
        (load ? ' | highest loading per sample median ' + load.p50.toFixed(0) + ' %' : '');
      var body = document.querySelector('#pfGroupTable tbody');
      body.innerHTML = '';
      rep.grids.forEach(function (g) {
        var tr = document.createElement('tr');
        [g.mv_grid_id, g.run_id, g.status, g.n_nodes, g.n_connection_points, g.outputs, g.converged,
         pfPct(g.undervoltage_rate), pfPct(g.overvoltage_rate), pfPct(g.thermal_rate),
         (splits.grids || {})[g.mv_grid_id] || '—'].forEach(function (v, i) {
          var td = document.createElement('td');
          if (i === 1 && v) {
            var a = document.createElement('a');
            a.href = '/pf-datagen/ui/runs/' + v;
            a.textContent = v;
            td.appendChild(a);
          } else {
            td.textContent = v == null ? '—' : v;
          }
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      document.getElementById('pfGroupLinks').innerHTML = 'Group report: <a href="/pf-datagen/groups/' + groupId +
        '/report">JSON</a> | split by grid: <a href="/pf-datagen/groups/' + groupId + '/splits">JSON</a>';
      if (rep.voltage_pu) {
        ensurePlotly(function () {
          Plotly.newPlot('pfGroupVoltageChart', [pfHistogramTrace(rep.voltage_pu, GA.palette.cyan)], GA.chartLayout({
            title: 'Bus voltage distribution (whole group)',
            xaxis: { title: 'Voltage (p.u. of nominal)' }, yaxis: { title: 'Bus results' }, bargap: 0.05
          }));
        });
      }
    } catch (err) {
      info.textContent = 'Error: ' + err.message;
    }
  }

  // ── Results viewer ──

  // Bars from a report histogram ({edges, counts}).
  function pfHistogramTrace(h, color) {
    var mids = h.edges.slice(0, -1).map(function (e, i) { return (e + h.edges[i + 1]) / 2; });
    return { x: mids, y: h.counts, type: 'bar', marker: { color: color },
             width: (h.edges[1] - h.edges[0]) * 0.95 };
  }

  function pfShowChart(id, traces, layout) {
    var el = document.getElementById(id);
    el.style.display = 'block';
    Plotly.newPlot(id, traces, GA.chartLayout(layout));
  }

  function pfReportText(r) {
    var parts = [];
    if (r.loadability) {
      var l = r.loadability;
      parts.push('P_max ' + (l.p_max_w / 1e6).toFixed(3) + ' MW (limited by ' + l.p_max_limit + ', ' +
                 l.p_max_solves + ' solves)');
    }
    var pf = r.load && r.load.load_pf, cv = r.load && r.load.spatial_cv;
    if (pf) parts.push('net load power factor ' + pf.p5.toFixed(2) + '–' + pf.p95.toFixed(2) +
                       ' (5–95 %; negative = the loads export on balance)');
    if (cv) parts.push('spatial spread of load (CV) median ' + cv.p50.toFixed(2));
    var ratio = r.der_load_ratio;
    if (ratio && ratio.p95 != null) parts.push('DER / gross load median ' + ratio.p50.toFixed(2) + ', 95th percentile ' +
                                       ratio.p95.toFixed(2));
    var tp = r.topology;
    if (tp && tp.n_variants > 1) {
      parts.push(tp.distinct_topologies + ' distinct topologies from ' + tp.n_variants + ' variants' +
                 (tp.n_configurations > 1 ? ' (' + tp.n_configurations + ' configurations)' : ''));
      var isl = tp.islanded_load_share;
      if (isl && isl.p100 > 0) parts.push('islanded load median ' + (100 * isl.p50).toFixed(1) + ' %, max ' +
                                          (100 * isl.p100).toFixed(1) + ' %');
    }
    if (r.design) {
      var dz = r.design;
      parts.push('design ' + dz.mode + ': ' + dz.strata_filled + '/' + dz.n_strata + ' strata filled' +
                 (dz.max_rank_correlation != null ? ', largest rank correlation ' + dz.max_rank_correlation.toFixed(3) : ''));
    }
    var tm = r.target_mix;
    if (tm && tm.achieved_share != null) {
      parts.push('target mix ' + (100 * tm.min_violation_share).toFixed(0) + ' %: reached ' +
                 (100 * tm.achieved_share).toFixed(1) + ' %' + (tm.reachable === false ? ' (not reachable on this grid)' : ''));
    }
    if (r.load && r.load.near_duplicate_share) {
      parts.push((100 * r.load.near_duplicate_share).toFixed(1) + ' % near-duplicate load samples');
    }
    var temp = r.parameters && r.parameters.conductor_temp_c;
    if (temp) parts.push('conductors ' + temp.p0.toFixed(0) + '–' + temp.p100.toFixed(0) + ' °C');
    var sp = r.setpoints && r.setpoints.v_slack_pu;
    if (sp && sp.p100 - sp.p0 > 1e-9) parts.push('busbar ' + sp.p0.toFixed(3) + '–' + sp.p100.toFixed(3) + ' p.u.');
    var tl = r.timeline;
    if (tl && tl.mode === 'window') {
      parts.push('window ' + tl.first + ' to ' + tl.last + (tl.n_gaps ? ' (' + tl.n_gaps + ' gaps)' : '') +
                 (tl.per_sample_axes && tl.per_sample_axes.length
                   ? ', still drawn per sample: ' + tl.per_sample_axes.join(', ') : ''));
    }
    var ms = r.measurements;
    if (ms) {
      var nMeters = Object.keys(ms.meters).filter(function (k) { return k !== 'pseudo'; })
        .reduce(function (a, k) { return a + ms.meters[k]; }, 0);
      parts.push(nMeters + ' meters, ' + (ms.meters.pseudo || 0) + ' pseudo points, ' + ms.readings + ' readings' +
                 (ms.missing_rate != null ? ', ' + (100 * ms.missing_rate).toFixed(1) + ' % missing' : '') +
                 (ms.outlier_rate != null ? ', ' + (100 * ms.outlier_rate).toFixed(2) + ' % outliers' : ''));
    }
    if (r.load_level && r.load_level.unscaled_samples) {
      parts.push(r.load_level.unscaled_samples + ' near-empty samples left unscaled');
    }
    return parts.length ? 'Report (whole run): ' + parts.join(' | ') : '';
  }

  async function loadRunResults(runId) {
    var section = document.getElementById('pfResultsSection');
    var info = document.getElementById('pfResultsInfo');
    section.style.display = 'block';
    info.textContent = 'Loading...';

    try {
      var runRes = await fetch('/pf-datagen/runs/' + runId);
      var run = await runRes.json();
      if (!runRes.ok) throw new Error(run.detail || 'Failed to load run');

      var summary = run.summary || {};
      var failureReasons = summary.failure_reasons || {};
      var failureText = Object.keys(failureReasons).map(function (k) { return k + ' ' + failureReasons[k]; }).join(', ');
      info.textContent = 'MV Grid: ' + run.mv_grid_id +
        ' | Status: ' + run.status +
        ' | Converged: ' + (summary.converged || 0) + '/' + (summary.total_output_groups || 0) +
        (failureText ? ' (failed: ' + failureText + ')' : '') +
        ' | Thermal violations: ' + (summary.thermal_violations || 0) +
        (summary.thermal_violations_emergency ? ' (emergency rating: ' + summary.thermal_violations_emergency + ')' : '') +
        ' | Voltage violations: ' + (summary.voltage_violations || 0) +
        (summary.max_loading_pct != null ? ' | Max loading: ' + summary.max_loading_pct.toFixed(1) + ' %' : '') +
        (summary.loss_pct_mean != null ? ' | Mean losses: ' + summary.loss_pct_mean.toFixed(2) + ' %' : '') +
        ' | Islanded bus rows: ' + (summary.islanded_bus_rows || 0) +
        ' (the thermal-violation chart previews the first 500 branch rows)';

      var exportLinks = document.getElementById('pfExportLinks');
      exportLinks.innerHTML = 'Export: ' +
        ['bus', 'branch', 'ybus', 'runtime', 'summary', 'der', 'variants', 'measurements', 'layout'].map(function (t) {
          return '<a href="/pf-datagen/runs/' + runId + '/export.csv?table=' + t + '">' + t + '.csv</a>';
        }).join(' | ') + ' &mdash; Split suggestions: ' + ['time', 'topology'].map(function (by) {
          return '<a href="/pf-datagen/runs/' + runId + '/splits?by=' + by + '&format=csv">by ' + by + '</a>';
        }).join(' | ');

      var busRes = await fetch('/pf-datagen/runs/' + runId + '/results?table=bus&limit=500');
      var busData = await busRes.json();
      // Per-unit (of the line-to-line nominal voltage) when the run has it;
      // runs written before the *_pu columns existed fall back to volts.
      // Islanded buses (energised === false) are written with 0 V and would
      // pile up at zero, so they're left out; older runs have energised null.
      var busRows = (busData.rows || []).filter(function (r) { return r.energised !== false; });
      var vmPu = busRows.map(function (r) { return r.vm_pu; }).filter(function (v) { return v != null; });
      var usePu = vmPu.length > 0;
      var vmValues = usePu ? vmPu : busRows.map(function (r) { return r.vm; }).filter(function (v) { return v != null; });

      var branchRes = await fetch('/pf-datagen/runs/' + runId + '/results?table=branch&limit=500');
      var branchData = await branchRes.json();

      // Diversity report (whole run), written when the run finishes; older runs have none.
      var reportRes = await fetch('/pf-datagen/runs/' + runId + '/report');
      var report = reportRes.ok ? await reportRes.json() : null;
      document.getElementById('pfReportInfo').textContent = report ? pfReportText(report) : '';
      var violationCounts = {};
      (branchData.rows || []).forEach(function (r) {
        if (r.thermal_violation) {
          violationCounts[r.connection_id] = (violationCounts[r.connection_id] || 0) + 1;
        }
      });

      ensurePlotly(function () {
        // The run's voltage band (default EN 50160 ±10 %), only meaningful on
        // the p.u. axis; runs from before it was configurable use the default.
        var limits = (run.config && run.config.voltage_limits) || { vm_min: 0.9, vm_max: 1.1 };
        var limitLines = usePu ? [limits.vm_min, limits.vm_max].map(function (x) {
          return {
            type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1,
            line: { color: GA.palette.amber, width: 1.5, dash: 'dash' }
          };
        }) : [];
        if (report) {
          // Whole run, from the report's histogram (always p.u.).
          var vmLines = [limits.vm_min, limits.vm_max].map(function (x) {
            return {
              type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1,
              line: { color: GA.palette.amber, width: 1.5, dash: 'dash' }
            };
          });
          Plotly.newPlot('pfVoltageChart', [pfHistogramTrace(report.voltage_pu, GA.palette.cyan)], GA.chartLayout({
            title: 'Bus voltage distribution (whole run)',
            xaxis: { title: 'Voltage (p.u. of nominal)' }, yaxis: { title: 'Bus results' },
            bargap: 0.05, shapes: vmLines
          }));
          pfShowChart('pfLoadingChart', [pfHistogramTrace(report.loading_pct, GA.palette.purple)], {
            title: 'Branch loading distribution (whole run)',
            xaxis: { title: 'Loading (% of normal rating)' }, yaxis: { title: 'Branch results' }, bargap: 0.05,
            shapes: [{ type: 'line', xref: 'x', yref: 'paper', x0: 100, x1: 100, y0: 0, y1: 1,
                       line: { color: GA.palette.red, width: 1.5, dash: 'dash' } }]
          });
          if (report.load_level) {
            var lv = report.load_level;
            var mids = lv.edges.slice(0, -1).map(function (e, i) { return (e + lv.edges[i + 1]) / 2; });
            var rateTrace = function (key, name, color) {
              return { x: mids, y: lv[key], name: name, yaxis: 'y2', mode: 'lines+markers',
                       line: { color: color } };
            };
            pfShowChart('pfLevelChart', [
              { x: mids, y: lv.samples, type: 'bar', name: 'Samples', marker: { color: GA.palette.slate } },
              rateTrace('undervoltage_rate', 'Undervoltage', GA.palette.amber),
              rateTrace('overvoltage_rate', 'Overvoltage', GA.palette.pink),
              rateTrace('thermal_rate', 'Thermal', GA.palette.red)
            ], {
              title: 'Load level: samples and violation rate per band',
              xaxis: { title: lv.mode === 'loadability' ? 'Load level (fraction of P_max)' : 'Load multiplier' },
              yaxis: { title: 'Samples' },
              yaxis2: { title: 'Share of converged samples', overlaying: 'y', side: 'right', range: [0, 1] },
              legend: { orientation: 'h' }
            });
          } else {
            document.getElementById('pfLevelChart').style.display = 'none';
          }
          var kinds = report.topology && report.topology.outcomes_by_kind;
          if (kinds && Object.keys(kinds).length > 1) {
            var names = Object.keys(kinds);
            var kindRate = function (key, name, color) {
              return { x: names, y: names.map(function (k) { return kinds[k][key]; }), name: name, yaxis: 'y2',
                       mode: 'lines+markers', line: { color: color } };
            };
            pfShowChart('pfTopoChart', [
              { x: names, y: names.map(function (k) { return kinds[k].outputs; }), type: 'bar', name: 'Outputs',
                marker: { color: GA.palette.slate } },
              kindRate('undervoltage_rate', 'Undervoltage', GA.palette.amber),
              kindRate('overvoltage_rate', 'Overvoltage', GA.palette.pink),
              kindRate('thermal_rate', 'Thermal', GA.palette.red)
            ], {
              title: 'Topology: outputs and violation rates per variant kind',
              xaxis: { title: 'Variant kind' }, yaxis: { title: 'Outputs' },
              yaxis2: { title: 'Share of converged outputs', overlaying: 'y', side: 'right', range: [0, 1] },
              legend: { orientation: 'h' }
            });
          } else {
            document.getElementById('pfTopoChart').style.display = 'none';
          }
          if (report.der_penetration) {
            var pp = report.der_penetration;
            var pmids = pp.edges.slice(0, -1).map(function (e, i) { return (e + pp.edges[i + 1]) / 2; });
            var penRate = function (key, name, color) {
              return { x: pmids, y: pp[key], name: name, yaxis: 'y2', mode: 'lines+markers', line: { color: color } };
            };
            pfShowChart('pfPenChart', [
              { x: pmids, y: pp.samples, type: 'bar', name: 'Samples', marker: { color: GA.palette.slate } },
              penRate('reverse_flow_rate', 'Reverse flow', GA.palette.green),
              penRate('overvoltage_rate', 'Overvoltage', GA.palette.pink),
              penRate('thermal_rate', 'Thermal', GA.palette.red)
            ], {
              title: 'DER penetration: samples and rates per band (installed capacity / P_max)',
              xaxis: { title: 'Installed DER penetration' }, yaxis: { title: 'Samples' },
              yaxis2: { title: 'Share of converged samples', overlaying: 'y', side: 'right', range: [0, 1] },
              legend: { orientation: 'h' }
            });
          } else {
            document.getElementById('pfPenChart').style.display = 'none';
          }
        } else {
          document.getElementById('pfLoadingChart').style.display = 'none';
          document.getElementById('pfLevelChart').style.display = 'none';
          document.getElementById('pfPenChart').style.display = 'none';
          document.getElementById('pfTopoChart').style.display = 'none';
          Plotly.newPlot('pfVoltageChart', [{ x: vmValues, type: 'histogram', marker: { color: GA.palette.cyan } }], GA.chartLayout({
            title: 'Solved Voltage Magnitude Distribution (preview)',
            xaxis: { title: usePu ? 'Voltage (p.u. of nominal)' : 'Voltage, line-to-line (V)' },
            yaxis: { title: 'Count' },
            shapes: limitLines
          }));
        }

        var connIds = Object.keys(violationCounts);
        Plotly.newPlot('pfViolationChart', [{
          x: connIds, y: connIds.map(function (c) { return violationCounts[c]; }),
          type: 'bar', marker: { color: GA.palette.red }
        }], GA.chartLayout({
          title: 'Thermal Violations by Connection (preview)',
          xaxis: { title: 'Connection ID' }, yaxis: { title: 'Violation Count' }
        }));
      });
    } catch (err) {
      info.textContent = 'Error: ' + err.message;
    }
  }

  // ── Page set-up ──

  var ACTIVE = { queued: true, running: true };

  // A run page: follow its progress while queued or running, then show its results.
  window.pfInitRunPage = function (runId, status) {
    if (status === 'completed') {
      loadRunResults(runId);
      return;
    }
    if (!ACTIVE[status]) return;
    var timer = setInterval(async function () {
      try {
        var res = await fetch('/pf-datagen/runs/' + runId);
        if (!res.ok) return;
        var run = await res.json();
        var badge = document.getElementById('pfRunStatus');
        badge.textContent = run.status;
        badge.className = 'status-badge status-badge--' + run.status;
        document.getElementById('pfRunProgress').textContent = samplesText(run);
        var bar = document.getElementById('pfRunBar');
        if (bar && run.total_scenarios) {
          bar.style.width = (100 * (run.completed_scenarios + run.failed_scenarios) / run.total_scenarios) + '%';
        }
        if (!ACTIVE[run.status]) {
          clearInterval(timer);
          location.reload();          // the page renders the finished run's details and results
        }
      } catch (err) { /* keep polling */ }
    }, 3000);
  };

  // A group page: its runs' progress (refreshed while any is active) and the combined report.
  window.pfInitGroupPage = function (groupId) {
    var refresh = async function () {
      var active = false;
      try {
        var res = await fetch('/pf-datagen/groups/' + groupId);
        if (res.ok) {
          var g = await res.json();
          var p = g.progress;
          document.getElementById('pfGroupProgress').textContent = p.solved + ' / ' + p.total + ' samples';
          document.getElementById('pfGroupBar').style.width = (p.total ? 100 * p.solved / p.total : 0) + '%';
          active = g.runs.some(function (r) { return ACTIVE[r.status]; });
          document.getElementById('pfGroupBarWrap').style.display = active ? '' : 'none';
        }
      } catch (err) { /* the report below says what failed */ }
      await loadGroupReport(groupId);
      if (active) setTimeout(refresh, 5000);
    };
    refresh();
  };
})();
