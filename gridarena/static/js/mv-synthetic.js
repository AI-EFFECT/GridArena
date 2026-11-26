/**
 * Synthetic Grids page (/pf-datagen/ui/synthetic-grids): generate grids
 * calibrated on the stored ones (POST /mv_grid/synthetic) and show the
 * generator report, or delete every synthetic grid (DELETE /mv_grid/synthetic).
 */
(function () {
  var msg = document.getElementById('synthMsg');
  if (!msg) return;

  var METRIC_LABEL = {
    n_nodes: 'Nodes', section_m: 'Section length (m)', depth: 'Depth (sections)', far_km: 'Farthest node (km)',
    branching: 'Branching nodes', cp_share: 'Connection points', overhead_share: 'Overhead share'
  };

  function setMsg(text, isError) {
    msg.textContent = text;
    msg.style.color = isError ? 'var(--color-danger)' : '';
  }

  function num(id) {
    var v = document.getElementById(id).value;
    return v === '' ? null : Number(v);
  }

  function fmt(x) {
    if (x == null) return '—';
    return Math.abs(x) >= 10 ? x.toFixed(0) : x.toFixed(2);
  }

  function renderReport(report) {
    var out = document.getElementById('synthReport');
    out.innerHTML = '';
    Object.keys(report.archetypes).forEach(function (a) {
      var arch = report.archetypes[a];
      var h = document.createElement('h4');
      h.textContent = a + ' (' + arch.generated + ' generated): generated vs stored, p10 / p50 / p90';
      out.appendChild(h);
      var table = document.createElement('table');
      table.className = 'data-table';
      table.innerHTML = '<thead><tr><th>Statistic</th><th>Generated</th><th>Stored</th></tr></thead>';
      var body = document.createElement('tbody');
      Object.keys(arch.metrics).forEach(function (k) {
        var m = arch.metrics[k];
        var tr = document.createElement('tr');
        [METRIC_LABEL[k] || k,
         m.generated ? [m.generated.p10, m.generated.p50, m.generated.p90].map(fmt).join(' / ') : '—',
         m.stored ? [m.stored.p10, m.stored.p50, m.stored.p90].map(fmt).join(' / ') : '—'
        ].forEach(function (v) {
          var td = document.createElement('td');
          td.textContent = v;
          tr.appendChild(td);
        });
        body.appendChild(tr);
      });
      table.appendChild(body);
      out.appendChild(table);
    });
  }

  document.getElementById('synthGenerate').addEventListener('click', async function () {
    var btn = this;
    var body = {
      archetype: document.getElementById('synthArchetype').value,
      count: num('synthCount') || 5,
      seed: num('synthSeed'),
      feeders_per_substation: num('synthFeeders') || 1,
      second_substation: document.getElementById('synthSecondSub').checked,
      ties_between_feeders: num('synthTies') || 0,
      n_nodes_min: num('synthNodesMin'),
      n_nodes_max: num('synthNodesMax')
    };
    btn.disabled = true;
    setMsg('Generating...', false);
    try {
      var res = await fetch('/mv_grid/synthetic', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
      });
      var data = await res.json();
      if (!res.ok) {
        var d = data.detail;
        throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map(function (e) { return e.msg; }).join(' ')
          : JSON.stringify(d));
      }
      setMsg(data.message + ' Seed ' + data.seed + ': ' + data.created.join(', ') + '. Reload the page to list them below.', false);
      renderReport(data.report);
    } catch (err) {
      setMsg('Error: ' + err.message, true);
    } finally {
      btn.disabled = false;
    }
  });

  document.getElementById('synthDelete').addEventListener('click', async function () {
    if (!window.confirm('Delete every synthetic grid (SYN-...) and everything stored for them?')) return;
    var btn = this;
    btn.disabled = true;
    try {
      var res = await fetch('/mv_grid/synthetic', { method: 'DELETE' });
      var data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      setMsg(data.message, false);
      if (data.deleted) setTimeout(function () { location.reload(); }, 1200);
    } catch (err) {
      setMsg('Error: ' + err.message, true);
    } finally {
      btn.disabled = false;
    }
  });
})();
