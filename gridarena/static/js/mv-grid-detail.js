(function () {
  var mvGridId = window.MV_GRID_ID;

  // ── Render network graph on load ──
  if (window.GRAPH_DATA) {
    ensurePlotly(function () {
      renderMVGraph('networkGraph', window.GRAPH_DATA, function (cpId) {
        showConnectForm(cpId);
      });
    });
  }

  var labelToggle = document.getElementById('toggleNodeLabels');
  if (labelToggle) {
    labelToggle.addEventListener('change', function () {
      Plotly.restyle('networkGraph', { mode: this.checked ? 'markers+text' : 'markers' },
        mvGraphLabelTraces('networkGraph'));
    });
  }

  // ── Tie-line suggestions: proposals drawn dashed on the graph, stored on "Add selected" ──
  var tieSuggestions = [];
  var tieMsg = document.getElementById('tieMsg');
  var tiePanel = document.getElementById('tiePanel');
  var TIE_WHY = { distant_along_cables: 'far along the cables', different_feeders: 'different feeders' };

  function clearTieTrace() {
    var gd = document.getElementById('networkGraph');
    if (!gd || !gd.data) return;
    var idx = gd.data.map(function (t, i) { return t.meta === 'tie-suggestions' ? i : -1; })
      .filter(function (i) { return i >= 0; });
    if (idx.length) Plotly.deleteTraces('networkGraph', idx);
  }

  function drawTieTrace() {
    clearTieTrace();
    var pos = {};
    window.GRAPH_DATA.nodes.forEach(function (n) { pos[n.id] = n; });
    var x = [], y = [], hx = [], hy = [], text = [];
    tieSuggestions.forEach(function (t) {
      var a = pos[t.from_node_id], b = pos[t.to_node_id];
      if (!a || !b) return;
      x.push(a.x, b.x, null); y.push(a.y, b.y, null);
      hx.push((a.x + b.x) / 2); hy.push((a.y + b.y) / 2);
      text.push('<b>' + t.connection_id + '</b> (suggested tie)<br>' + t.from_node_id + ' – ' + t.to_node_id +
        '<br>' + Math.round(t.distance_m) + ' m apart, ' +
        (t.cable_distance_m === null ? 'no cable path' : Math.round(t.cable_distance_m) + ' m along cables'));
    });
    Plotly.addTraces('networkGraph', [
      { x: x, y: y, mode: 'lines', line: { width: 3, color: GA.palette.purple, dash: 'dash' },
        hoverinfo: 'skip', name: 'Suggested tie', meta: 'tie-suggestions' },
      { x: hx, y: hy, text: text, mode: 'markers', marker: { size: 12, color: 'rgba(0,0,0,0)' },
        hoverinfo: 'text', showlegend: false, meta: 'tie-suggestions' }
    ]);
  }

  function tieRow(t, i) {
    var tr = document.createElement('tr');
    var cells = [
      '<input type="checkbox" data-tie="' + i + '" checked>', t.connection_id, t.from_node_id, t.to_node_id,
      Math.round(t.distance_m), t.cable_distance_m === null ? '—' : Math.round(t.cable_distance_m),
      TIE_WHY[t.reason] || t.reason, t.cable_id || '—', t.length
    ];
    cells.forEach(function (c, k) {
      var td = document.createElement('td');
      if (k === 0) td.innerHTML = c; else td.textContent = c;
      tr.appendChild(td);
    });
    return tr;
  }

  var suggestBtn = document.getElementById('tieSuggest');
  if (suggestBtn) {
    suggestBtn.addEventListener('click', async function () {
      var q = new URLSearchParams({
        max_distance_m: document.getElementById('tieMaxDist').value,
        min_cable_distance_m: document.getElementById('tieMinCable').value,
        max_ties: document.getElementById('tieMax').value
      });
      tiePanel.hidden = false;
      tieMsg.classList.remove('error');
      tieMsg.textContent = 'Looking for tie lines...';
      try {
        var res = await fetch('/mv_grid/' + encodeURIComponent(mvGridId) + '/tie-suggestions?' + q.toString());
        var data = await res.json();
        if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed');
        tieSuggestions = data.suggestions;
        var body = document.querySelector('#tieTable tbody');
        body.innerHTML = '';
        tieSuggestions.forEach(function (t, i) { body.appendChild(tieRow(t, i)); });
        document.getElementById('tieTable').hidden = tieSuggestions.length === 0;
        document.getElementById('tieAdd').hidden = tieSuggestions.length === 0;
        tieMsg.textContent = tieSuggestions.length
          ? tieSuggestions.length + ' tie(s) proposed from ' + data.n_candidates +
            ' candidate pair(s). Keep the ones you want and press Add selected; they are stored as normally-open switches.'
          : 'No pair of nodes meets these distances.';
        drawTieTrace();
      } catch (err) {
        tieMsg.classList.add('error');
        tieMsg.textContent = 'Error: ' + err.message;
      }
    });
    document.getElementById('tieCancel').addEventListener('click', function () {
      tieSuggestions = [];
      clearTieTrace();
      tiePanel.hidden = true;
    });
    document.getElementById('tieAdd').addEventListener('click', async function () {
      var btn = this;
      var ties = Array.prototype.slice.call(document.querySelectorAll('[data-tie]'))
        .filter(function (el) { return el.checked; })
        .map(function (el) {
          var t = tieSuggestions[+el.dataset.tie];
          return { connection_id: t.connection_id, from_node_id: t.from_node_id, to_node_id: t.to_node_id,
                   cable_id: t.cable_id, length: t.length };
        });
      tieMsg.classList.remove('error');
      if (!ties.length) { tieMsg.textContent = 'Select at least one tie.'; return; }
      btn.disabled = true;
      tieMsg.textContent = 'Adding...';
      try {
        var res = await fetch('/mv_grid/' + encodeURIComponent(mvGridId) + '/ties', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ties: ties })
        });
        var data = await res.json();
        if (!res.ok) {
          var d = data.detail;
          throw new Error(d && d.validation_errors ? d.validation_errors.join(' ')
            : Array.isArray(d) ? d.map(function (e) { return e.msg; }).join(' ') : (d || 'Request failed'));
        }
        tieMsg.textContent = data.message;
        location.reload();
      } catch (err) {
        tieMsg.classList.add('error');
        tieMsg.textContent = 'Error: ' + err.message;
        btn.disabled = false;
      }
    });
  }

  // ── Current configuration: tabs ──
  document.querySelectorAll('[data-cfg-tab]').forEach(function (tab) {
    tab.addEventListener('click', function () {
      document.querySelectorAll('[data-cfg-tab]').forEach(function (t) { t.classList.remove('active'); });
      document.querySelectorAll('.cfg-pane').forEach(function (p) { p.classList.remove('active'); });
      tab.classList.add('active');
      document.getElementById('cfg-' + tab.dataset.cfgTab).classList.add('active');
    });
  });

  // ── Current configuration: edits ──
  var cfgFields = Array.prototype.slice.call(document.querySelectorAll('.cfg-field'));
  var cfgActions = document.getElementById('cfgActions');
  var cfgMsg = document.getElementById('cfgMsg');

  function parseField(el, raw) {
    if (el.dataset.type === 'int') return raw === '' ? NaN : parseInt(raw, 10);
    if (el.dataset.type === 'float') return raw === '' ? NaN : parseFloat(raw);
    if (el.dataset.type === 'bool') return raw === 'true';
    return raw;
  }

  function isChanged(el) {
    return parseField(el, el.value) !== parseField(el, el.dataset.original);
  }

  // DER settings only some control modes use: cos phi / excitation (fixed cos phi, P(V)),
  // V set (voltage control), Q limit (every mode but fixed cos phi).
  var DER_FIELDS_BY_MODE = {
    cos_phi: ['fixed_pf', 'p_v'], excitation: ['fixed_pf', 'p_v'],
    v_set_pu: ['voltage'], q_limit_pu: ['q_v', 'p_v', 'voltage']
  };
  function refreshDerRow(row) {
    var mode = row.querySelector('[data-field="control_mode"]').value;
    Object.keys(DER_FIELDS_BY_MODE).forEach(function (field) {
      var el = row.querySelector('[data-field="' + field + '"]');
      if (el) el.disabled = DER_FIELDS_BY_MODE[field].indexOf(mode) < 0;
    });
  }

  function refreshConfig() {
    var n = 0;
    cfgFields.forEach(function (el) {
      var changed = isChanged(el);
      el.classList.toggle('cfg-changed', changed);
      if (changed) n++;
    });
    document.querySelectorAll('.cfg-der').forEach(refreshDerRow);
    cfgActions.hidden = n === 0;
    document.getElementById('cfgCount').textContent = n === 1 ? '1 change' : n + ' changes';
  }

  cfgFields.forEach(function (el) {
    el.addEventListener('input', refreshConfig);
    el.addEventListener('change', refreshConfig);
  });
  if (cfgActions) refreshConfig();

  var discardBtn = document.getElementById('cfgDiscard');
  if (discardBtn) {
    discardBtn.addEventListener('click', function () {
      cfgFields.forEach(function (el) { el.value = el.dataset.original; });
      cfgMsg.textContent = '';
      cfgMsg.classList.remove('error');
      refreshConfig();
    });
  }

  var ID_FIELD = { connections: 'connection_id', transformers: 'transformer_id', regulators: 'regulator_id',
                   shunts: 'shunt_id', ders: 'der_id' };

  var updateBtn = document.getElementById('cfgUpdate');
  if (updateBtn) {
    updateBtn.addEventListener('click', async function () {
      // One edit per element, holding only its changed fields.
      var edits = {}, invalid = [];
      cfgFields.forEach(function (el) {
        if (!isChanged(el)) return;
        var value = parseField(el, el.value);
        if (typeof value === 'number' && (isNaN(value) || el.validity.rangeUnderflow || el.validity.rangeOverflow)) {
          invalid.push(el.dataset.id + ' ' + el.dataset.field);
          return;
        }
        var kind = el.dataset.kind, byId = edits[kind] = edits[kind] || {};
        var edit = byId[el.dataset.id] = byId[el.dataset.id] || {};
        edit[ID_FIELD[kind]] = el.dataset.id;
        edit[el.dataset.field] = value;
      });
      cfgMsg.classList.remove('error');
      if (invalid.length) {
        cfgMsg.classList.add('error');
        cfgMsg.textContent = 'Out of range: ' + invalid.join(', ');
        return;
      }
      var payload = {};
      Object.keys(edits).forEach(function (kind) {
        payload[kind] = Object.keys(edits[kind]).map(function (id) { return edits[kind][id]; });
      });

      updateBtn.disabled = true;
      cfgMsg.textContent = 'Updating...';
      try {
        var res = await fetch('/mv_grid/' + encodeURIComponent(mvGridId) + '/configuration', {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        var data = await res.json();
        if (!res.ok) {
          var d = data.detail;
          var msg = d && d.validation_errors ? d.validation_errors.join(' ')
            : Array.isArray(d) ? d.map(function (e) { return e.msg; }).join(' ') : (d || 'Update failed');
          throw new Error(msg);
        }
        cfgMsg.textContent = data.message;
        location.reload();
      } catch (err) {
        cfgMsg.classList.add('error');
        cfgMsg.textContent = 'Error: ' + err.message;
        updateBtn.disabled = false;
      }
    });
  }

  // ── Connect form ──
  window.showConnectForm = function (cpId) {
    document.getElementById('connectSection').style.display = 'block';
    document.getElementById('connectPointId').textContent = cpId;
    document.getElementById('connectCpId').value = cpId;
    document.getElementById('connectMsg').textContent = '';
    document.getElementById('lvGridSelect').value = '';
    document.getElementById('connectSection').scrollIntoView({ behavior: 'smooth' });
  };

  var connectForm = document.getElementById('connectForm');
  if (connectForm) {
    connectForm.addEventListener('submit', async function (e) {
      e.preventDefault();
      var cpId = document.getElementById('connectCpId').value;
      var lvGridId = document.getElementById('lvGridSelect').value;
      var msg = document.getElementById('connectMsg');

      if (!lvGridId) { msg.textContent = 'Please select an LV grid.'; return; }

      var impReal = document.getElementById('trImpReal').value;
      var impImag = document.getElementById('trImpImag').value;

      var payload = {
        lv_grid_id: lvGridId,
        rated_power_kva: parseFloat(document.getElementById('ratedPower').value) || 400,
        primary_voltage_kv: parseFloat(document.getElementById('primaryVoltage').value) || 20,
        secondary_voltage_v: parseFloat(document.getElementById('secondaryVoltage').value) || 230,
        imp_real: impReal ? parseFloat(impReal) : null,
        imp_imag: impImag ? parseFloat(impImag) : null
      };

      msg.textContent = 'Connecting...';
      try {
        var res = await fetch('/mv_grid/' + mvGridId + '/connect/' + encodeURIComponent(cpId), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        var data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Connection failed');
        msg.textContent = data.message;
        setTimeout(function () { location.reload(); }, 1500);
      } catch (err) {
        msg.textContent = 'Error: ' + err.message;
      }
    });
  }

  // ── Disconnect ──
  window.disconnectLV = async function (mvId, cpId, lvId) {
    if (!confirm('Disconnect LV grid "' + lvId + '" from point "' + cpId + '"?')) return;
    try {
      var res = await fetch('/mv_grid/' + mvId + '/disconnect/' + encodeURIComponent(cpId) + '/' + encodeURIComponent(lvId), {
        method: 'DELETE'
      });
      var data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Disconnect failed');
      location.reload();
    } catch (err) {
      alert('Error: ' + err.message);
    }
  };

  // ── View P/V ──
  window.viewPV = async function (mvId, lvId) {
    var section = document.getElementById('pvSection');
    var info = document.getElementById('pvInfo');

    section.style.display = 'block';
    document.getElementById('pvLvId').textContent = lvId;
    info.textContent = 'Loading...';
    section.scrollIntoView({ behavior: 'smooth' });

    try {
      var res = await fetch('/mv_grid/' + mvId + '/pv/' + encodeURIComponent(lvId));
      var data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load P/V data');

      info.textContent = data.n_nodes + ' nodes | ' + data.count + ' timestamps | Aggregation: ' + data.aggregation;

      var timestamps = data.pv_pairs.map(function (p) { return p.datetime; });
      var powers = data.pv_pairs.map(function (p) { return p.avg_power_active; });
      var voltages = data.pv_pairs.map(function (p) { return p.avg_voltage_magnitude; });

      ensurePlotly(function () {
        Plotly.newPlot('pvChart', [
          { x: timestamps, y: powers, name: 'Avg Power (W)', yaxis: 'y1' },
          { x: timestamps, y: voltages, name: 'Avg Voltage (V)', yaxis: 'y2' }
        ], GA.chartLayout({
          title: 'Aggregated P/V at MV/LV Boundary — ' + lvId,
          xaxis: { title: 'Datetime' },
          yaxis: { title: 'Power (W)', side: 'left' },
          yaxis2: { title: 'Voltage (V)', side: 'right', overlaying: 'y' },
          legend: { orientation: 'h' }
        }));
      });
    } catch (err) {
      info.textContent = 'Error: ' + err.message;
    }
  };
})();
