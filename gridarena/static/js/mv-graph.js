/**
 * Shared function to render an MV grid graph with Plotly.
 * Connection point nodes are green and clickable.
 *
 * Besides nodes and cables it draws the elements attached to them:
 * switches (filled square closed, hollow square open) and regulators
 * (hourglass, near the regulated end) on connections; capacitor banks /
 * reactors and DER as small symbols offset from their node, joined to it
 * by a thin stem. Tie lines (normally open) are dashed, out-of-service
 * connections dotted. Hover any element for its details.
 *
 * @param {string} containerId  - DOM element ID for the chart
 * @param {object} graphData    - {nodes: [...], edges: [...]} from the server
 * @param {function} onClickCP  - callback(connection_point_id) when a CP node is clicked
 */
var MV_DER_STYLE = {
  pv: { symbol: 'star', color: '#E3B341', name: 'PV' },
  wind: { symbol: 'star-triangle-up', color: '#39C5CF', name: 'Wind' },
  chp: { symbol: 'hexagon', color: '#FF7B72', name: 'CHP' },
  battery: { symbol: 'square-x', color: '#7EE787', name: 'Battery' },
  ev_hub: { symbol: 'pentagon', color: '#D2A8FF', name: 'EV hub' },
  other: { symbol: 'circle-x', color: '#8b949e', name: 'Other DER' }
};

var MV_CONTROL_LABEL = { fixed_pf: 'fixed cos φ', q_v: 'Q(V)', p_v: 'P(V)', voltage: 'voltage control' };

function renderMVGraph(containerId, graphData, onClickCP) {
  var nodes = graphData.nodes;
  var edges = graphData.edges;
  var P = GA.palette;

  var posMap = {};
  nodes.forEach(function (n) { posMap[n.id] = { x: n.x, y: n.y }; });

  // Offset for the symbols drawn beside a node: a small share of the plot's extent.
  var xs = nodes.map(function (n) { return n.x; }), ys = nodes.map(function (n) { return n.y; });
  var span = Math.max(Math.max.apply(null, xs) - Math.min.apply(null, xs),
                      Math.max.apply(null, ys) - Math.min.apply(null, ys)) || 1;
  var offset = 0.035 * span;

  function fmt(v, digits) { return (v === null || v === undefined) ? '—' : Number(v).toFixed(digits); }

  // ── Connections: one line trace per state, plus hover points at midpoints ──
  var lines = { closed: { x: [], y: [] }, normally_open: { x: [], y: [] }, out_of_service: { x: [], y: [] } };
  var mid = { x: [], y: [], text: [] };
  var switchClosed = { x: [], y: [], text: [] }, switchOpen = { x: [], y: [], text: [] };
  var regs = { x: [], y: [], text: [] };
  edges.forEach(function (e) {
    var a = posMap[e.from], b = posMap[e.to];
    if (!a || !b) return;
    var state = e.state || (e.open ? 'normally_open' : 'closed');
    lines[state].x.push(a.x, b.x, null);
    lines[state].y.push(a.y, b.y, null);
    var stateLabel = { closed: 'in service', normally_open: 'normally open', out_of_service: 'out of service' }[state];
    var info = '<b>' + (e.id || (e.from + ' – ' + e.to)) + '</b><br>' + e.from + ' → ' + e.to +
      (e.cable_id ? '<br>Cable ' + e.cable_id + ', ' + fmt(e.length, 0) + ' m' : '') +
      '<br>' + (e.is_switch ? 'Switch, ' : '') + stateLabel;
    var mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    if (e.is_switch) {
      var target = state === 'closed' ? switchClosed : switchOpen;
      target.x.push(mx); target.y.push(my); target.text.push(info);
    } else {
      mid.x.push(mx); mid.y.push(my); mid.text.push(info);
    }
    if (e.regulator) {
      var r = e.regulator;
      regs.x.push(a.x + 0.75 * (b.x - a.x));
      regs.y.push(a.y + 0.75 * (b.y - a.y));
      regs.text.push('<b>' + r.regulator_id + '</b> (voltage regulator)<br>On ' + e.id + ', regulates ' + e.to +
        '<br>Set-point ' + fmt(r.v_set_pu, 3) + ' ± ' + fmt(r.deadband_pu / 2, 3) + ' p.u.' +
        '<br>Tap ' + r.tap_pos + ' in [' + r.tap_min + ', ' + r.tap_max + '], ' + r.tap_step_pct + ' %/step');
    }
  });

  var traces = [
    { x: lines.closed.x, y: lines.closed.y, mode: 'lines', line: { width: 2, color: P.slate },
      hoverinfo: 'skip', showlegend: false },
    { x: lines.normally_open.x, y: lines.normally_open.y, mode: 'lines',
      line: { width: 2, color: P.amber, dash: 'dash' }, hoverinfo: 'skip',
      name: 'Tie line (normally open)', showlegend: lines.normally_open.x.length > 0 },
    { x: lines.out_of_service.x, y: lines.out_of_service.y, mode: 'lines',
      line: { width: 2, color: '#8b949e', dash: 'dot' }, hoverinfo: 'skip',
      name: 'Out of service', showlegend: lines.out_of_service.x.length > 0 },
    { x: mid.x, y: mid.y, text: mid.text, mode: 'markers', marker: { size: 12, color: 'rgba(0,0,0,0)' },
      hoverinfo: 'text', showlegend: false }
  ];

  // ── Stems from nodes to their shunt / DER symbols ──
  var stems = { x: [], y: [] };
  var shuntTraces = { cap: { x: [], y: [], text: [] }, reac: { x: [], y: [], text: [] } };
  var derTraces = {};
  nodes.forEach(function (n) {
    var items = [];
    (n.shunts || []).forEach(function (s) { items.push({ kind: 'shunt', v: s }); });
    (n.ders || []).forEach(function (d) { items.push({ kind: 'der', v: d }); });
    items.forEach(function (it, i) {
      // Fan the symbols out above the node, 35° apart.
      var ang = (90 + (i - (items.length - 1) / 2) * 35) * Math.PI / 180;
      var x = n.x + offset * Math.cos(ang), y = n.y + offset * Math.sin(ang);
      stems.x.push(n.x, x, null); stems.y.push(n.y, y, null);
      var off = it.v.active ? '' : '<br><i>out of service</i>';
      if (it.kind === 'shunt') {
        var s = it.v, q = s.q_step_kvar * s.step;
        var t = s.q_step_kvar > 0 ? shuntTraces.cap : shuntTraces.reac;
        t.x.push(x); t.y.push(y);
        t.text.push('<b>' + s.shunt_id + '</b> (' + (s.q_step_kvar > 0 ? 'capacitor bank' : 'reactor') + ') at ' + n.id +
          '<br>' + s.step + ' of ' + s.n_steps + ' steps × ' + Math.abs(s.q_step_kvar) + ' kvar = ' +
          Math.abs(q) + ' kvar' + off);
      } else {
        var d = it.v, type = MV_DER_STYLE[d.type] ? d.type : 'other';
        derTraces[type] = derTraces[type] || { x: [], y: [], text: [] };
        derTraces[type].x.push(x); derTraces[type].y.push(y);
        derTraces[type].text.push('<b>' + d.der_id + '</b> (' + MV_DER_STYLE[type].name + ') at ' + n.id +
          '<br>' + d.s_rated_kva + ' kVA, ' + (MV_CONTROL_LABEL[d.control_mode] || d.control_mode) + off);
      }
    });
  });
  traces.push({ x: stems.x, y: stems.y, mode: 'lines', line: { width: 1, color: P.slate },
    hoverinfo: 'skip', showlegend: false });

  // ── Nodes ──
  var groups = { substation: [], junction: [], connection_point: [], connected_cp: [], cp_out: [] };
  nodes.forEach(function (n) {
    var trs = n.transformers || [];
    if (n.type === 'connection_point' && trs.length > 0 && !trs.some(function (t) { return t.active; })) {
      groups.cp_out.push(n);
    } else if (n.type === 'connection_point' && n.connected_lv && n.connected_lv.length > 0) {
      groups.connected_cp.push(n);
    } else if (n.type === 'connection_point') {
      groups.connection_point.push(n);
    } else if (n.type === 'substation') {
      groups.substation.push(n);
    } else {
      groups.junction.push(n);
    }
  });

  function nodeHover(n) {
    var label = '<b>' + n.id + '</b>';
    if (n.connection_point_id && n.connection_point_id !== n.id) label += ' [' + n.connection_point_id + ']';
    (n.transformers || []).forEach(function (t) {
      label += '<br>' + t.transformer_id + ': ' + t.rated_power_kva + ' kVA, tap ' + (t.tap_pos > 0 ? '+' : '') +
        t.tap_pos + (t.active ? '' : ' (out of service)');
    });
    if (n.connected_lv) label += '<br>→ ' + n.connected_lv.join(', ');
    return label;
  }

  function makeTrace(arr, color, symbol, size, name) {
    return {
      x: arr.map(function (n) { return n.x; }),
      y: arr.map(function (n) { return n.y; }),
      text: arr.map(function (n) { return n.id; }),
      hovertext: arr.map(nodeHover),
      customdata: arr.map(function (n) { return n.connection_point_id || null; }),
      mode: 'markers+text',
      marker: { size: size, color: color, symbol: symbol, line: { width: 1, color: '#ffffff' } },
      textposition: 'bottom center',
      textfont: { size: 10 },
      name: name,
      hoverinfo: 'text',
      meta: 'node-labels',
      showlegend: arr.length > 0
    };
  }

  traces.push(
    makeTrace(groups.substation, P.red, 'square', 16, 'Substation'),
    makeTrace(groups.junction, P.cyan, 'circle', 10, 'Junction'),
    makeTrace(groups.connection_point, P.green, 'diamond', 14, 'Connection Point (free)'),
    makeTrace(groups.connected_cp, P.orange, 'diamond', 14, 'Connection Point (connected)'),
    makeTrace(groups.cp_out, '#8b949e', 'diamond-open', 14, 'Connection Point (transformer out of service)')
  );

  // ── Attached elements, drawn over the lines ──
  function symbolTrace(t, symbol, color, size, name, hollow) {
    return {
      x: t.x, y: t.y, hovertext: t.text, mode: 'markers', hoverinfo: 'text', name: name,
      marker: { size: size, symbol: symbol, color: hollow ? 'rgba(0,0,0,0)' : color,
                line: { width: hollow ? 2 : 1, color: hollow ? color : '#ffffff' } },
      showlegend: t.x.length > 0
    };
  }
  traces.push(
    symbolTrace(switchClosed, 'square', P.slate, 9, 'Switch (closed)'),
    symbolTrace(switchOpen, 'square', P.amber, 9, 'Switch (open)', true),
    symbolTrace(regs, 'hourglass', P.purple, 13, 'Voltage regulator'),
    symbolTrace(shuntTraces.cap, 'triangle-up', '#58A6FF', 11, 'Capacitor bank'),
    symbolTrace(shuntTraces.reac, 'triangle-down', P.pink, 11, 'Reactor')
  );
  Object.keys(MV_DER_STYLE).forEach(function (type) {
    if (derTraces[type]) {
      var st = MV_DER_STYLE[type];
      traces.push(symbolTrace(derTraces[type], st.symbol, st.color, 13, 'DER: ' + st.name));
    }
  });

  var layout = GA.chartLayout({
    showlegend: true,
    legend: { orientation: 'h', y: -0.05 },
    hovermode: 'closest',
    xaxis: { visible: false },
    yaxis: { visible: false, scaleanchor: 'x' },
    margin: { l: 10, r: 10, t: 10, b: 40 }
  });

  Plotly.newPlot(containerId, traces, layout, { responsive: true });

  if (onClickCP) {
    document.getElementById(containerId).on('plotly_click', function (data) {
      if (data.points && data.points.length > 0) {
        var pt = data.points[0];
        var cpId = pt.customdata;
        if (cpId) {
          onClickCP(cpId);
        }
      }
    });
  }
}

/** Indices of the node traces (the ones whose labels the "Show node labels" toggle hides). */
function mvGraphLabelTraces(containerId) {
  var gd = document.getElementById(containerId);
  return (gd.data || []).map(function (t, i) { return t.meta === 'node-labels' ? i : -1; })
    .filter(function (i) { return i >= 0; });
}

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
