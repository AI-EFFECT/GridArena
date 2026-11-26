/* App shell: theme persistence, navbar dropdowns and Plotly theming.
   Loaded synchronously in <head> so the theme is applied before first paint
   and window.GA exists before any page script draws a chart. */
(function () {
  var THEME_KEY = 'gridarena-theme';
  var root = document.documentElement;

  function storedTheme() {
    try { return localStorage.getItem(THEME_KEY) === 'light' ? 'light' : 'dark'; }
    catch (e) { return 'dark'; }
  }

  root.setAttribute('data-theme', storedTheme());

  function currentTheme() {
    return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  // ── Charts ──────────────────────────────────────────────────────────
  // Series colours are fixed across themes (tef-services RESULTS_PALETTE);
  // only chart chrome (backgrounds, gridlines, text) follows the theme.
  var palette = {
    cyan: '#64C8EB', green: '#3FB950', orange: '#F0883E', purple: '#BC8CFF',
    teal: '#39C5CF', pink: '#F778BA', lavender: '#D2A8FF', red: '#E5484D',
    amber: '#FFA657', lime: '#7EE787', slate: '#6a90a8'
  };
  var series = [palette.cyan, palette.green, palette.orange, palette.purple,
    palette.teal, palette.pink, palette.lavender, palette.red, palette.amber, palette.lime];

  function chrome() {
    return {
      plot: cssVar('--chart-plot-bg'),
      text: cssVar('--text2'),
      strong: cssVar('--text'),
      faint: cssVar('--text3'),
      accent: cssVar('--accent'),
      grid: cssVar('--chart-grid'),
      zero: cssVar('--chart-zero'),
      line: cssVar('--border'),
      hoverBg: cssVar('--bg2'),
      hoverBorder: cssVar('--border-hover')
    };
  }

  function axisKeys(layout) {
    var keys = Object.keys(layout).filter(function (k) { return /^[xy]axis\d*$/.test(k); });
    if (keys.indexOf('xaxis') === -1) keys.push('xaxis');
    if (keys.indexOf('yaxis') === -1) keys.push('yaxis');
    return keys;
  }

  // Flat "dotted path" -> value map of every themed property, usable both to
  // build a fresh layout and as a Plotly.relayout() update.
  function themeUpdates(layout) {
    var c = chrome();
    var u = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: c.plot,
      'font.color': c.text,
      'font.family': "'DM Sans', sans-serif",
      'title.font.color': c.strong,
      'legend.bgcolor': 'rgba(0,0,0,0)',
      'legend.font.color': c.text,
      'hoverlabel.bgcolor': c.hoverBg,
      'hoverlabel.bordercolor': c.hoverBorder,
      'hoverlabel.font.color': c.strong,
      'hoverlabel.font.family': "'Space Mono', monospace",
      'modebar.bgcolor': 'rgba(0,0,0,0)',
      'modebar.color': c.faint,
      'modebar.activecolor': c.accent
    };
    axisKeys(layout).forEach(function (k) {
      u[k + '.gridcolor'] = c.grid;
      u[k + '.zerolinecolor'] = c.zero;
      u[k + '.linecolor'] = c.line;
    });
    return u;
  }

  function setPath(obj, path, value) {
    var parts = path.split('.');
    var o = obj;
    for (var i = 0; i < parts.length - 1; i++) {
      var p = parts[i];
      // A string title becomes { text } so its font colour can be set
      if (typeof o[p] === 'string') o[p] = { text: o[p] };
      else if (!o[p] || typeof o[p] !== 'object') o[p] = {};
      else o[p] = Object.assign({}, o[p]);
      o = o[p];
    }
    o[parts[parts.length - 1]] = value;
  }

  function chartLayout(layout) {
    var out = Object.assign({}, layout || {});
    if (!out.colorway) out.colorway = series;
    var u = themeUpdates(out);
    Object.keys(u).forEach(function (path) { setPath(out, path, u[path]); });
    return out;
  }

  function rethemeCharts() {
    if (!window.Plotly) return;
    document.querySelectorAll('.js-plotly-plot').forEach(function (gd) {
      try { window.Plotly.relayout(gd, themeUpdates(gd.layout || {})); } catch (e) { /* chart mid-teardown */ }
    });
  }

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (e) { /* storage disabled */ }
    rethemeCharts();
    document.dispatchEvent(new CustomEvent('gridarena:themechange', { detail: { theme: theme } }));
  }

  window.GA = {
    theme: currentTheme,
    setTheme: setTheme,
    chartLayout: chartLayout,
    palette: palette,
    series: series
  };

  // ── Navbar ──────────────────────────────────────────────────────────
  function closeMenus(except) {
    document.querySelectorAll('.nav-menu.open').forEach(function (m) {
      if (m !== except) {
        m.classList.remove('open');
        var t = m.querySelector('[data-nav-trigger]');
        if (t) t.setAttribute('aria-expanded', 'false');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('[data-nav-trigger]').forEach(function (trigger) {
      trigger.addEventListener('click', function (e) {
        e.stopPropagation();
        var menu = trigger.closest('.nav-menu');
        var open = !menu.classList.contains('open');
        closeMenus(menu);
        menu.classList.toggle('open', open);
        trigger.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
    });

    document.addEventListener('click', function (e) {
      if (!e.target.closest('.nav-menu')) closeMenus(null);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeMenus(null);
    });

    var toggle = document.getElementById('themeToggle');
    if (toggle) {
      toggle.addEventListener('click', function () {
        setTheme(currentTheme() === 'light' ? 'dark' : 'light');
      });
    }
  });
})();
