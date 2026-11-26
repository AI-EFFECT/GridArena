(function () {
  var API = '/voltage_control_benchmark';
  var S = window.GridArenaSubmissions;
  var voltageLimit = document.getElementById('voltageLimit');
  var scenarioSelect = document.getElementById('scenarioSelect');
  var difficultySelect = document.getElementById('difficultySelect');

  function showSettings() {
    document.getElementById('scSettings').textContent = scenarioSelect.value + ' / ' + difficultySelect.value;
  }
  scenarioSelect.addEventListener('change', showSettings);
  difficultySelect.addEventListener('change', showSettings);

  function limit() {
    var v = parseFloat(voltageLimit.value);
    if (!(v > 0)) throw new Error('Please enter a Voltage Limit (V) in Benchmark Setup.');
    return v;
  }

  document.getElementById('btnDownloadTrain').addEventListener('click', function () {
    var msg = document.getElementById('downloadMsg');
    var v;
    try { v = limit(); } catch (e) { msg.textContent = e.message; return; }
    S.downloadJson(API + '/training-scenarios?voltage_limit=' + encodeURIComponent(v)
      + '&Scenario=' + encodeURIComponent(scenarioSelect.value)
      + '&difficulty=' + encodeURIComponent(difficultySelect.value),
      'training_voltage_control_' + scenarioSelect.value + '_' + difficultySelect.value + '.json', msg);
  });

  S.init({
    api: API,
    settings: function () {
      return { voltage_limit: limit(), Scenario: scenarioSelect.value, difficulty: difficultySelect.value };
    },
    describeSettings: function (s) { return s.scenario + ' > ' + s.voltage_limit + ' V, ' + s.difficulty; },
    headline: function (score) { return S.num(score.score); },
    headlineLabel: 'Score',
    summaryRows: function (s) {
      return [
        ['Score (0-1, higher is better)', S.num(s.score)],
        ['Snapshots', s.snapshots + ' on ' + s.grids + ' grids' + (s.failed_snapshots ? ' (' + s.failed_snapshots + ' failed)' : '')],
        ['Scored nodes', s.nodes],
        ['Model runtime', s.runtime_seconds + ' s']
      ];
    },
    detailHeader: ['Grid', 'Datetime', 'Nodes', 'Changed', 'Mode', 'Score', 'Ref. V MAE', 'Effort', 'Error'],
    detailRows: function (s) {
      return s.per_snapshot.map(function (r) {
        return [r.grid_id, r.datetime, r.nodes, r.changed_nodes, r.mode, S.num(r.timestamp_score),
          S.num(r.reference_voltage_mae, 3), S.num(r.effort_norm, 3), r.error || ''];
      });
    }
  });
})();
