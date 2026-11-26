(function () {
  var API = '/state_estimation_benchmark';
  var S = window.GridArenaSubmissions;
  var noiseSelect = document.getElementById('noiseSelect');
  var obsSelect = document.getElementById('obsSelect');

  function showSettings() {
    document.getElementById('scSettings').textContent = noiseSelect.value + ' / ' + obsSelect.value;
  }
  noiseSelect.addEventListener('change', showSettings);
  obsSelect.addEventListener('change', showSettings);

  document.getElementById('btnDownloadTrain').addEventListener('click', function () {
    var noise = noiseSelect.value;
    S.downloadJson(API + '/training-state-data?noise_difficulty=' + encodeURIComponent(noise),
      'training_state_data_' + noise + '.json', document.getElementById('downloadMsg'));
  });

  S.init({
    api: API,
    settings: function () { return { noise_difficulty: noiseSelect.value, observability: obsSelect.value }; },
    describeSettings: function (s) { return 'noise ' + s.noise_difficulty + ', observability ' + s.observability; },
    headline: function (score) { return S.num(score.score); },
    headlineLabel: 'Score',
    summaryRows: function (s) {
      return [
        ['Score (lower is better)', S.num(s.score, 6)],
        ['Voltage magnitude RMSE', S.num(s.rmse_voltage_magnitude)],
        ['Voltage angle MAE', S.num(s.mae_voltage_angle)],
        ['Estimates', s.estimates + (s.missing || s.invalid ? ' (' + s.missing + ' missing, ' + s.invalid + ' invalid)' : '')],
        ['Tasks', s.tasks + ' on ' + s.grids + ' grids' + (s.failed_tasks ? ' (' + s.failed_tasks + ' failed)' : '')],
        ['Model runtime', s.runtime_seconds + ' s']
      ];
    },
    detailHeader: ['Grid', 'Tasks', 'Estimates', 'Score', '|V| RMSE', 'Angle MAE', 'Missing', 'Invalid', 'Time (s)', 'Error'],
    detailRows: function (s) {
      return s.per_grid.map(function (g) {
        return [g.grid_id, g.tasks, g.estimates, S.num(g.score, 6), S.num(g.rmse_voltage_magnitude),
          S.num(g.mae_voltage_angle), g.missing, g.invalid, g.seconds, g.error || ''];
      });
    }
  });
})();
