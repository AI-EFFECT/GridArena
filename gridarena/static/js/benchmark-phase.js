(function () {
  var API = '/phase_identification_benchmark';
  var S = window.GridArenaSubmissions;
  var difficultySelect = document.getElementById('difficultySelect');

  difficultySelect.addEventListener('change', function () {
    document.getElementById('scDifficulty').textContent = difficultySelect.value;
  });

  document.getElementById('btnDownloadTrain').addEventListener('click', function () {
    var diff = difficultySelect.value;
    S.downloadJson(API + '/training-phase-data?difficulty=' + encodeURIComponent(diff),
      'training_phase_data_' + diff + '.json', document.getElementById('downloadMsg'));
  });

  S.init({
    api: API,
    settings: function () { return { difficulty: difficultySelect.value }; },
    describeSettings: function (s) { return s.difficulty || ''; },
    headline: function (score) { return S.pct(score.accuracy); },
    headlineLabel: 'Accuracy',
    summaryRows: function (s) {
      return [
        ['Accuracy', S.pct(s.accuracy)],
        ['Correct', s.correct + ' / ' + s.total + ' series'],
        ['Test grids', s.grids + (s.failed_grids ? ' (' + s.failed_grids + ' failed)' : '')],
        ['Model runtime', s.runtime_seconds + ' s']
      ];
    },
    detailHeader: ['Grid', 'Series', 'Accuracy', 'Missing', 'Invalid', 'Time (s)', 'Error'],
    detailRows: function (s) {
      return s.per_grid.map(function (g) {
        return [g.grid_id, g.series, S.pct(g.accuracy), g.missing, g.invalid, g.seconds, g.error || ''];
      });
    }
  });
})();
