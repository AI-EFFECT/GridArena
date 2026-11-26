(function () {
  var API = '/topology_discovery_benchmark';
  var S = window.GridArenaSubmissions;
  var difficultySelect = document.getElementById('difficultySelect');

  difficultySelect.addEventListener('change', function () {
    document.getElementById('scDifficulty').textContent = difficultySelect.value;
  });

  document.getElementById('btnDownloadTrain').addEventListener('click', function () {
    var diff = difficultySelect.value;
    S.downloadJson(API + '/training-topology-data?difficulty=' + encodeURIComponent(diff),
      'training_topology_data_' + diff + '.json', document.getElementById('downloadMsg'));
  });

  S.init({
    api: API,
    settings: function () { return { difficulty: difficultySelect.value }; },
    describeSettings: function (s) { return s.difficulty || ''; },
    headline: function (score) { return S.pct(score.f1); },
    headlineLabel: 'F1',
    summaryRows: function (s) {
      return [
        ['F1 score', S.pct(s.f1)],
        ['Precision', S.pct(s.precision) + ' (' + s.correct + ' of ' + (s.correct + s.extra) + ' predicted edges correct)'],
        ['Recall', S.pct(s.recall) + ' (' + s.correct + ' of ' + (s.correct + s.missed) + ' true edges found)'],
        ['Test grids', s.grids + (s.failed_grids ? ' (' + s.failed_grids + ' failed)' : '')],
        ['Model runtime', s.runtime_seconds + ' s']
      ];
    },
    detailHeader: ['Grid', 'Nodes', 'True edges', 'Predicted', 'Correct', 'F1', 'Invalid', 'Time (s)', 'Error'],
    detailRows: function (s) {
      return s.per_grid.map(function (g) {
        return [g.grid_id, g.nodes, g.true_edges, g.predicted_edges, g.correct, S.pct(g.f1), g.invalid,
          g.seconds, g.error || ''];
      });
    }
  });
})();
