// Shared logic of the benchmark pages: upload a script, follow its evaluation,
// list past submissions. Each page calls GridArenaSubmissions.init() with how
// to read its settings and how to display its score (see benchmark-*.js).
window.GridArenaSubmissions = (function () {

  // Build table rows with textContent only: names and errors come from users.
  function row(cells, header) {
    var tr = document.createElement('tr');
    cells.forEach(function (c) {
      var cell = document.createElement(header ? 'th' : 'td');
      if (c instanceof Node) cell.appendChild(c); else cell.textContent = (c === null || c === undefined) ? '—' : c;
      tr.appendChild(cell);
    });
    return tr;
  }

  function pct(x) { return (x === null || x === undefined) ? '—' : (x * 100).toFixed(2) + '%'; }
  function num(x, digits) { return (x === null || x === undefined) ? '—' : Number(x).toFixed(digits === undefined ? 4 : digits); }

  async function errorText(res) {
    var text = await res.text();
    try { var d = JSON.parse(text).detail; return typeof d === 'string' ? d : JSON.stringify(d); }
    catch (e) { return text; }
  }

  // Fetch a JSON endpoint and save the response as a file.
  async function downloadJson(url, filename, msgEl) {
    msgEl.textContent = 'Preparing download...';
    try {
      var res = await fetch(url);
      if (!res.ok) throw new Error(await errorText(res));
      var data = await res.json();
      var blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      var blobUrl = window.URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = blobUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(blobUrl);
      msgEl.textContent = 'Download started.';
    } catch (e) {
      msgEl.textContent = 'Download failed: ' + e.message;
    }
  }

  // cfg: {
  //   api: '/x_benchmark',
  //   settings: function () -> {field: value}      form fields sent with the upload
  //   describeSettings: function (settings) -> str  shown in messages and the history
  //   headline: function (score) -> str            summary card + history column
  //   headlineLabel: str                           history column title
  //   summaryRows: function (score) -> [[label, value], ...]
  //   detailHeader: [..], detailRows: function (score) -> [[..], ..]
  // }
  function init(cfg) {
    var submissionName = document.getElementById('submissionName');

    // ── Step 2: Submit ──

    var scriptFile = document.getElementById('scriptFile');
    var scriptDrop = document.getElementById('scriptDrop');
    var scriptLabel = document.getElementById('scriptFileLabel');
    var btnSubmit = document.getElementById('btnSubmit');
    var uploadMsg = document.getElementById('uploadMsg');

    scriptDrop.addEventListener('click', function (e) { e.stopPropagation(); scriptFile.value = ''; scriptFile.click(); });
    ['dragenter', 'dragover'].forEach(function (ev) {
      scriptDrop.addEventListener(ev, function (e) { e.preventDefault(); scriptDrop.classList.add('is-over'); });
    });
    ['dragleave', 'dragend'].forEach(function (ev) {
      scriptDrop.addEventListener(ev, function (e) { e.preventDefault(); scriptDrop.classList.remove('is-over'); });
    });
    scriptDrop.addEventListener('drop', function (e) {
      e.preventDefault(); scriptDrop.classList.remove('is-over');
      if (e.dataTransfer && e.dataTransfer.files.length) {
        scriptFile.files = e.dataTransfer.files;
        onFileSelected();
      }
    });
    scriptFile.addEventListener('change', onFileSelected);

    function onFileSelected() {
      var f = scriptFile.files[0];
      scriptLabel.textContent = f ? f.name : 'No file selected';
      btnSubmit.disabled = !f;
      uploadMsg.textContent = '';
    }

    btnSubmit.addEventListener('click', async function () {
      var name = submissionName.value.trim();
      uploadMsg.textContent = '';

      if (!name) { uploadMsg.textContent = 'Please enter a Submission Name in Benchmark Setup.'; return; }
      if (!scriptFile.files.length) { uploadMsg.textContent = 'Please select a script.'; return; }

      var settings;
      try { settings = cfg.settings(); } catch (e) { uploadMsg.textContent = e.message; return; }

      btnSubmit.disabled = true;
      uploadMsg.textContent = 'Uploading...';

      try {
        var form = new FormData();
        form.append('file', scriptFile.files[0]);
        form.append('submission_name', name);
        Object.keys(settings).forEach(function (k) { form.append(k, settings[k]); });

        var res = await fetch(cfg.api + '/submissions', { method: 'POST', body: form });
        if (!res.ok) throw new Error(await errorText(res));
        var record = await res.json();

        uploadMsg.textContent = 'Submitted as ' + record.submission_id + '.';
        watch(record.submission_id);
        loadHistory();
      } catch (e) {
        uploadMsg.textContent = 'Error: ' + e.message;
      } finally {
        btnSubmit.disabled = !scriptFile.files.length;
      }
    });

    // ── Step 3: Results ──

    var runMsg = document.getElementById('runMsg');
    var scoreResult = document.getElementById('scoreResult');
    var scoreTable = document.getElementById('scoreTable');
    var gridTable = document.getElementById('gridTable');
    var runError = document.getElementById('runError');
    var runErrorMsg = document.getElementById('runErrorMsg');
    var logBox = document.getElementById('logBox');
    var logText = document.getElementById('logText');
    var scStatus = document.getElementById('scStatus');
    var scScore = document.getElementById('scScore');
    var scRuntime = document.getElementById('scRuntime');
    var pollTimer = null;
    var watchedId = null;

    function watch(id) {
      watchedId = id;
      if (pollTimer) clearTimeout(pollTimer);
      poll();
    }

    async function poll() {
      var id = watchedId;
      try {
        var res = await fetch(cfg.api + '/submissions/' + encodeURIComponent(id));
        if (!res.ok) throw new Error(await errorText(res));
        var record = await res.json();
        if (id !== watchedId) return;  // another submission was selected meanwhile
        render(record);
        if (record.status === 'queued' || record.status === 'running') {
          pollTimer = setTimeout(poll, 2000);
        } else {
          loadHistory();
        }
      } catch (e) {
        runMsg.textContent = 'Could not load submission ' + id + ': ' + e.message;
      }
    }

    function render(record) {
      runMsg.textContent = 'Submission ' + record.submission_id + ' (' + record.submission_name + ', '
        + cfg.describeSettings(record.settings || {}) + ', ' + record.filename + '): ' + record.status + '.';
      if (scStatus) scStatus.textContent = record.status;

      scoreResult.style.display = 'none';
      runError.style.display = 'none';

      if (record.status === 'completed' && record.score) {
        var s = record.score;
        scoreResult.style.display = 'block';
        scoreTable.replaceChildren.apply(scoreTable, cfg.summaryRows(s).map(function (r) { return row(r); }));
        gridTable.replaceChildren(row(cfg.detailHeader, true));
        cfg.detailRows(s).forEach(function (r) { gridTable.appendChild(row(r)); });
        if (scScore) scScore.textContent = cfg.headline(s);
        if (scRuntime) scRuntime.textContent = s.runtime_seconds + ' s';
      } else {
        if (record.status === 'failed') {
          runError.style.display = 'block';
          runErrorMsg.textContent = record.error;
        }
        if (scScore) scScore.textContent = '—';
        if (scRuntime) scRuntime.textContent = '—';
      }

      logBox.style.display = record.log_tail ? 'block' : 'none';
      logText.textContent = record.log_tail || '';
    }

    // ── Step 4: History ──

    var historyTable = document.getElementById('historyTable');
    var onlyMine = document.getElementById('onlyMine');

    async function loadHistory() {
      var url = cfg.api + '/submissions';
      var name = submissionName.value.trim();
      if (onlyMine.checked && name) url += '?submission_name=' + encodeURIComponent(name);
      var header = ['Submitted', 'Name', 'Settings', 'Status', cfg.headlineLabel, 'ID'];
      try {
        var res = await fetch(url);
        if (!res.ok) throw new Error(await errorText(res));
        var data = await res.json();
        historyTable.replaceChildren(row(header, true));
        data.submissions.forEach(function (r) {
          var tr = row([
            new Date(r.created_at).toLocaleString(), r.submission_name, cfg.describeSettings(r.settings || {}),
            r.status, r.score ? cfg.headline(r.score) : '—', r.submission_id
          ]);
          tr.style.cursor = 'pointer';
          tr.addEventListener('click', function () { watch(r.submission_id); });
          historyTable.appendChild(tr);
        });
        if (!data.submissions.length) historyTable.appendChild(row(['No submissions yet.', '', '', '', '', '']));
      } catch (e) {
        historyTable.replaceChildren(row(['Could not load submissions: ' + e.message]));
      }
    }

    document.getElementById('btnRefresh').addEventListener('click', loadHistory);
    onlyMine.addEventListener('change', loadHistory);
    loadHistory();
  }

  return { init: init, downloadJson: downloadJson, pct: pct, num: num };
})();
