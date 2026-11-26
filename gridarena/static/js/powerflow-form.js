(function () {
  var form = document.getElementById('pfForm');
  var msg = document.getElementById('msg');
  var resultCard = document.getElementById('resultCard');
  var resultMsg = document.getElementById('resultMsg');
  var resultLink = document.getElementById('resultLink');
  var mode = document.getElementById('mode');
  var neutralField = document.getElementById('neutralField');

  // The neutral only matters in the unbalanced model.
  function syncNeutral() { neutralField.style.display = mode.value === 'unbalanced' ? '' : 'none'; }
  mode.addEventListener('change', syncNeutral);
  form.addEventListener('reset', function () { setTimeout(syncNeutral, 0); });
  syncNeutral();

  form.addEventListener('submit', async function (e) {
    e.preventDefault();

    var grid = document.getElementById('gridId').value;
    var ratio = document.getElementById('neutralRatio').value;
    var start = document.getElementById('start').value;
    var end = document.getElementById('end').value;

    if (!grid) { msg.textContent = 'Please select a grid.'; return; }
    if (mode.value === 'unbalanced' && !(parseFloat(ratio) >= 0)) {
      msg.textContent = 'The neutral Z ratio must be a number >= 0.'; return;
    }

    var url = new URL('/powerflow/' + encodeURIComponent(grid) + '/run', window.location.origin);
    url.searchParams.append('mode', mode.value);
    if (mode.value === 'unbalanced') url.searchParams.append('neutral_impedance_ratio', ratio);
    if (start) url.searchParams.append('start_time', start);
    if (end) url.searchParams.append('end_time', end);

    msg.textContent = 'Running power flow...';
    resultCard.style.display = 'none';
    document.getElementById('runBtn').disabled = true;

    try {
      var res = await fetch(url, { method: 'POST' });
      var j = await res.json();
      if (!res.ok) throw new Error(typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail || j));

      var notes = [];
      if (j.not_converged) notes.push(j.not_converged + ' did not converge');
      if (j.skipped_timestamps) notes.push(j.skipped_timestamps + ' skipped (no substation voltage)');
      if (j.unreachable_nodes.length) notes.push('not connected to PT: ' + j.unreachable_nodes.join(', '));

      msg.textContent = '';
      resultCard.style.display = 'block';
      resultMsg.textContent = 'Grid: ' + grid + ' | ' + (j.mode === 'unbalanced'
          ? 'Unbalanced, neutral Z ratio ' + j.neutral_impedance_ratio : 'Balanced')
        + ' | ' + j.timestamps + ' timestamps solved' + (notes.length ? ' (' + notes.join('; ') + ')' : '');
      resultLink.href = '/powerflow/ui/' + encodeURIComponent(grid);
      resultLink.style.display = 'inline-block';
    } catch (err) {
      msg.textContent = 'Error: ' + err.message;
      resultCard.style.display = 'none';
    } finally {
      document.getElementById('runBtn').disabled = false;
    }
  });
})();
