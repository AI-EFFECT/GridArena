/* Stop buttons for background training runs (RL and diffusion).
   Any element with data-cancel-url POSTs there after confirmation. */
(function () {
  document.addEventListener('click', async function (ev) {
    var btn = ev.target.closest('[data-cancel-url]');
    if (!btn) return;
    ev.preventDefault();
    if (!confirm('Stop this run? Progress so far is discarded.')) return;

    btn.disabled = true;
    var label = btn.textContent;
    btn.textContent = 'Stopping…';
    try {
      var res = await fetch(btn.getAttribute('data-cancel-url'), { method: 'POST' });
      var data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Could not stop the run');
      location.reload();
    } catch (e) {
      alert(e.message);
      btn.disabled = false;
      btn.textContent = label;
    }
  });
})();
