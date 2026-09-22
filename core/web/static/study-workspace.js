/* Presentation only. Study identity, answers and persistence belong to each study. */
(function (root) {
  'use strict';
  function appearance(payload = {}) {
    const value = Number(payload.scale);
    return {
      theme: payload.theme === 'dark' ? 'dark' : 'light',
      scale: Number.isFinite(value) && payload.scale !== null && payload.scale !== ''
        ? Math.min(1.5, Math.max(.8, value)) : 1,
    };
  }
  function wheelDelta(event, viewportHeight = 0) {
    const raw = Number(event && event.deltaY);
    if (!Number.isFinite(raw) || raw === 0) return 0;
    if (event.deltaMode === 1) return raw * 18;
    if (event.deltaMode === 2) return raw * Math.max(1, viewportHeight || 0);
    return raw;
  }
  function scrollWithWheel(scroller, event) {
    if (!scroller || event.ctrlKey || event.metaKey) return false;
    const maximum = Math.max(0, Number(scroller.scrollHeight || 0) - Number(scroller.clientHeight || 0));
    const delta = wheelDelta(event, scroller.clientHeight);
    if (!maximum || !delta) return false;
    const before = Math.max(0, Math.min(maximum, Number(scroller.scrollTop || 0)));
    const after = Math.max(0, Math.min(maximum, before + delta));
    if (after === before) return false;
    scroller.scrollTop = after;
    if (typeof event.preventDefault === 'function') event.preventDefault();
    return true;
  }
  function mount(win) {
    const doc = win.document, element = doc.documentElement;
    const params = new URLSearchParams(win.location.search);
    const embedded = params.get('embedded') === '1' && win.parent !== win;
    element.dataset.embedded = String(embedded);
    element.dataset.studyView = params.get('view') === 'results' ? 'results' : 'review';
    function applyAppearance(payload) {
      const next = appearance(payload);
      if (payload.theme != null) element.dataset.theme = next.theme;
      if (payload.scale !== undefined) element.style.setProperty('--study-text-scale', String(next.scale));
    }
    applyAppearance({theme: params.get('theme') || 'light', scale: params.get('textScale')});
    doc.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      const info = doc.querySelector('.study-session-info[open]');
      if (info) { info.open = false; info.querySelector('summary')?.focus(); }
    });
    doc.addEventListener('pointerdown', event => {
      const info = doc.querySelector('.study-session-info[open]');
      if (info && !info.contains(event.target)) info.open = false;
    });
    if (embedded) {
      // Chromium normally scrolls these panes itself.  Electron can instead
      // route a wheel gesture to the fixed iframe viewport, whose overflow is
      // intentionally hidden.  Delegate the gesture to the nearest ancestor
      // that can actually move, chaining outward like native scrolling, and
      // keep overscroll contained when nothing under the cursor can move.
      doc.addEventListener('wheel', event => {
        if (event.ctrlKey || event.metaKey) return;
        if (!wheelDelta(event, 1)) return;
        let node = event.target;
        if (node && node.nodeType !== 1) node = node.parentElement;
        while (node && node.nodeType === 1) {
          const style = typeof win.getComputedStyle === 'function' ? win.getComputedStyle(node) : null;
          const overflowY = style ? style.overflowY : '';
          if (/^(auto|scroll|overlay)$/.test(overflowY)
              && Number(node.scrollHeight || 0) > Number(node.clientHeight || 0) + 1
              && scrollWithWheel(node, event)) return;
          node = node.parentElement;
        }
        if (typeof event.preventDefault === 'function') event.preventDefault();
      }, {passive: false, capture: true});
    }
    function resetReadingPosition() {
      // Navigation already waits for a successful save. Reset only the reading
      // panes; appearance/language updates must never reset scroll or inputs.
      if (!embedded) return;
      doc.querySelectorAll('#app, #material, .questions').forEach(pane => { pane.scrollTop = 0; });
    }
    return {applyAppearance, resetReadingPosition};
  }
  const api = {appearance, wheelDelta, scrollWithWheel, mount};
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.StudyWorkspace = {...api, ...mount(root)};
})(typeof window === 'undefined' ? globalThis : window);
