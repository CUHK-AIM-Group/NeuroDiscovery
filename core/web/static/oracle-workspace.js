/* Presentation bridge only. No graph mutations, model calls or evidence scoring. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.NeuroOracleWorkspace = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  let hooks, mounted = false;
  const byId = id => document.getElementById(id);
  function appearance(value = {}) {
    const result = {};
    if (['light', 'dark'].includes(value.theme)) result.theme = value.theme;
    if (value.language === 'zh' || value.language === 'en') result.language = value.language;
    const scale = Number(value.scale);
    if (value.scale != null && Number.isFinite(scale)) result.scale = Math.max(.8, Math.min(1.5, scale));
    return result;
  }
  function trustedMessage(event, origin, parent) {
    return event.origin === origin && event.source === parent &&
      ['neurodiscovery:appearance', 'neurodiscovery:theme', 'neuroclaw:text-scale'].includes(event.data?.type);
  }
  const tr = (en, zh) => hooks?.language() === 'zh' ? zh : en;
  function resized() { requestAnimationFrame(() => window.dispatchEvent(new Event('resize'))); }
  function browse(open, focus = false) {
    document.body.dataset.oracleBrowse = open ? 'open' : 'closed';
    byId('oracleBrowseToggle').setAttribute('aria-expanded', String(open));
    if (open && focus) (byId(document.body.classList.contains('claim-evidence-active') ? 'ceSearch' : 'searchInput'))?.focus();
    resized();
  }
  function inspector(open) {
    document.body.dataset.oracleInspector = open ? 'open' : 'closed';
    byId('oracleInspectorToggle').setAttribute('aria-expanded', String(open));
    byId('oracleInspector').inert = !open;
    resized();
  }
  function revealInspector() {
    inspector(true);
    if (window.matchMedia('(max-width: 760px)').matches) { browse(false); byId('oracleInspectorClose').focus({preventScroll:true}); }
  }
  function revealEvidence() {
    if (window.matchMedia('(max-width: 760px)').matches) { browse(false); byId('ceDetail')?.focus(); }
  }
  function translate() {
    byId('oracleBrowseLabel').textContent = tr('Browse', '检索');
    byId('oracleBrowseToggle').title = tr('Show or hide the search panel', '展开或收起检索栏');
    byId('oracleBrowseToggle').setAttribute('aria-label', byId('oracleBrowseToggle').title);
    byId('oracleInspectorToggle').textContent = tr('Details', '详情');
    byId('oracleInspectorLabel').textContent = tr('Evidence & details', '证据与详情');
    byId('oracleInspectorClose').setAttribute('aria-label', tr('Close details', '关闭详情'));
  }
  function mount(options) {
    if (mounted) return;
    mounted = true; hooks = options;
    const params = new URLSearchParams(location.search);
    document.documentElement.dataset.embedded = String(window.parent !== window && params.get('embedded') === '1');
    const apply = value => {
      const next = appearance(value);
      if (next.theme) document.documentElement.dataset.theme = next.theme;
      if (next.scale != null) document.documentElement.style.setProperty('--text-scale', next.scale);
      hooks.onAppearance?.(next); translate(); resized();
    };
    apply({theme:params.get('theme'), language:params.get('lang'), scale:params.get('scale')});
    browse(true); inspector(false);
    byId('oracleBrowseToggle').addEventListener('click', () => browse(document.body.dataset.oracleBrowse !== 'open', true));
    byId('oracleInspectorToggle').addEventListener('click', () => inspector(document.body.dataset.oracleInspector !== 'open'));
    byId('oracleInspectorClose').addEventListener('click', () => { inspector(false); byId('oracleInspectorToggle').focus(); });
    document.addEventListener('click', event => {
      const menu = byId('oracleWorkspaceOptions');
      if (menu.open && !menu.contains(event.target)) menu.open = false;
    });
    window.addEventListener('message', event => {
      if (window.parent !== window && trustedMessage(event, location.origin, window.parent)) apply(event.data);
    });
    document.addEventListener('keydown', event => {
      if (event.key !== 'Escape' || document.querySelector('dialog[open]')) return;
      const menu = byId('oracleWorkspaceOptions');
      const options = byId('oracleGraphOptions');
      if (menu.open) { menu.open = false; menu.querySelector('summary').focus(); }
      else if (options.open) { options.open = false; options.querySelector('summary').focus(); }
      else if (document.body.dataset.oracleInspector === 'open') { inspector(false); byId('oracleInspectorToggle').focus(); }
    });
    new MutationObserver(translate).observe(document.documentElement, {attributes:true, attributeFilter:['lang']});
  }
  return {appearance, trustedMessage, mount, revealInspector, revealEvidence};
});
