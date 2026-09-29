/* One code-block toolbar for streamed messages, history and file previews. */
(function () {
  'use strict';
  const key = 'neurodiscovery.code-wrap';
  let wrapped = false;
  try { wrapped = localStorage.getItem(key) === 'true'; } catch (_) {}
  const zh = () => document.documentElement.lang.startsWith('zh');
  const label = (en, cn) => zh() ? cn : en;
  const icon = path => '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + path + '</svg>';
  const copyIcon = icon('<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V4H4v12h4"/>');
  function refresh(panel) {
    panel.classList.toggle('is-wrapped', wrapped);
    const wrap = panel.querySelector('.code-wrap'), copy = panel.querySelector('.code-copy');
    wrap.setAttribute('aria-pressed', String(wrapped));
    wrap.title = wrap.ariaLabel = label('Word wrap', '自动换行');
    wrap.innerHTML = icon('<path d="' + (wrapped ? 'M12 3v5m0 8v5M3 12h18m-4-4 4 4-4 4' : 'M21 3v18M3 7h8a4 4 0 0 1 0 8H3m4-4-4 4 4 4') + '"/>');
    const state = panel._codeState;
    const feedback = state.until > Date.now() ? state.feedback : '';
    copy.title = copy.ariaLabel = feedback === 'copied' ? label('Code copied', '已复制代码') : feedback === 'error' ? label('Copy failed. Try again.', '复制失败，请重试') : label('Copy code', '复制代码');
    copy.innerHTML = feedback === 'copied' ? icon('<path d="m5 12 4 4L19 6"/>') : copyIcon;
    panel.querySelector('.code-feedback').textContent = feedback ? copy.title : '';
  }
  function capture(root) {
    return [...root.querySelectorAll('.code-block')].map(panel => {
      const state = panel._codeState;
      state.left = panel.querySelector('pre').scrollLeft;
      state.top = panel.querySelector('pre').scrollTop;
      return state;
    });
  }
  function enhance(root, states = []) {
    root.querySelectorAll('pre > code').forEach((code, index) => {
      const pre = code.parentElement;
      if (pre.parentElement.classList.contains('code-block')) return;
      const panel = document.createElement('div'); panel.className = 'code-block';
      const state = states[index] || { left: 0, top: 0 }; state.panel = panel; panel._codeState = state;
      const header = document.createElement('div'); header.className = 'code-header';
      const language = document.createElement('span'); language.className = 'code-language';
      const lang = code.dataset.language || [...code.classList].find(c => c.startsWith('language-'))?.slice(9) || 'text';
      language.textContent = lang;
      if (!code.classList.contains('hljs') && code.textContent.length < 50000 && window.hljs?.getLanguage?.(lang)) {
        code.classList.add('language-' + lang); window.hljs.highlightElement(code);
      }
      const actions = document.createElement('div'); actions.className = 'code-actions';
      for (const name of ['wrap', 'copy']) { const button = document.createElement('button'); button.type = 'button'; button.className = 'code-' + name; actions.append(button); }
      const feedback = document.createElement('span'); feedback.className = 'code-feedback'; feedback.setAttribute('role', 'status');
      header.append(language, actions, feedback); pre.replaceWith(panel); panel.append(header, pre);
      pre.tabIndex = 0; refresh(panel); pre.scrollLeft = state.left; pre.scrollTop = state.top;
    });
  }
  function refreshAll() { document.querySelectorAll('.code-block').forEach(refresh); }
  document.addEventListener('click', async event => {
    const button = event.target.closest('.code-wrap,.code-copy');
    if (!button) return;
    const panel = button.closest('.code-block'); if (!panel) return;
    if (button.classList.contains('code-wrap')) {
      wrapped = !wrapped; try { localStorage.setItem(key, String(wrapped)); } catch (_) {}
      refreshAll(); return;
    }
    const state = panel._codeState; if (state.copying) return;
    state.copying = true;
    try { await navigator.clipboard.writeText(panel.querySelector('pre > code').textContent); state.feedback = 'copied'; }
    catch (_) { state.feedback = 'error'; }
    finally {
      state.copying = false; state.until = Date.now() + 2000;
      refresh(state.panel); clearTimeout(state.timer);
      state.timer = setTimeout(() => { if (state.panel.isConnected) refresh(state.panel); }, 2050);
    }
  });
  window.NeuroCodeBlocks = { capture, enhance, refresh: refreshAll };
})();
