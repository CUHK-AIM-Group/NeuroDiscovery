(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NeuroDiscoveryHarness = factory();
})(typeof window === 'object' ? window : globalThis, function () {
  'use strict';

  const labels = {
    chat: ['Research chat', '研究对话'], graph: ['Knowledge graph', '知识图谱'],
    autoresearch: ['AutoResearch', 'AutoResearch'], evaluation: ['Human Evaluation', '人工评估'],
    ranking: ['Hypothesis ranking', '假设排序'], results: ['Evaluation results', '评估结果'],
  };
  const symbols = { chat: '◌', graph: '◇', autoresearch: '↻', evaluation: '◎', ranking: '≡', results: '▤' };
  const modes = { off: ['Off', '关闭'], data: ['Data', '数据'], model: ['Model', '模型'], idea: ['Idea', '假设'], 'end-to-end': ['End to end', '全流程'] };

  function sync(document, { language = 'en', view = 'chat', mode = 'off' } = {}) {
    const locale = language.startsWith('zh') ? 1 : 0;
    for (const button of document.querySelectorAll('[data-harness-capability]')) {
      const id = button.dataset.harnessCapability;
      const label = labels[id]?.[locale] || id;
      button.querySelector('.harness-label').textContent = label;
      if (button.dataset.harnessView === view) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
      if (id === 'autoresearch') {
        const scope = (modes[mode] || modes.off)[locale];
        button.querySelector('.harness-mode').textContent = scope;
        button.setAttribute('aria-label', `${label}: ${scope}`);
      }
    }
    const heading = document.querySelector('.harness-section-title');
    if (heading) heading.textContent = locale ? '研究工具' : 'Research tools';
    const description = document.querySelector('.harness-welcome-copy');
    if (description) description.textContent = locale ? '从一个研究想法开始，或添加数据与文献。' : 'Start with a research idea, or add data and papers.';
  }

  function activate(capability, adapter) {
    if (capability.kind === 'configure' && capability.id === 'autoresearch') {
      adapter.configureResearch();
    } else if (capability.kind === 'view') {
      adapter.openView(capability.view);
    }
  }

  async function mount({ document, location, fetch, adapter, evaluationEnabled, getState = () => ({}) }) {
    if (location.pathname !== '/harness') return;
    const response = await fetch('/api/harness');
    if (!response.ok) throw new Error('Workspace capabilities unavailable');
    const manifest = await response.json();
    if (manifest.schema_version !== 1) throw new Error('Unsupported workspace schema');
    const capabilities = manifest.capabilities.filter(capability => evaluationEnabled || !['evaluation', 'ranking', 'results'].includes(capability.id));
    const navigation = document.createElement('nav');
    navigation.className = 'harness-capabilities';
    navigation.setAttribute('aria-label', 'NeuroDiscovery research capabilities');
    const heading = document.createElement('div');
    heading.className = 'harness-section-title';
    navigation.append(heading);
    for (const capability of capabilities.filter(item => ['chat', 'graph'].includes(item.id))) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.harnessCapability = capability.id;
      if (capability.view) button.dataset.harnessView = capability.view;
      const icon = document.createElement('span');
      icon.className = 'harness-icon';
      icon.setAttribute('aria-hidden', 'true');
      icon.textContent = symbols[capability.id] || '○';
      const label = document.createElement('span');
      label.className = 'harness-label';
      button.append(icon, label);
      if (capability.kind === 'configure') {
        button.setAttribute('aria-haspopup', 'listbox');
        const scope = document.createElement('span');
        scope.className = 'harness-mode';
        button.append(scope);
      }
      button.addEventListener('click', () => activate(capability, adapter));
      navigation.append(button);
    }
    const sidebar = document.querySelector('#app-sidebar');
    sidebar.querySelector('.primary-nav').after(navigation);
    const welcome = document.querySelector('#welcome');
    if (welcome) {
      const hero = document.createElement('div');
      hero.className = 'harness-welcome';
      const title = document.createElement('h1');
      title.textContent = 'NeuroDiscovery';
      const logo = welcome.querySelector('.welcome-logo');
      if (logo) {
        logo.alt = '';
        title.prepend(logo);
      }
      const description = document.createElement('p');
      description.className = 'harness-welcome-copy';
      hero.append(title, description);
      welcome.append(hero);
    }
    document.documentElement.dataset.harness = 'true';
    sync(document, getState());
    const requested = new URLSearchParams(location.search).get('capability');
    const initial = capabilities.find(capability => capability.id === requested);
    if (initial) activate(initial, adapter);
  }

  return { activate, mount, sync };
});
