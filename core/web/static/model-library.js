/* Model discovery is a draft, not the user's saved model library. */
(function(root) {
  'use strict';
  function ids(config) {
    const values = Array.isArray(config.llmAddedModels) ? config.llmAddedModels : (config.llmModel ? [config.llmModel] : []);
    return [...new Set(values.filter(value => typeof value === 'string' && value.trim()).map(value => value.trim()))];
  }
  function create({getConfig, discover, changed}) {
    let revision = 0;
    const state = {status:'idle', models:[], selected:[], error:'', search:'', truncated:false};
    const connection = () => JSON.stringify(['llmProvider','llmBaseUrl','llmApiKey','llmApiKeyEnv','llmApiKeyFile','llmApiKeySlot','llmApiMode'].map(key => getConfig()[key] || ''));
    function invalidate() {
      revision++;
      Object.assign(state, {status:'idle', models:[], selected:[], error:'', search:'', truncated:false});
    }
    function replace(values) {
      const config = getConfig();
      config.llmAddedModels = [...new Set(values)];
      if (!config.llmAddedModels.includes(config.llmModel)) config.llmModel = config.llmAddedModels[0] || '';
    }
    return {
      state, invalidate,
      async fetch() {
        const request = ++revision, identity = connection();
        Object.assign(state, {status:'loading', models:[], selected:[], error:'', truncated:false});
        changed();
        try {
          const result = await discover({...getConfig()});
          if (request !== revision || identity !== connection()) return;
          if (!result.ok) throw new Error(result.message || 'Unable to fetch models.');
          const seen = new Set();
          state.models = (result.models || []).filter(item => typeof item.model === 'string' && item.model.trim() && !seen.has(item.model) && seen.add(item.model));
          state.status = 'ready';
          state.truncated = Boolean(result.truncated);
        } catch (error) {
          if (request !== revision || identity !== connection()) return;
          state.status = 'error'; state.error = error.message;
        }
        if (request === revision && identity === connection()) changed();
      },
      toggle(model, checked) {
        if (!state.models.some(item => item.model === model) || ids(getConfig()).includes(model)) return;
        state.selected = state.selected.filter(value => value !== model);
        if (checked) state.selected.push(model);
        changed();
      },
      addSelected() {
        replace([...ids(getConfig()), ...state.selected.filter(model => state.models.some(item => item.model === model))]);
        state.selected = []; changed();
      },
      addManual(model) {
        model = String(model || '').trim();
        if (!model || model.length > 256 || /[\r\n\0]/.test(model)) return false;
        replace([...ids(getConfig()), model]); changed(); return true;
      },
      remove(model) { replace(ids(getConfig()).filter(value => value !== model)); changed(); },
      makeDefault(model) { if (ids(getConfig()).includes(model)) { getConfig().llmModel = model; changed(); } },
    };
  }
  root.NDModelLibrary = {ids, create};
})(typeof window === 'undefined' ? globalThis : window);
