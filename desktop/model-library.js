'use strict';

const { resolveDesktopApiKey } = require('./llm-credentials');

function selectedModelIds(config) {
  // Legacy desktop configs migrate only their active model, never a provider catalog.
  const values = config.llmAddedModels == null
    ? (config.llmModel ? [config.llmModel] : []) : config.llmAddedModels;
  if (!Array.isArray(values) || values.length > 500) throw new Error('Choose at most 500 models.');
  if (values.some(value => typeof value !== 'string' || !value.trim() || value.length > 256 || /[\r\n\0]/.test(value))) {
    throw new Error('Model IDs must be non-empty single-line text.');
  }
  return [...new Set(values.map(value => value.trim()))];
}

async function discoverModels(config, fetcher = fetch, environment = process.env) {
  let base;
  try { base = new URL(String(config.llmBaseUrl || '').trim()); } catch (_) { /* rejected below */ }
  if (!base || !['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
    throw new Error('Enter an HTTP(S) API base URL without credentials, query parameters or fragments.');
  }
  if (base.protocol !== 'https:' && !['localhost', '127.0.0.1', '[::1]'].includes(base.hostname)) {
    throw new Error('Use HTTPS for remote model endpoints. HTTP is supported for local servers only.');
  }
  if (config.llmProvider === 'ollama_cloud' && (base.origin !== 'https://ollama.com' || !['/v1', '/v1/'].includes(base.pathname))) {
    throw new Error('Ollama Cloud requires https://ollama.com/v1. For a custom proxy, choose an OpenAI-compatible provider.');
  }
  const key = resolveDesktopApiKey(config) || String(environment[config.llmApiKeyEnv] || '').trim();
  const nativeAnthropic = config.llmProvider === 'anthropic' || config.llmApiMode === 'anthropic';
  const headers = {Accept: 'application/json'};
  if (nativeAnthropic) {
    headers['anthropic-version'] = '2023-06-01';
    if (key) headers['x-api-key'] = key;
  } else if (key) headers.Authorization = `Bearer ${key}`;
  const root = base.href.replace(/\/$/, '');
  const endpoint = root + (nativeAnthropic && !root.endsWith('/v1') ? '/v1/models' : '/models');
  const models = new Map();
  let after = '', truncated = false;
  for (let page = 0; page < 10; page++) {
    let response;
    try {
      response = await fetcher(endpoint + (after ? `?after_id=${encodeURIComponent(after)}` : ''), {
        method: 'GET', headers, redirect: 'error', signal: AbortSignal.timeout(20000),
      });
    } catch (_) {
      throw new Error('Could not reach the model endpoint. Check the address, proxy and connection; redirects are not followed.');
    }
    if (!response.ok) {
      const reasons = {401:'API key was rejected',403:'This key does not have access',404:'Model-list endpoint not found',429:'Provider rate limit reached; no key rotation was attempted'};
      throw new Error(`${reasons[response.status] || 'Model-list request failed'} (HTTP ${response.status}).`);
    }
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error('The model endpoint did not return valid JSON.'); }
    if (!payload || !Array.isArray(payload.data)) throw new Error('The endpoint returned no compatible model list. You can add an exact model ID manually.');
    for (const item of payload.data) {
      const id = typeof item?.id === 'string' ? item.id.trim() : '';
      if (id && id.length <= 256 && !/[\r\n\0]/.test(id)) models.set(id, {model:id, label:id});
    }
    if (!nativeAnthropic || !payload.has_more) break;
    if (!payload.last_id || after === payload.last_id) throw new Error('The model endpoint returned an invalid pagination cursor.');
    after = payload.last_id;
    truncated = page === 9;
  }
  return {models: [...models.values()].sort((a,b) => a.model.localeCompare(b.model, undefined, {numeric:true})), truncated};
}

module.exports = {selectedModelIds, discoverModels};
