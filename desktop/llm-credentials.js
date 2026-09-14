'use strict';

const fs = require('node:fs');
const path = require('node:path');

function validateKeyFileConfig(config) {
  if (!String(config.llmApiKeyFile || '').trim()) return;
  let endpoint;
  try { endpoint = new URL(config.llmBaseUrl); } catch (_) { /* rejected below */ }
  if (config.llmProvider !== 'ollama_cloud' || !endpoint
      || endpoint.origin !== 'https://ollama.com' || endpoint.username || endpoint.password
      || !['/v1', '/v1/'].includes(endpoint.pathname) || endpoint.search || endpoint.hash) {
    throw new Error('The Ollama key file can only be used with https://ollama.com/v1.');
  }
  if (!path.isAbsolute(config.llmApiKeyFile) || !['primary', 'reserve'].includes(config.llmApiKeySlot)) {
    throw new Error('Choose an absolute Ollama key file path and a primary or reserve slot.');
  }
  if (String(config.llmApiKey || '').trim()) {
    throw new Error('Use either the Ollama key file or a pasted API key, not both.');
  }
  if (config.llmApiKeyEnv !== 'OLLAMA_API_KEY') {
    throw new Error('The Ollama key file requires OLLAMA_API_KEY as its environment variable.');
  }
}

function resolveDesktopApiKey(config) {
  if (!String(config.llmApiKeyFile || '').trim()) return String(config.llmApiKey || '').trim();
  validateKeyFileConfig(config);
  let raw;
  try { raw = fs.readFileSync(config.llmApiKeyFile, 'utf8'); }
  catch (_) { throw new Error('Cannot read the Ollama key file. Check the path and permissions.'); }
  // Ignore a final line terminator, not blank lines or earlier unrelated credentials.
  const lines = raw.replace(/\r?\n$/, '').split(/\r?\n/).slice(-2).map(line => line.trim());
  if (lines.length !== 2 || lines.some(line => !/^[A-Za-z0-9_.:-]{16,}$/.test(line))) {
    throw new Error('The last two lines of the Ollama key file must each contain one API key.');
  }
  // No automatic retry/rotation: switching accounts is always an explicit choice.
  return lines[config.llmApiKeySlot === 'reserve' ? 1 : 0];
}

function keyFileStatus(config) {
  if (!String(config.llmApiKeyFile || '').trim()) return null;
  try {
    return { apiKeyConfigured: Boolean(resolveDesktopApiKey(config)), apiKeySource: 'key-file',
      apiKeySlot: config.llmApiKeySlot };
  } catch (error) {
    return { apiKeyConfigured: false, apiKeySource: 'key-file-error', message: error.message };
  }
}

module.exports = { validateKeyFileConfig, resolveDesktopApiKey, keyFileStatus };
