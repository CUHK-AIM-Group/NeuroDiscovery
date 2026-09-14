'use strict';

const { CONTROL_FIELDS } = require('./llm-settings');
const { selectedModelIds } = require('./model-library');

// Compare only settings consumed at backend startup. Keep this snapshot in memory;
// never log, serialize to disk, or hash connection credentials for change tracking.
function runtimeSettings(config) {
  const text = key => String(config[key] ?? '').trim();
  const mode = text('runtimeMode');
  const models = selectedModelIds(config);
  const settings = {
    host: text('host'), port: Number(config.port) || 7080,
    runtimeMode: mode, repoRoot: text('repoRoot'),
    environmentFile: text('environmentFile'), fslDir: text('fslDir'),
    proxyUrl: text('proxyUrl'),
    llmProvider: text('llmProvider'), llmBaseUrl: text('llmBaseUrl'),
    llmApiKey: text('llmApiKey'), llmApiKeyEnv: text('llmApiKeyEnv'),
    llmApiKeyFile: text('llmApiKeyFile'),
    llmApiKeySlot: text('llmApiKeyFile') ? (text('llmApiKeySlot') || 'primary') : '',
    llmModel: models.includes(text('llmModel')) ? text('llmModel') : models[0] || '',
    llmAddedModels: models.slice().sort(),
  };
  if (mode === 'conda') {
    settings.condaExe = text('condaExe');
    settings.condaEnv = text('condaEnv');
  } else {
    settings.pythonExe = mode === 'python' && text('localPythonExe')
      ? text('localPythonExe') : text('pythonExe');
  }
  for (const key of Object.keys(CONTROL_FIELDS)) {
    let value = text(key);
    if (value === 'default' || (key === 'llmApiMode' && value === 'auto')) value = '';
    if (value && ['llmMaxOutputTokens', 'llmTemperature'].includes(key) && Number.isFinite(Number(value))) {
      value = String(Number(value));
    }
    settings[key] = value;
  }
  return settings;
}

function createRestartTracker(appliedConfig) {
  const baseline = runtimeSettings(appliedConfig);
  return {
    requiresRestart(config) {
      const current = runtimeSettings(config);
      return Object.keys({...baseline, ...current}).some(key =>
        Array.isArray(baseline[key])
          ? baseline[key].length !== current[key].length || baseline[key].some((value, i) => value !== current[key][i])
          : baseline[key] !== current[key]);
    },
  };
}

module.exports = { createRestartTracker };
