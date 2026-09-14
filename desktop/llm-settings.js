'use strict';

// Pure config translation: no filesystem, process environment, or network access.
const CONTROL_FIELDS = {
  llmApiMode: 'api_mode', llmReasoningEffort: 'reasoning_effort',
  llmThinkingMode: 'thinking_mode', llmMaxOutputTokens: 'max_output_tokens',
  llmTemperature: 'temperature',
};

function applyModelControls(config, target) {
  for (const [ui, wire] of Object.entries(CONTROL_FIELDS)) {
    if (!Object.prototype.hasOwnProperty.call(config, ui)) continue;
    const text = String(config[ui] ?? '').trim();
    if (!text || text === 'default' || (wire === 'api_mode' && text === 'auto')) {
      delete target[wire];
      continue;
    }
    if (wire === 'max_output_tokens') {
      if (!/^[1-9][0-9]*$/.test(text) || !Number.isSafeInteger(Number(text))) throw new Error('Output token limit must be a positive integer.');
      target[wire] = Number(text);
    } else if (wire === 'temperature') {
      if (!Number.isFinite(Number(text)) || Number(text) < 0 || Number(text) > 2) throw new Error('Temperature must be between 0 and 2, or blank for provider default.');
      target[wire] = Number(text);
    } else {
      const valid = { api_mode: ['chat_completions', 'responses', 'anthropic'],
        reasoning_effort: ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'],
        thinking_mode: ['enabled', 'disabled', 'adaptive'] };
      if (!valid[wire].includes(text)) throw new Error(`Invalid ${wire}.`);
      target[wire] = text;
    }
  }
  if (config.llmThinkingMode && config.llmThinkingMode !== 'default') delete target.thinking;
  return target;
}

module.exports = { CONTROL_FIELDS, applyModelControls };
