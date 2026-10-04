(function (root) {
  'use strict';
  const commands = [
    {command:'/autoresearch',icon:'◎',mode:'end-to-end',en:'Full workflow · continue within scope and budget',zh:'Full 全流程 · 在范围和预算内持续完成任务'},
    {command:'/data',icon:'D',mode:'data',en:'Data · processing and quality control',zh:'Data 子模式 · 数据处理与质量控制'},
    {command:'/model',icon:'M',mode:'model',en:'Model · implementation, training and evaluation',zh:'Model 子模式 · 模型实现、训练与评估'},
    {command:'/idea',icon:'I',mode:'idea',en:'Idea · literature and research hypotheses',zh:'Idea 子模式 · 文献检索与研究假设'},
    {command:'/goal',icon:'G',action:'goal',en:'Goal · bounded objective with explicit acceptance review',zh:'Goal · 有界目标与独立验收'},
    {command:'/tasks',icon:'◷',action:'tasks',en:'Manage periodic experiment status checks',zh:'管理实验状态定时检查'},
    {command:'/usage',icon:'▥',action:'usage',en:'View requests and token usage',zh:'查看请求次数与 token 使用量'},
    {command:'/compact',icon:'⛶',action:'compact',en:'Compact context; preserve the full local archive',zh:'压缩上下文，保留完整本地归档'},
  ];
  function parse(value) {
    const match = String(value || '').trim().match(/^(\/\S+)(?:\s+([\s\S]*))?$/);
    if (!match) return null;
    const item = commands.find(entry => entry.command === match[1].toLowerCase());
    if (!item) return null;
    let mode = item.mode, text = (match[2] || '').trim();
    if (item.command === '/autoresearch') {
      const scope = text.match(/^(full|data|model|idea)(?:\s+|$)([\s\S]*)/i);
      if (scope) { mode = scope[1].toLowerCase() === 'full' ? 'end-to-end' : scope[1].toLowerCase(); text = scope[2].trim(); }
    }
    return {...item, mode, text};
  }
  const api = Object.freeze({commands:Object.freeze(commands),parse});
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.ComposerCommands = api;
})(typeof window === 'undefined' ? globalThis : window);
