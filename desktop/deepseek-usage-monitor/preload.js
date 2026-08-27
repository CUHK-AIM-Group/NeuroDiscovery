// 预加载脚本：通过 contextBridge 暴露最小化 API
// 渲染进程无法访问 Node / Electron 内部，只能调用以下方法。
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('dsApi', {
  getSettings: () => ipcRenderer.invoke('settings:get'),
  saveSettings: (patch) => ipcRenderer.invoke('settings:save', patch),
  clearSettings: () => ipcRenderer.invoke('settings:clear'),
  addAccount: (name, apiKey, sessionToken) => ipcRenderer.invoke('account:add', name, apiKey, sessionToken),
  updateAccount: (id, patch) => ipcRenderer.invoke('account:update', id, patch),
  deleteAccount: (id) => ipcRenderer.invoke('account:delete', id),
  activateAccount: (id) => ipcRenderer.invoke('account:activate', id),
  getBalance: () => ipcRenderer.invoke('balance:get'),
  getUsage: (month, year) => ipcRenderer.invoke('usage:get', month, year),
  testConnection: (apiKey, sessionToken) => ipcRenderer.invoke('connection:test', apiKey, sessionToken),
  getHistory: (opts) => ipcRenderer.invoke('history:get', opts),
  clearHistory: () => ipcRenderer.invoke('history:clear'),
  openExternal: (url) => ipcRenderer.invoke('open-external', url),
});
