const { contextBridge, ipcRenderer, webUtils } = require('electron');

const DESKTOP_VERSION = '0.2.2';

contextBridge.exposeInMainWorld('neuroclawDesktop', {
  version: DESKTOP_VERSION,
  platform: process.platform,
  onMenuAction: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const listener = (_event, action) => callback(action);
    ipcRenderer.on('neuroclaw:menu-action', listener);
    return () => ipcRenderer.removeListener('neuroclaw:menu-action', listener);
  },
  getConfig: () => ipcRenderer.invoke('neuroclaw:get-config'),
  saveConfig: (config) => ipcRenderer.invoke('neuroclaw:save-config', config),
  detectLocalPythons: () => ipcRenderer.invoke('neuroclaw:detect-local-pythons'),
  selectAttachmentFiles: () => ipcRenderer.invoke('neuroclaw:select-attachment-files'),
  selectProjectFolder: () => ipcRenderer.invoke('neuroclaw:select-project-folder'),
  createProjectFolder: (name) => ipcRenderer.invoke('neuroclaw:create-project-folder', name),
  exportUserStudyResults: (request) => ipcRenderer.invoke('neuroclaw:export-user-study-results', request),
  getPathForFile: (file) => {
    if (!file) return '';
    try {
      return webUtils.getPathForFile(file) || '';
    } catch (_error) {
      return '';
    }
  },
  restart: () => ipcRenderer.invoke('neuroclaw:restart'),
});
