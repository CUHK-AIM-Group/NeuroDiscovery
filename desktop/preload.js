const { contextBridge, ipcRenderer, webUtils } = require('electron');

const DESKTOP_VERSION = '1.0.0';
let latestSettingsSection = null;
const settingsNavigateListeners = new Set();
ipcRenderer.on('neuroclaw:settings-navigate', (_event, section) => {
  latestSettingsSection = section;
  for (const callback of settingsNavigateListeners) callback(section);
});

contextBridge.exposeInMainWorld('neuroclawDesktop', {
  version: DESKTOP_VERSION,
  platform: process.platform,
  titleBarOverlay: false,
  showApplicationMenu: () => ipcRenderer.invoke('neuroclaw:show-application-menu'),
  openSettingsWindow: (target) => ipcRenderer.invoke('neuroclaw:open-settings-window', target),
  onSettingsNavigate: (callback) => {
    if (typeof callback !== 'function') return () => {};
    settingsNavigateListeners.add(callback);
    if (latestSettingsSection !== null) callback(latestSettingsSection);
    return () => settingsNavigateListeners.delete(callback);
  },
  onConfigChanged: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const listener = (_event, change) => callback(change);
    ipcRenderer.on('neuroclaw:config-changed', listener);
    return () => ipcRenderer.removeListener('neuroclaw:config-changed', listener);
  },
  onMenuAction: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const listener = (_event, action) => callback(action);
    ipcRenderer.on('neuroclaw:menu-action', listener);
    return () => ipcRenderer.removeListener('neuroclaw:menu-action', listener);
  },
  getConfig: () => ipcRenderer.invoke('neuroclaw:get-config'),
  subscriptionStatus: (engine) => ipcRenderer.invoke('neuroclaw:subscription-status', engine),
  subscriptionLogin: (engine) => ipcRenderer.invoke('neuroclaw:subscription-login', engine),
  saveConfig: (config) => ipcRenderer.invoke('neuroclaw:save-config', config),
  discoverModels: (config) => ipcRenderer.invoke('neuroclaw:discover-models', config),
  setLanguage: (language) => ipcRenderer.invoke('neuroclaw:set-language', language),
  setTheme: (theme) => ipcRenderer.invoke('neuroclaw:set-theme', theme),
  resetApplication: () => ipcRenderer.invoke('neuroclaw:reset-application'),
  detectLocalPythons: () => ipcRenderer.invoke('neuroclaw:detect-local-pythons'),
  selectAttachmentFiles: () => ipcRenderer.invoke('neuroclaw:select-attachment-files'),
  selectProjectFolder: () => ipcRenderer.invoke('neuroclaw:select-project-folder'),
  createProjectFolder: (name) => ipcRenderer.invoke('neuroclaw:create-project-folder', name),
  exportChatSession: (request) => ipcRenderer.invoke('neuroclaw:export-chat-session', request),
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
