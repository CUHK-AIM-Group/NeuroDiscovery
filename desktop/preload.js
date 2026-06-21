const { contextBridge, ipcRenderer } = require('electron');

const DESKTOP_VERSION = '0.2.0';

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
  restart: () => ipcRenderer.invoke('neuroclaw:restart'),
});
