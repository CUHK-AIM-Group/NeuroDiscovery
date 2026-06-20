const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('neuroclawDesktop', {
  version: '0.2.0',
  platform: process.platform,
  onMenuAction: (callback) => {
    if (typeof callback !== 'function') return () => {};
    const listener = (_event, action) => callback(action);
    ipcRenderer.on('neuroclaw:menu-action', listener);
    return () => ipcRenderer.removeListener('neuroclaw:menu-action', listener);
  },
  getConfig: () => ipcRenderer.invoke('neuroclaw:get-config'),
  saveConfig: (config) => ipcRenderer.invoke('neuroclaw:save-config', config),
  restart: () => ipcRenderer.invoke('neuroclaw:restart'),
});
