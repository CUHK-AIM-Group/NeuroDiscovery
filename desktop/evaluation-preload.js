const {contextBridge, ipcRenderer} = require('electron');
contextBridge.exposeInMainWorld('neuroclawDesktop', {
  exportUserStudyResults: request => ipcRenderer.invoke('evaluation:export', request),
});
