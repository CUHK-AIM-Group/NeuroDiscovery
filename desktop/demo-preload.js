const {contextBridge,ipcRenderer}=require('electron');
contextBridge.exposeInMainWorld('demoDesktop',Object.freeze({openData:()=>ipcRenderer.invoke('demo:open-data')}));
