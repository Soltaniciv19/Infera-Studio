'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('inferaStudio', {
  request: (id, action, params) => ipcRenderer.invoke('infera-studio:request', id, action, params),
  cancel: (id) => ipcRenderer.send('infera-studio:cancel', id),
  browse: (type) => ipcRenderer.invoke('infera-studio:browse', type),
  copy: (text) => ipcRenderer.invoke('infera-studio:copy', text),
  openLink: (url) => ipcRenderer.invoke('infera-studio:link', url),
  configureVoice: () => ipcRenderer.invoke('infera-studio:voice-configure'),
  writeVoice: (id, pcm) => ipcRenderer.send('infera-studio:voice-write', id, pcm),
  onEvent: (callback) => ipcRenderer.on('infera-studio:event', (_event, message) => callback(message)),
  onExit: (callback) => ipcRenderer.on('infera-studio:backend-exit', (_event, error) => callback(error)),
});
