'use strict';

const { app, BrowserWindow, clipboard, dialog, ipcMain, Menu, net, protocol, session, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const readline = require('node:readline');
const { pathToFileURL } = require('node:url');
const { VoiceSettings } = require('./voice.cjs');

const root = path.resolve(__dirname, '..');
const origin = 'infera-studio://desktop';
const entry = origin + '/index.html';
const assets = new Map([
  ['/index.html', 'electron/index.html'],
  ['/style.css', 'electron/style.css'],
  ['/text-reveal.js', 'electron/text-reveal.js'],
  ['/renderer.js', 'electron/renderer.js'],
  ['/voice.js', 'electron/voice.js'],
  ['/voice-capture.js', 'electron/voice-capture.js'],
  ['/dialogs.js', 'electron/dialogs.js'],
  ['/line-chart.js', 'electron/line-chart.js'],
  ['/analytics.js', 'electron/analytics.js'],
  ['/markdown.js', 'electron/markdown.js'],
  ['/icon.png', 'packaging/infera-studio.png'],
  ['/markdown-it.js', 'node_modules/markdown-it/dist/browser/markdown-it.umd.min.js'],
  ['/highlight.js', 'node_modules/@highlightjs/cdn-assets/highlight.min.js'],
]);
const actions = new Set([
  'initialize', 'search', 'conversation', 'preferences', 'rescan', 'open', 'new', 'configure', 'fork',
  'rename', 'set_pinned', 'delete', 'generate', 'budget', 'context', 'change_context', 'unload',
  'health', 'system_stats', 'diagnostics', 'workspace',
  'hub_search', 'hub_variants', 'hub_download', 'voice_status', 'voice_start',
]);

protocol.registerSchemesAsPrivileged([
  { scheme: 'infera-studio', privileges: { standard: true, secure: true, supportFetchAPI: true } },
]);

const configuredState = process.env.XDG_STATE_HOME || '';
const stateRoot = path.isAbsolute(configuredState) ? configuredState : path.join(os.homedir(), '.local/state');
const currentState = path.join(stateRoot, 'infera-studio');
const legacyState = path.join(stateRoot, 'cortex-local');
const state = path.join(
  !fs.existsSync(currentState) && fs.statSync(legacyState, { throwIfNoEntry: false })?.isDirectory()
    ? legacyState : currentState,
  'electron',
);
fs.mkdirSync(state, { recursive: true, mode: 0o700 });
app.setPath('userData', state);
app.setName('Infera Studio');
app.setDesktopName('infera-studio.desktop');
// The Debian package keeps the stock Electron executable and its resources/app layout.
const voiceBundle = root === path.join(process.resourcesPath, 'app')
  ? path.join(process.resourcesPath, 'whisper') : path.join(root, 'dist', 'whisper');
const voice = new VoiceSettings(state, dialog);

class Backend {
  constructor(onEvent, onExit) {
    this.pending = new Map();
    this.closing = false;
    this.error = '';
    const python = process.env.INFERA_PYTHON || (app.isPackaged ? '/usr/bin/python3' : 'python3');
    const args = ['-B', '-u', '-m', 'infera_studio.desktop.electron.bridge', voice.settings, voiceBundle];
    this.child = spawn(python, args, {
      cwd: root,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.done = new Promise((resolve) => {
      this.child.once('close', () => {
        const error = this.error.trim() || 'The local Infera Studio backend has closed.';
        for (const finish of this.pending.values()) finish({ ok: false, error });
        this.pending.clear();
        if (!this.closing) onExit(error);
        resolve();
      });
    });
    this.child.on('error', (error) => { this.error = error.message; });
    this.child.stdin.on('error', () => {});
    this.child.stderr.setEncoding('utf8');
    this.child.stderr.on('data', (data) => { this.error = (this.error + data).slice(-4096); });
    readline.createInterface({ input: this.child.stdout }).on('line', (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        this.error = 'Infera Studio returned an unreadable backend response.';
        this.child.stdin.end();
        return;
      }
      const finish = this.pending.get(message.id);
      if (!finish) return;
      if (message.event) {
        onEvent(message);
      } else {
        this.pending.delete(message.id);
        finish(message);
      }
    });
  }

  send(id, action, params) {
    if (this.closing || this.child.stdin.destroyed || this.child.stdin.writableEnded) {
      return Promise.resolve({ ok: false, error: this.error.trim() || 'The local Infera Studio backend has closed.' });
    }
    if (this.pending.has(id)) throw new Error('Duplicate Infera Studio request.');
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.child.stdin.write(JSON.stringify({ id, action, params }) + '\n');
    });
  }

  writeVoice(id, samples) {
    if (!this.pending.has(id) || !this.child.stdin.writable || this.child.stdin.writableEnded) return;
    const action = samples === null ? 'voice_finish' : 'voice_audio';
    const audio = samples === null ? undefined : Buffer.from(samples).toString('base64');
    this.child.stdin.write(JSON.stringify({ id, action, audio }) + '\n');
  }

  cancel(id) {
    if (this.pending.has(id) && this.child.stdin.writable && !this.child.stdin.writableEnded) {
      this.child.stdin.write(JSON.stringify({ id, action: 'cancel' }) + '\n');
    }
  }

  async close() {
    this.closing = true;
    this.child.stdin.end();
    const deadline = setTimeout(() => this.child.kill('SIGKILL'), 20_000);
    try {
      await this.done;
    } finally {
      clearTimeout(deadline);
    }
  }
}

let window;
let backend;
let quitting = false;
let disposed = false;

function notify(channel, payload) {
  if (window && !window.isDestroyed() && !window.webContents.isDestroyed()) {
    window.webContents.send(channel, payload);
  }
}

function trusted(event) {
  if (event.sender !== window?.webContents || event.senderFrame !== event.sender.mainFrame
      || event.senderFrame.url !== entry) throw new Error('Untrusted Infera Studio request.');
}

function validId(id) {
  return Number.isSafeInteger(id) && id > 0;
}

function registerHandlers() {
  ipcMain.handle('infera-studio:voice-configure', (event) => {
    trusted(event);
    return voice.configure(window);
  });
  ipcMain.on('infera-studio:voice-write', (event, id, pcm) => {
    if (quitting || event.sender !== window?.webContents || event.senderFrame !== event.sender.mainFrame
        || event.senderFrame.url !== entry || !validId(id)) return;
    if (pcm !== null && (!(pcm instanceof ArrayBuffer) || !pcm.byteLength
        || pcm.byteLength > 4000 * 4 || pcm.byteLength % 4)) return;
    backend.writeVoice(id, pcm);
  });
  ipcMain.handle('infera-studio:request', async (event, id, action, params) => {
    trusted(event);
    if (quitting || !validId(id) || !actions.has(action) || !params || typeof params !== 'object' || Array.isArray(params)) {
      throw new Error('Invalid Infera Studio request.');
    }
    if (action === 'set_pinned' && (!validId(params.chat_id) || typeof params.pinned !== 'boolean')) {
      throw new Error('Invalid conversation pin request.');
    }
    if (action === 'workspace') {
      if (!validId(params.chat_id) || !['ask', 'agent'].includes(params.mode)) {
        throw new Error('Invalid project folder request.');
      }
      let folder = '';
      if (params.mode === 'agent') {
        const result = await dialog.showOpenDialog(window, {
          title: 'Choose a project folder',
          buttonLabel: 'Allow Agent access',
          properties: ['openDirectory', 'createDirectory'],
        });
        if (result.canceled || !result.filePaths.length) return { ok: true, value: null };
        [folder] = result.filePaths;
      }
      return backend.send(id, action, { chat_id: params.chat_id, path: folder });
    }
    return backend.send(id, action, params);
  });
  ipcMain.on('infera-studio:cancel', (event, id) => {
    if (event.sender === window?.webContents && event.senderFrame === event.sender.mainFrame
        && event.senderFrame.url === entry && validId(id)) backend.cancel(id);
  });
  ipcMain.handle('infera-studio:browse', async (event, type) => {
    trusted(event);
    if (type !== 'directory' && type !== 'executable') throw new Error('Unknown file picker.');
    const directory = type === 'directory';
    const result = await dialog.showOpenDialog(window, {
      title: directory ? 'Choose a GGUF model folder' : 'Choose a llama-server executable',
      properties: directory ? ['openDirectory', 'multiSelections'] : ['openFile'],
    });
    return result.canceled ? [] : result.filePaths;
  });
  ipcMain.handle('infera-studio:copy', (event, value) => {
    trusted(event);
    if (typeof value === 'string') clipboard.writeText(value);
  });
  ipcMain.handle('infera-studio:link', (event, value) => {
    trusted(event);
    if (typeof value !== 'string') return;
    const link = new URL(value);
    if (link.protocol === 'https:' || link.protocol === 'http:') return shell.openExternal(link.href);
  });
}

function createWindow() {
  window = new BrowserWindow({
    title: 'Infera Studio',
    width: 1190,
    height: 820,
    minWidth: 760,
    minHeight: 560,
    show: false,
    backgroundColor: '#15171c',
    icon: path.join(root, 'packaging/infera-studio.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', (event) => event.preventDefault());
  window.webContents.on('will-attach-webview', (event) => event.preventDefault());
  window.webContents.on('render-process-gone', () => app.quit());
  window.once('ready-to-show', () => window.show());
  return window.loadURL(entry);
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (!window || window.isDestroyed()) return;
    if (window.isMinimized()) window.restore();
    window.focus();
  });
  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', (event) => {
    if (!backend || disposed) return;
    event.preventDefault();
    if (quitting) return;
    quitting = true;
    backend.close().finally(() => {
      disposed = true;
      app.quit();
    });
  });
  process.on('SIGTERM', () => app.quit());
  process.on('SIGINT', () => app.quit());

  app.whenReady().then(() => {
    Menu.setApplicationMenu(null);
    session.defaultSession.setPermissionRequestHandler((contents, permission, reply, details) => {
      reply(Boolean(contents === window?.webContents && permission === 'media' && details.isMainFrame
        && details.requestingUrl === entry && details.mediaTypes?.length === 1 && details.mediaTypes[0] === 'audio'));
    });
    session.defaultSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) =>
      contents === window?.webContents && permission === 'media' && details.isMainFrame
      && contents?.mainFrame.url === entry && [origin, origin + '/'].includes(requestingOrigin)
      && details.mediaType === 'audio');
    protocol.handle('infera-studio', (request) => {
      const url = new URL(request.url);
      const asset = url.hostname === 'desktop' && assets.get(url.pathname);
      if (request.method !== 'GET' || !asset) return new Response('Not found', { status: 404 });
      return net.fetch(pathToFileURL(path.join(root, asset)).href);
    });
    backend = new Backend((message) => notify('infera-studio:event', message),
      (error) => notify('infera-studio:backend-exit', error));
    registerHandlers();
    return createWindow();
  }).catch((error) => {
    if (quitting) return;
    dialog.showErrorBox('Infera Studio', error.message);
    app.quit();
  });
}
