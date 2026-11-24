'use strict';

const { constants, promises: fs } = require('node:fs');
const path = require('node:path');

class VoiceSettings {
  /** @param {string} state @param {Electron.Dialog} dialog */
  constructor(state, dialog) {
    this.settings = path.join(state, 'voice-input.json');
    this.dialog = dialog;
    this.configuring = false;
  }

  async configure(window) {
    if (this.configuring) return false;
    this.configuring = true;
    try {
      const binary = await this.dialog.showOpenDialog(window, {
        title: 'Choose the whisper-cli executable', buttonLabel: 'Choose executable', properties: ['openFile'],
      });
      if (binary.canceled) return false;
      const [executable] = binary.filePaths;
      if (!(await fs.stat(executable)).isFile()) throw new Error('Choose a whisper-cli executable file.');
      await fs.access(executable, constants.X_OK).catch(() => {
        throw new Error('The selected whisper-cli file needs execute permission.');
      });
      const weights = await this.dialog.showOpenDialog(window, {
        title: 'Choose a Whisper speech model', buttonLabel: 'Choose model', properties: ['openFile'],
        filters: [{ name: 'Whisper models', extensions: ['bin'] }],
      });
      if (weights.canceled) return false;
      const [model] = weights.filePaths;
      if (!(await fs.stat(model)).isFile()) throw new Error('Choose a Whisper model file.');
      await fs.access(model, constants.R_OK).catch(() => {
        throw new Error('The selected Whisper model file needs read permission.');
      });
      const temporary = this.settings + '.tmp';
      try {
        await fs.writeFile(temporary, JSON.stringify({ executable, model }) + '\n', { mode: 0o600 });
        await fs.rename(temporary, this.settings);
      } finally {
        await fs.rm(temporary, { force: true });
      }
      return true;
    } finally {
      this.configuring = false;
    }
  }
}

module.exports = { VoiceSettings };
