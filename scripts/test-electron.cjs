'use strict';

const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const { once } = require('node:events');
const fs = require('node:fs');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');

async function main() {
  const { INFERA_TEST_LAUNCHER, INFERA_TEST_SCREENSHOT, INFERA_TEST_ATTACHMENT,
    INFERA_TEST_IMAGE, INFERA_TEST_PROJECTOR, INFERA_TEST_SAVED_MESSAGE, INFERA_TEST_VOICE_AUDIO } = process.env;
  const root = path.resolve(__dirname, '..');
  const binary = INFERA_TEST_LAUNCHER || require('electron');
  const flags = ['--headless', '--ozone-platform=headless', '--disable-gpu', '--remote-debugging-pipe'];
  if (INFERA_TEST_VOICE_AUDIO) {
    flags.push('--use-fake-device-for-media-stream', `--use-file-for-fake-audio-capture=${INFERA_TEST_VOICE_AUDIO}`);
  }
  const app = spawn(binary, INFERA_TEST_LAUNCHER ? flags : [...flags, root], {
    cwd: root,
    stdio: ['ignore', 'ignore', 'inherit', 'pipe', 'pipe'],
  });
  const exited = once(app, 'exit');
  let active = true;
  let diagnostics = [];
  let session;
  let nextId = 0;
  let buffer = '';
  const pending = new Map();

  app.stdio[3].on('error', () => {});
  app.stdio[4].setEncoding('utf8');
  app.stdio[4].on('data', (chunk) => {
    buffer += chunk;
    let end;
    while ((end = buffer.indexOf('\0')) >= 0) {
      const message = JSON.parse(buffer.slice(0, end));
      buffer = buffer.slice(end + 1);
      if (message.method === 'Runtime.exceptionThrown' || message.method === 'Log.entryAdded') {
        diagnostics.push(message.params);
        diagnostics = diagnostics.slice(-12);
      }
      pending.get(message.id)?.(message);
    }
  });
  app.on('exit', () => {
    active = false;
    for (const handler of pending.values()) handler({ error: { message: 'Electron exited early.' } });
  });

  async function send(method, params = {}, attached = session) {
    if (!active) throw new Error('Electron exited early.');
    const id = ++nextId;
    const response = await new Promise((resolve) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        resolve({ error: { message: method + ' timed out.' } });
      }, 8000);
      pending.set(id, (message) => {
        clearTimeout(timer);
        pending.delete(id);
        resolve(message);
      });
      app.stdio[3].write(JSON.stringify({ id, method, params, sessionId: attached }) + '\0');
    });
    if (response.error) throw new Error(response.error.message);
    return response.result;
  }

  async function evaluate(expression) {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }

  async function wait(fn, description, timeout = 12_000) {
    const limit = Date.now() + timeout;
    while (Date.now() < limit) {
      const answer = await fn();
      if (answer) return answer;
      await delay(70);
    }
    throw new Error('Timed out waiting for ' + description);
  }

  async function click(selector) {
    await evaluate('document.querySelector(' + JSON.stringify(selector) + ').click()');
  }

  async function shortcut(key) {
    const options = { modifiers: 2, key, code: 'Key' + key.toUpperCase() };
    await send('Input.dispatchKeyEvent', { type: 'keyDown', ...options });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', ...options });
  }

  async function type(selector, text) {
    await evaluate(`(() => {
      const input = document.querySelector(${JSON.stringify(selector)});
      input.value = ${JSON.stringify(text)};
      input.dispatchEvent(new Event('input', { bubbles: true }));
    })()`);
  }

  async function inputWithLabel(label, value) {
    await evaluate(`(() => {
      const label = [...document.querySelectorAll('#modal-content label')].find((item) =>
        item.firstElementChild?.textContent === ${JSON.stringify(label)});
      label.querySelector('input, select').value = ${JSON.stringify(value)};
    })()`);
  }

  const status = (value) => evaluate(`document.querySelector('#status-dot').dataset.state === ${JSON.stringify(value)}`);
  const contains = (selector, text) => evaluate(`document.querySelector(${JSON.stringify(selector)})
    ?.textContent.includes(${JSON.stringify(text)}) ?? false`);

  try {
    const target = await wait(async () => (await send('Target.getTargets')).targetInfos
      .find((item) => item.url === 'infera-studio://desktop/index.html'), 'the Infera Studio desktop');
    ({ sessionId: session } = await send('Target.attachToTarget', { targetId: target.targetId, flatten: true }));
    await send('Runtime.enable');
    await send('Log.enable');
    if (INFERA_TEST_SAVED_MESSAGE) {
      const loading = await wait(() => evaluate(`(() => {
        if (document.querySelector('#status-dot')?.dataset.state !== 'Loading') return null;
        return {
          splash: !document.querySelector('#startup').hidden,
          conversation: document.querySelector('#messages').textContent.includes(${JSON.stringify(INFERA_TEST_SAVED_MESSAGE)}),
          sidebar: !document.querySelector('#sidebar').inert,
          workspace: !document.querySelector('main').inert,
          settings: !document.querySelector('#settings').disabled,
          stop: !document.querySelector('#stop').hidden,
          composer: !document.querySelector('#draft').disabled,
          loadingIndicator: getComputedStyle(document.querySelector('#loading')).display !== 'none',
        };
      })()`), 'startup while the saved model is loading');
      assert.deepEqual(loading, {
        splash: false, conversation: true, sidebar: true, workspace: true, settings: true, stop: true, composer: false,
        loadingIndicator: true,
      });
      await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'no-preference' }] });
      const [start, early, extending, stretched, retracting, nextLap] = await evaluate(`(() => {
        const strokes = [...document.querySelectorAll('#loading rect')];
        const animations = strokes.flatMap((stroke) => stroke.getAnimations());
        const initialTimes = animations.map((animation) => animation.currentTime);
        const frames = [0, .03, .25, .5, .75, 1].map((progress) => {
          animations.forEach((animation) => {
            animation.currentTime = animation.effect.getTiming().duration * progress;
          });
          return strokes.map((stroke) => {
            const { strokeDashoffset, strokeDasharray } = getComputedStyle(stroke);
            return { offset: Number.parseFloat(strokeDashoffset), pattern: strokeDasharray.split(',').map(Number.parseFloat) };
          });
        });
        animations.forEach((animation, index) => { animation.currentTime = initialTimes[index]; });
        return frames;
      })()`);
      for (let index = 0; index < start.length; index++) {
        const offset = start[index].offset;
        assert.equal(early[index].offset, offset - 30, 'every stroke moves immediately');
        assert.equal(stretched[index].offset, offset - 500, 'every stroke completes half a lap at the same speed');
      }
      const gapToLead = ([lead, dashes]) => dashes.offset - lead.offset
        - dashes.pattern.slice(0, 5).reduce((total, segment) => total + segment, 0);
      assert.ok(gapToLead(start) > gapToLead(extending) && gapToLead(extending) > 0);
      assert.equal(gapToLead(stretched), 0, 'the stretched tail reaches the lead');
      assert.equal(gapToLead(retracting), gapToLead(extending));
      assert.ok(stretched[2].pattern[4] > start[2].pattern[4], 'the dots lengthen with the dashes');
      assert.deepEqual(nextLap, start, 'the loop resets to the same stroke positions and lengths');
      await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
      assert.equal(await evaluate("document.querySelector('#loading').getAnimations({ subtree: true }).length"), 0);
      await send('Emulation.setEmulatedMedia', { features: [] });
      await evaluate("document.querySelector('#search').focus()");
    }
    await wait(() => evaluate("document.querySelector('#startup')?.hidden"), 'the startup splash to finish');
    assert.equal(await evaluate("document.querySelector('#models').disabled"), false);
    if (INFERA_TEST_SAVED_MESSAGE) {
      await wait(() => status('Ready'), 'the restored model to be ready');
      assert.equal(await evaluate("getComputedStyle(document.querySelector('#loading')).display"), 'none');
      assert.equal(await evaluate("document.activeElement.id"), 'search');
    }
    if (process.env.INFERA_TEST_SAVED_CHAT) {
      await wait(() => contains('#history', process.env.INFERA_TEST_SAVED_CHAT), 'the saved conversation');
    }
    assert.equal(await evaluate("typeof require"), 'undefined');
    assert.deepEqual(await evaluate("Array.from(document.querySelectorAll('.composer-tools button'), (button) => button.id)"),
      ['attach', 'vision', 'web-search', 'mtp', 'thinking']);
    assert.equal(await contains('#attach', 'Add files'), true);
    assert.equal(await contains('#mode-ask', 'Ask'), true);
    assert.equal(await contains('#mode-agent', 'Agent'), true);
    assert.equal(await evaluate("document.querySelector('#web-search').textContent.trim()"), '');
    assert.match(await evaluate("document.querySelector('#web-search').getAttribute('aria-label')"), /web search/i);
    assert.equal(await evaluate("document.querySelector('#web-search use').getAttribute('href')"), '#icon-globe');
    if (process.env.INFERA_TEST_AGENT) {
      if (INFERA_TEST_SCREENSHOT) {
        await send('Emulation.setDeviceMetricsOverride', { width: 1120, height: 740, deviceScaleFactor: 1, mobile: false });
      }
      await wait(() => evaluate("!document.querySelector('#draft').disabled"), 'the Agent conversation');
      assert.equal(await evaluate("document.querySelector('#mode-agent').getAttribute('aria-pressed')"), 'true');
      await type('#draft', 'Create the fixture project');
      await click('#send');
      const assistant = '#messages .assistant:last-child';
      const progress = assistant + ' .message-progress';
      await wait(() => contains(progress, 'Generating file content…'), 'the live file creation status');
      assert.equal(await contains(assistant, 'I’ll write the result file.'), true);
      assert.equal(await contains(assistant, 'src/result.txt'), false);
      if (INFERA_TEST_SCREENSHOT) {
        const screenshot = await send('Page.captureScreenshot', { format: 'png' });
        fs.writeFileSync(INFERA_TEST_SCREENSHOT, Buffer.from(screenshot.data, 'base64'));
      }
      const firstReasoning = assistant + ' .response-step:first-child details';
      await click(firstReasoning + ' summary');
      await wait(() => status('Ready'), 'the completed Agent response');
      assert.equal(await evaluate(`document.querySelector('${firstReasoning}').open`), true);
      const verifyTimeline = async () => {
        assert.deepEqual(await evaluate(`[...document.querySelectorAll('${assistant} .response-step')]
          .map((step) => [step.querySelector(':scope > .message-body').textContent.trim(),
            step.querySelector('details .message-body').textContent.trim(),
            step.querySelector('li > span:last-child')?.textContent || ''])`), [
          ['I’ll locate the seed file 🔎.', 'Project step 1 →', 'Searched project'],
          ['I’ll read the seed.', 'Project step 2 →', 'Read seed.txt'],
          ['I’ll write the result file.', 'Project step 3 →', 'Created src/result.txt'],
          ['Project created.', '', ''],
        ]);
        assert.equal(await evaluate(`document.querySelector('${progress}') === null`), true);
      };
      await verifyTimeline();
      await click('#new-chat');
      await wait(() => contains('#chat-title', 'New chat'), 'another conversation');
      await click('.history-name[title="Agent fixture"]');
      await wait(() => contains(assistant, 'Project created.'), 'the saved Agent response');
      await verifyTimeline();
      return;
    }
    if (process.env.INFERA_TEST_SAVED_WORKSPACE) {
      await wait(() => contains('#workspace-name', process.env.INFERA_TEST_SAVED_WORKSPACE), 'the project folder');
      assert.equal(await evaluate("document.querySelector('#mode-agent').getAttribute('aria-pressed')"), 'true');
      await wait(() => evaluate("!document.querySelector('#mode-ask').disabled"), 'the enabled mode switch');
      await click('#mode-ask');
      await wait(() => evaluate("document.querySelector('#workspace').hidden"), 'Ask mode');
    }
    assert.equal(await evaluate("document.querySelector('#mode-ask').getAttribute('aria-pressed')"), 'true');
    assert.equal(await evaluate(`(() => {
      const model = document.querySelector('#models').getBoundingClientRect();
      const modes = document.querySelector('.composer-mode').getBoundingClientRect();
      const draft = document.querySelector('#draft').getBoundingClientRect();
      return model.left >= modes.right && model.bottom <= draft.top && model.right === draft.right;
    })()`), true);
    assert.deepEqual(await evaluate(`[...document.querySelectorAll('.toolbar-actions > button')]
      .map((button) => [button.id, button.textContent.trim(), Boolean(button.getAttribute('aria-label'))])`), [
      ['server', '', true], ['context', '', true], ['model-settings', '', true],
      ['advanced-stats', '', true], ['system-stats-toggle', '', true],
    ]);
    if (INFERA_TEST_ATTACHMENT) {
      assert.equal(await evaluate("document.querySelector('#system-stats').matches(':popover-open')"), false);
      await wait(() => contains('[data-stat="cpu_load"] strong', '%'), 'live CPU load');
      assert.equal(await contains('[data-stat="ram"] strong', 'GiB'), true);
      if (process.env.INFERA_EXPECT_CPU_TEMP === '1') {
        assert.equal(await contains('[data-stat="cpu_temp"] strong', '°C'), true);
      }
      if (process.env.INFERA_EXPECT_GPU_STATS === '1') {
        assert.equal(await evaluate(`[...document.querySelectorAll('[data-stat^="gpu_"] strong')]
          .some(({ textContent }) => textContent !== '—')`), true);
      }
      await send('Emulation.setDeviceMetricsOverride', { width: 760, height: 600, deviceScaleFactor: 1, mobile: false });
      assert.equal(await evaluate(`(() => {
        const sidebar = document.querySelector('#sidebar').getBoundingClientRect();
        const search = document.querySelector('.search').getBoundingClientRect();
        const create = document.querySelector('#new-chat').getBoundingClientRect();
        return search.right < create.left && create.right < sidebar.right && search.top === create.top;
      })()`), true);
      await evaluate("document.querySelector('#system-stats-toggle').focus()");
      await click('#system-stats-toggle');
      const layout = await evaluate(`(() => {
        const stats = document.querySelector('#system-stats').getBoundingClientRect();
        const toolbar = document.querySelector('.toolbar').getBoundingClientRect();
        const toggle = document.querySelector('#system-stats-toggle').getBoundingClientRect();
        return { stats: stats.toJSON(), toolbar: toolbar.toJSON(), toggle: toggle.toJSON() };
      })()`);
      assert.ok(layout.stats.width > 0 && layout.stats.left >= layout.toolbar.left &&
        layout.stats.right === layout.toggle.right && layout.stats.top >= layout.toolbar.bottom, JSON.stringify(layout));
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
      await wait(() => evaluate("!document.querySelector('#system-stats').matches(':popover-open')"), 'statistics to close on Escape');
      assert.equal(await evaluate('document.activeElement.id'), 'system-stats-toggle');
      await send('Emulation.clearDeviceMetricsOverride');
    }

    if (INFERA_TEST_SCREENSHOT) {
      const screenshot = await send('Page.captureScreenshot', { format: 'png' });
      fs.writeFileSync(INFERA_TEST_SCREENSHOT, Buffer.from(screenshot.data, 'base64'));
    }
    if (process.env.INFERA_TEST_BUNDLED_VOICE === '1') {
      await wait(() => evaluate("document.querySelector('#dictate').ariaLabel === 'Dictate message'"), 'bundled dictation');
      await click('#settings');
      await click('[aria-controls="settings-voice"]');
      assert.equal(await contains('#settings-voice', 'Ready with the bundled multilingual Whisper model.'), true);
      await click('#modal-close');
    }
    if (process.env.INFERA_TEST_STARTUP_ONLY === '1') return;

    await click('#models');
    assert.equal(await evaluate(`(() => {
      const picker = document.querySelector('#models').getBoundingClientRect();
      const menu = document.querySelector('#model-menu').getBoundingClientRect();
      return menu.bottom < picker.top && menu.right === picker.right && document.activeElement.id === 'model-search';
    })()`), true);
    await click('.model-option');
    await wait(() => status('Ready'), 'the model to load');
    if (INFERA_TEST_VOICE_AUDIO) {
      assert.equal(await evaluate(`navigator.mediaDevices.getUserMedia({ video: true }).then(
        (stream) => { stream.getTracks().forEach((track) => track.stop()); return 'allowed'; },
        (error) => error.name)`), 'NotAllowedError');
      await wait(() => evaluate("!document.querySelector('#dictate').disabled"), 'voice input');
      assert.equal(await evaluate("document.querySelector('#dictate').ariaLabel"), 'Dictate message');
      const realWhisper = process.env.INFERA_TEST_VOICE_MATCH;
      const draftText = () => evaluate("document.querySelector('#draft').value");
      await type('#draft', 'Existing draft.');
      const composerHeight = () => evaluate("document.querySelector('#composer').getBoundingClientRect().height");
      const idleHeight = await composerHeight();
      await click('#dictate');
      await wait(() => contains('#voice-status', 'Recording'), 'microphone recording');
      assert.equal(await evaluate("document.querySelector('#send').disabled"), true);
      assert.equal(await composerHeight(), idleHeight);
      assert.equal(await evaluate(`(() => {
        const stop = document.querySelector('#dictate');
        const button = stop.getBoundingClientRect();
        const signal = document.querySelector('#voice-signal').getBoundingClientRect();
        const tools = document.querySelector('.composer-tools').getBoundingClientRect();
        return stop.ariaLabel === 'Stop recording and transcribe' && !stop.textContent.trim()
          && stop.querySelector('use').getAttribute('href') === '#icon-record-stop'
          && getComputedStyle(stop).borderRadius === '50%' && button.width === button.height
          && signal.left >= tools.right && signal.right < button.left && signal.width > 64;
      })()`), true);
      const waveform = await wait(() => evaluate(`(() => {
        const bars = document.querySelector('#voice-waveform-bars');
        return bars.getBBox().height > 5 ? bars.getAttribute('d') : null;
      })()`), 'live microphone waveform');
      await wait(() => evaluate(`document.querySelector('#voice-waveform-bars').getAttribute('d') !==
        ${JSON.stringify(waveform)}`), 'new microphone samples');
      await wait(() => contains('#voice-elapsed', '0:01'), 'elapsed recording time');
      await click('#sidebar-close');
      await send('Emulation.setDeviceMetricsOverride', { width: 410, height: 560, deviceScaleFactor: 1, mobile: false });
      assert.equal(await evaluate(`(() => {
        const signal = document.querySelector('#voice-signal').getBoundingClientRect();
        const controls = document.querySelector('.composer-controls').getBoundingClientRect();
        const tools = document.querySelector('.composer-tools').getBoundingClientRect();
        return signal.width > 130 && signal.top >= tools.bottom && signal.right <= controls.right;
      })()`), true);
      await send('Emulation.clearDeviceMetricsOverride');
      await shortcut('b');
      if (!realWhisper) {
        await wait(async () => (await draftText()).startsWith('Existing draft. Hello from'), 'dictation to cancel');
      }
      await evaluate("document.querySelector('#voice-cancel').focus()");
      await click('#voice-cancel');
      await wait(() => contains('#voice-status', 'Dictation canceled'), 'canceled recording');
      assert.equal(await evaluate("document.activeElement.id"), 'dictate');
      assert.equal(await evaluate("document.querySelector('#voice-signal').hidden"), true);
      assert.equal(await draftText(), 'Existing draft.');

      await type('#draft', '');
      assert.equal(await evaluate("document.querySelector('#draft').matches(':placeholder-shown')"), true);
      await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] });
      await click('#dictate');
      await wait(() => contains('#voice-status', 'Recording'), 'reduced-motion recording');
      assert.equal(await evaluate("getComputedStyle(document.querySelector('#voice-waveform')).display"), 'none');
      await wait(() => evaluate("document.querySelector('#voice-level').value > 0"), 'reduced-motion microphone level');
      if (!realWhisper) {
        await wait(async () => (await draftText()) === 'Hello from', 'dictation in an empty draft');
        assert.equal(await evaluate("document.querySelector('#draft').matches(':placeholder-shown')"), false);
      }
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
      await wait(() => contains('#voice-status', 'Dictation canceled'), 'canceled recording');
      assert.equal(await draftText(), '');
      assert.equal(await evaluate("document.querySelector('#draft').matches(':placeholder-shown')"), true);
      await send('Emulation.setEmulatedMedia', { features: [] });
      await type('#draft', 'Existing draft.');
      await click('#dictate');
      await wait(() => contains('#voice-status', 'Recording'), 'recording to transcribe');
      if (INFERA_TEST_SCREENSHOT) {
        await send('Emulation.setDeviceMetricsOverride', { width: 760, height: 600, deviceScaleFactor: 1, mobile: false });
        await wait(() => evaluate("document.querySelector('#voice-waveform-bars').getBBox().height > 5"), 'visible waveform');
        const recording = await send('Page.captureScreenshot', { format: 'png' });
        fs.writeFileSync(INFERA_TEST_SCREENSHOT, Buffer.from(recording.data, 'base64'));
      }
      if (realWhisper) {
        const pattern = new RegExp(realWhisper, 'i');
        await wait(async () => pattern.test((await draftText()).slice('Existing draft. '.length)),
          'recognized words in the draft as you speak', 60_000);
      } else {
        await wait(async () => (await draftText()) === 'Existing draft. Hello from', 'first live transcript in the draft');
        await wait(async () => (await draftText()) === 'Existing draft. Hello from voice', 'revised transcript in the draft');
      }
      assert.equal(await evaluate("document.querySelector('#dictate').dataset.state"), 'recording');
      assert.equal(await evaluate("document.querySelector('#voice-preview')"), null);
      if (INFERA_TEST_SCREENSHOT) {
        const preview = await send('Page.captureScreenshot', { format: 'png' });
        fs.writeFileSync(INFERA_TEST_SCREENSHOT.replace(/\.png$/, '-live.png'), Buffer.from(preview.data, 'base64'));
        await click('#sidebar-close');
        await send('Emulation.setDeviceMetricsOverride', { width: 410, height: 560, deviceScaleFactor: 1, mobile: false });
        assert.equal(await evaluate(`(() => {
          const draft = document.querySelector('#draft').getBoundingClientRect();
          const composer = document.querySelector('#composer').getBoundingClientRect();
          return draft.left >= composer.left && draft.right <= composer.right && draft.height > 0;
        })()`), true);
        const narrow = await send('Page.captureScreenshot', { format: 'png' });
        fs.writeFileSync(INFERA_TEST_SCREENSHOT.replace(/\.png$/, '-live-narrow.png'), Buffer.from(narrow.data, 'base64'));
        await send('Emulation.clearDeviceMetricsOverride');
      }
      await delay(220);
      await click('#dictate');
      await wait(() => contains('#voice-status', 'Transcribing'), 'local transcription');
      assert.equal(await evaluate("document.querySelector('#voice-signal').hidden"), true);
      await wait(() => contains('#voice-status', 'Dictation added'), 'editable dictation',
        process.env.INFERA_TEST_VOICE_MATCH ? 120_000 : 12_000);
      assert.match(await draftText(),
        new RegExp('^Existing draft\\. ' + (process.env.INFERA_TEST_VOICE_MATCH || 'Hello from voice input\\.$'), 'i'));
      assert.equal(await evaluate("document.querySelector('#draft').readOnly"), false);
      assert.equal(await evaluate("document.querySelectorAll('#messages .message').length"), 0);
      await click('#send');
      await wait(() => contains('#messages', 'Hello world'), 'the reply to the dictated prompt');
      return;
    }
    if (await evaluate("document.querySelector('#dictate').ariaLabel === 'Dictate message'")) {
      assert.equal(await evaluate("document.querySelector('#dictate').ariaLabel"), 'Dictate message');
    } else {
      await click('#dictate');
      assert.equal(await contains('#modal-content', 'Choose Whisper files'), true);
      await click('#modal-close');
    }
    assert.equal(await evaluate("document.querySelector('#server').title"), 'Inference server · Ready · CPU');
    await click('#server');
    assert.equal(await contains('#modal-title', 'Inference server'), true);
    await click('#modal-close');
    await click('#advanced-stats');
    assert.equal(await contains('#analytics-overview', 'Send a message'), true);
    await click('#analytics-recent-tab');
    assert.equal(await contains('#analytics-recent', 'Recorded responses will appear'), true);
    await click('#analytics-close');
    await wait(() => evaluate("!document.querySelector('#kv-occupancy').hidden"), 'reported KV occupancy');
    assert.equal(await contains('#kv-occupancy', '12.5%'), true);
    await wait(() => evaluate("document.querySelector('#context').title.includes('tokens')"), 'the model token budget');
    const pressed = (id, value) => evaluate(`document.querySelector('#${id}').getAttribute('aria-pressed') === '${value}'`);
    if (INFERA_TEST_PROJECTOR) {
      assert.equal(await pressed('mtp', true), true);
      await click('#mtp');
      await wait(async () => await status('Ready') && await pressed('mtp', false), 'MTP to turn off');
      await click('#mtp');
      await wait(async () => await status('Ready') && await pressed('mtp', true), 'MTP to turn back on');
      await click('#thinking');
      await wait(async () => await status('Ready') && await pressed('thinking', true), 'Thinking to turn on');
      await click('#vision');
      await wait(() => contains('#modal-title', 'Vision'), 'the vision projector picker');
      await inputWithLabel('Vision projector', INFERA_TEST_PROJECTOR);
      await click('#modal-actions [type="submit"]');
      await wait(() => evaluate("!document.querySelector('#modal').open && document.querySelector('#vision').dataset.active === 'true'"),
        'vision to enable after loading the projector');
    }
    if (INFERA_TEST_ATTACHMENT) {
      const { root: documentNode } = await send('DOM.getDocument');
      const { nodeId } = await send('DOM.querySelector', { nodeId: documentNode.nodeId, selector: '#file-picker' });
      const selectFile = () => send('DOM.setFileInputFiles', { nodeId, files: [INFERA_TEST_ATTACHMENT] });
      await selectFile();
      await wait(() => contains('#draft-files', 'attachment.md'), 'the draft attachment');
      assert.equal(await evaluate("document.querySelector('#send').disabled"), false);
      await click('#draft-files button');
      assert.equal(await evaluate("document.querySelector('#send').disabled"), true);
      await selectFile();
      await wait(() => contains('#draft-files', 'attachment.md'), 'the reattached file');
    }
    if (INFERA_TEST_IMAGE) {
      const { root: documentNode } = await send('DOM.getDocument');
      const { nodeId } = await send('DOM.querySelector', { nodeId: documentNode.nodeId, selector: '#image-picker' });
      await send('DOM.setFileInputFiles', { nodeId, files: [INFERA_TEST_IMAGE] });
      await wait(() => contains('#draft-files', 'pixel.png'), 'the draft image');
    }
    await type('#draft', 'Hello through Electron');
    await click('#sidebar-close');
    assert.equal(await evaluate("document.querySelector('#sidebar').hidden && document.querySelector('main').offsetLeft === 0"), true);
    assert.equal(await evaluate("document.activeElement.id"), 'sidebar-open');
    assert.equal(await evaluate("document.activeElement.getAttribute('aria-expanded')"), 'false');
    await click('#sidebar-open');
    assert.equal(await evaluate("document.activeElement.id"), 'sidebar-close');
    await shortcut('b');
    assert.equal(await evaluate("document.querySelector('#sidebar').hidden"), true);
    await shortcut('k');
    assert.equal(await evaluate("document.activeElement.id"), 'search');
    assert.equal(await evaluate("document.querySelector('#sidebar').hidden"), false);
    assert.equal(await evaluate("document.querySelector('#draft').value"), 'Hello through Electron');

    await click('#send');
    await wait(() => contains('#messages', 'Hello world'), 'the streamed response');
    const latestPerformance = '#messages .assistant:last-child .message-performance';
    await wait(() => contains(latestPerformance + ' summary', '20.0 generation tok/s'), 'the saved generation timings');
    assert.equal(await evaluate("document.querySelector('#request-performance')"), null);
    await click(latestPerformance + ' summary');
    assert.equal(await contains(latestPerformance, 'Time to first token'), true);
    assert.equal(await contains(latestPerformance, '64.0 MiB'), true);
    assert.equal(await contains(latestPerformance, '2 tokens · 50.0%'), true);
    assert.equal(await contains(latestPerformance, '6 / '), true);
    await evaluate(`(() => {
      const prototype = CanvasRenderingContext2D.prototype;
      const bezier = prototype.bezierCurveTo;
      window.drawnCurves = 0;
      prototype.bezierCurveTo = function (...points) {
        window.drawnCurves++;
        return bezier.apply(this, points);
      };
    })()`);
    await click(latestPerformance + ' .analytics-link');
    assert.equal(await contains('#analytics-overview h3', 'Request 1'), true);
    assert.equal(await contains('#analytics-overview .analytics-kpis', '20.0 tok/s'), true);
    assert.equal(await contains('#analytics-overview', '50.0%'), true);
    assert.equal(await evaluate(`(() => {
      const dialog = document.querySelector('#inference-analytics');
      const plots = [...dialog.querySelectorAll('#analytics-overview > .analytics-grid canvas:not([hidden])')];
      return dialog.open && Math.abs(dialog.offsetWidth / innerWidth - .9) < .01
        && plots.length === 2 && plots.every(plot => plot.clientWidth > 0) && window.drawnCurves > 0;
    })()`), true);
    await evaluate(`(() => {
      const chart = document.querySelector('#analytics-overview canvas');
      chart.focus();
      chart.dispatchEvent(new KeyboardEvent('keydown', { key: 'End', bubbles: true }));
    })()`);
    assert.equal(await contains('#analytics-overview .analytics-chart-reading', 'Generation 20 tok/s'), true);
    const detailCharts = '#analytics-overview .analytics-detail-charts';
    await click('#analytics-overview .analytics-measurements summary');
    await wait(() => evaluate(`[...document.querySelectorAll('${detailCharts} canvas')]
      .every(chart => !chart.hidden && chart.clientWidth > 0 && chart.width >= chart.clientWidth)`),
      'the timing, context and KV cache graphs to render');
    assert.equal(await evaluate(`(() => {
      const timing = [...document.querySelectorAll('.analytics-timing-bars dt')]
        .find(label => label.textContent === 'Generation').nextElementSibling;
      return timing.textContent === '100 ms'
        && parseFloat(timing.firstElementChild.style.getPropertyValue('--duration-share')) > 0;
    })()`), true);
    assert.deepEqual(await evaluate(`[...document.querySelectorAll('${detailCharts} canvas')].map(chart => {
      chart.focus({ preventScroll: true });
      chart.dispatchEvent(new KeyboardEvent('keydown', { key: 'End' }));
      return chart.nextElementSibling.textContent.split(' · ').slice(1);
    })`), [
      ['Prompt tokens 4', 'Newly processed 2', 'Reused from KV cache 2'],
      ['Context used 0.04%'],
      ['Prompt reused 50.00%'],
      ['Allocated 64.0 MiB'],
    ]);
    await click('#analytics-recent-tab');
    assert.equal(await evaluate("document.querySelectorAll('#analytics-recent canvas:not([hidden])').length"), 2);
    await click('#analytics-recent tbody button');
    assert.equal(await contains('#analytics-overview h3', 'Request 1'), true);
    await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
    await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
    assert.equal(await evaluate("document.querySelector('#inference-analytics').open"), false);
    assert.equal(await evaluate("document.activeElement.matches('.message-performance button')"), true);
    await click(latestPerformance + ' summary');
    assert.equal(await evaluate("document.querySelector('#draft').value"), '');
    if (INFERA_TEST_ATTACHMENT) {
      assert.equal(await evaluate("document.querySelector('#draft-files').hidden"), true);
      await click('.message-files summary');
      assert.equal(await contains('.message-files pre', '<b>Local attachment snapshot</b>'), true);
      assert.equal(await evaluate("Boolean(document.querySelector('.message-files pre b'))"), false);
    }
    if (INFERA_TEST_IMAGE) {
      await click('.message-files details:has(img) summary');
      await wait(() => evaluate("document.querySelector('.message-files img').naturalWidth === 1"), 'the stored image preview');
    }
    await click('.message.assistant .message-reasoning summary');
    assert.equal(await contains('.message.assistant .message-reasoning', 'private'), true);
    await click('#context');
    await wait(() => contains('#modal-content', 'Successfully loaded slot'), 'the context and memory report');
    assert.equal(await contains('#modal-content', '20.0 generation tok/s'), true);
    if (INFERA_TEST_ATTACHMENT) assert.equal(await contains('#modal-content', 'MTP enabled with the embedded head'), true);
    await click('#modal-close');

    await evaluate(`(() => {
      const stats = document.querySelector('#system-stats');
      const observer = new MutationObserver(() => {
        if (document.querySelector('#status-dot').dataset.state === 'Generating') {
          stats.dataset.updatedDuringGeneration = 'true';
          observer.disconnect();
        }
      });
      observer.observe(stats, { childList: true, subtree: true });
    })()`);
    await type('#draft', 'slow while switching conversations');
    await click('#send');
    await click('#advanced-stats');
    await wait(() => contains('#analytics-overview .analytics-live', 'Processing prompt'), 'live prompt processing');
    await wait(() => contains('#analytics-overview .performance-details', '80.0 tok/s'), 'progressive inference timings');
    await wait(() => status('Generating'), 'streaming before switching conversations');
    assert.equal(await contains('#analytics-overview h3', 'Current request'), true);
    assert.equal(await evaluate("document.querySelectorAll('#analytics-overview > .analytics-grid canvas:not([hidden])').length"), 2);
    const elapsed = await evaluate("document.querySelector('#analytics-elapsed').textContent");
    await wait(() => evaluate(`document.querySelector('#analytics-elapsed').textContent !== ${JSON.stringify(elapsed)}`),
      'the live analytics clock to advance');
    await click('#analytics-close');
    await click('#new-chat');
    await wait(() => contains('#chat-title', 'New chat'), 'opening another conversation during generation');
    assert.equal(await status('Generating'), true);
    assert.equal(await contains('#messages', 'slow while switching conversations'), false);
    const source = '.history-name[title="Hello through Electron"]';
    await click(source);
    await wait(() => contains('#messages', 'slow while switching conversations'), 'the original streaming turn');
    const latestAnswer = '#messages .assistant:last-child .response-step > .message-body';
    await wait(() => contains(latestAnswer, 'Hello'), 'restored live output');
    assert.equal(await evaluate(`document.querySelector(${JSON.stringify(latestPerformance)}).hidden`), true);
    await click('.history-name[title="New chat"]');
    await wait(() => contains('#chat-title', 'New chat'), 'switching away before completion');
    await wait(() => status('Ready'), 'the background response to finish');
    assert.equal(await evaluate("document.querySelector('#system-stats').dataset.updatedDuringGeneration"), 'true');
    assert.equal(await contains('#messages', 'Hello world'), false);
    await click(source);
    await wait(() => contains(latestAnswer, 'Hello world'), 'the completed background response');
    assert.equal(await contains(latestPerformance + ' summary', '20.0 generation tok/s'), true);
    await wait(() => status('Ready'), 'the source model to be ready');
    await click('#advanced-stats');
    await evaluate("window.drawnCurves = 0");
    await click('#analytics-recent-tab');
    assert.equal(await evaluate("window.drawnCurves > 0"), true);
    assert.equal(await evaluate("document.querySelectorAll('#analytics-recent tbody tr').length"), 2);
    await click('#analytics-recent tbody tr:last-child button');
    assert.equal(await contains('#analytics-overview h3', 'Request 1'), true);
    await click('#analytics-close');

    await type('#draft', 'cancel this response');
    await click('#send');
    await wait(() => status('Generating'), 'streaming before cancellation');
    await click('#stop');
    await wait(() => contains('#messages', 'Stopped'), 'the stopped response');

    await click('.message.assistant button:nth-child(2)');
    await wait(() => contains('#chat-title', 'branch'), 'a regenerated branch');
    await wait(() => status('Ready'), 'regeneration to finish');
    await type('#search', 'Hello through Electron');
    await wait(() => evaluate("document.querySelectorAll('.history-entry').length === 2"), 'preserved source history');

    await click('#chat-actions');
    assert.equal(await contains('#pin-chat', 'Pin to top'), true);
    await click('#pin-chat');
    await wait(() => contains('#pinned-chats', 'branch'), 'the pinned branch');
    assert.equal(await contains('#recent-chats', 'branch'), false);
    assert.ok(await evaluate(`document.querySelector('#recent-history').getBoundingClientRect().top -
      document.querySelector('#pinned-history').getBoundingClientRect().bottom >= 24`));
    assert.equal(await evaluate("document.querySelector('#pinned-chats .history-name').getAttribute('aria-current')"), 'true');
    assert.equal(await evaluate("document.querySelector('#pinned-count').hidden"), true);
    await type('#search', '');
    await wait(() => evaluate("!document.querySelector('#pinned-count').hidden"), 'the pinned chat count');
    assert.equal(await contains('#pinned-count', '1/10'), true);
    await click('#pinned-chats .history-actions');
    assert.equal(await contains('#pin-chat', 'Unpin'), true);
    await click('#pin-chat');
    await wait(() => contains('#recent-chats', 'branch'), 'the unpinned branch');
    assert.equal(await evaluate("document.querySelector('#pinned-history').hidden"), true);

    await click('#model-settings');
    await wait(() => contains('#modal-title', 'Model settings'), 'model settings');
    const contextSlider = '#settings-runtime input[type="range"]';
    const contextValue = () => evaluate(`document.querySelector('${contextSlider}').getAttribute('aria-valuetext')`);
    assert.equal(await contextValue(), 'Automatic');
    await evaluate(`document.querySelector('${contextSlider}').focus()`);
    for (const [key, windowsVirtualKeyCode, expected] of [
      ['End', 35, '16,384 tokens'], ['ArrowLeft', 37, '8,192 tokens'],
      ['Home', 36, 'Automatic'], ['ArrowRight', 39, '4,096 tokens'],
    ]) {
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key, code: key, windowsVirtualKeyCode });
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key, code: key, windowsVirtualKeyCode });
      assert.equal(await contextValue(), expected);
    }
    await wait(() => contains('.model-memory-estimate', '4,096 tokens'), 'the selected context estimate');
    await inputWithLabel('Maximum output tokens', '0');
    await click('[aria-controls="settings-sampling"]');
    assert.equal(await evaluate("document.querySelector('#settings-runtime').hidden"), true);
    await click('#modal-actions [type="submit"]');
    assert.equal(await evaluate("document.activeElement.matches('#settings-runtime input:invalid')"), true);
    await inputWithLabel('Maximum output tokens', '512');
    if (INFERA_TEST_ATTACHMENT) await inputWithLabel('Multi-token prediction (MTP)', 'Off');
    await click('[aria-controls="settings-instructions"]');
    assert.equal(await evaluate("document.querySelector('#settings-instructions').hidden"), false);
    await click('#modal-actions [type="submit"]');
    await wait(() => evaluate("!document.querySelector('#modal').open"), 'model settings to save');
    await wait(() => status('Ready'), 'the configured model to load');

    await click('#models');
    await click('#library');
    await click('#modal-actions button:nth-child(2)');
    assert.equal(await evaluate("document.querySelector('#settings-folders').hidden"), false);
    await click('#modal-close');

    await click('#web-search');
    assert.equal(await evaluate("document.querySelector('#settings-web-search').hidden"), false);
    assert.equal(await evaluate("document.querySelector('#settings-web-search input').type"), 'password');
    await inputWithLabel('Tavily API key', 'tvly-electron-fixture');
    await inputWithLabel('Theme', 'Light');
    await click('[aria-controls="settings-folders"]');
    assert.equal(await contains('#settings-folders', 'GGUF model folders'), true);
    await click('[aria-controls="settings-inference"]');
    assert.equal(await contains('#settings-inference', 'Inference executables'), true);
    await click('[aria-controls="settings-stats"]');
    assert.equal(await evaluate("document.querySelectorAll('#settings-stats input:checked').length"), 6);
    await evaluate("document.querySelectorAll('#settings-stats input').forEach(input => { input.checked = false; })");
    await click('#modal-actions [type="submit"]');
    await wait(() => evaluate("document.documentElement.dataset.theme === 'Light'"), 'the light theme to apply');
    await wait(() => evaluate("!document.querySelector('#modal').open"), 'Tavily settings to save');
    const searchEnabled = () => evaluate("document.querySelector('#web-search').getAttribute('aria-pressed')");
    assert.equal(await searchEnabled(), 'false');
    await click('#web-search');
    assert.equal(await searchEnabled(), 'true');
    await click('#web-search');
    assert.equal(await searchEnabled(), 'false');
    assert.equal(await evaluate("document.querySelector('#system-stats').hidden"), true);
    assert.equal(await evaluate("document.querySelector('#system-stats-toggle').hidden"), true);
    await click('#settings');
    await click('[aria-controls="settings-web-search"]');
    assert.equal(await evaluate("document.querySelector('#settings-web-search input').placeholder"), 'Saved key');
    assert.equal(await evaluate("document.querySelector('#settings-web-search input').value"), '');
    await click('[aria-controls="settings-stats"]');
    assert.equal(await evaluate("document.querySelectorAll('#settings-stats input:checked').length"), 0);
    await evaluate(`document.querySelectorAll('#settings-stats label').forEach(label => {
      label.querySelector('input').checked = ['CPU load', 'RAM usage'].includes(label.textContent);
    })`);
    await click('#modal-actions [type="submit"]');
    await wait(() => evaluate("!document.querySelector('#system-stats-toggle').hidden"), 'selected statistics');
    assert.deepEqual(await evaluate(`[...document.querySelectorAll('#system-stats [data-stat]:not([hidden])')]
      .map(({ dataset }) => dataset.stat)`), ['cpu_load', 'ram']);

    const markup = '<script>alert(1)</script>\n\n![hotlink](https://example.org/a.png)\n\n'
      + '[local](file:///etc/passwd)\n\n```js\nreturn 1;\n```';
    const safe = await evaluate(`(() => {
      const target = document.createElement('div');
      window.InferaStudioMarkdown.render(target, ${JSON.stringify(markup)});
      return [Boolean(target.querySelector('script, img, a')), Boolean(target.querySelector('[data-copy]'))];
    })()`);
    assert.deepEqual(safe, [false, true]);
  } catch (error) {
    if (session && active) {
      console.error('Desktop state:', await evaluate('document.body.innerText').catch(() => 'Unavailable'));
      console.error('Renderer diagnostics:', JSON.stringify(diagnostics));
    }
    throw error;
  } finally {
    if (active) {
      app.kill('SIGTERM');
      const force = setTimeout(() => app.kill('SIGKILL'), 24_000);
      try {
        const [code, signal] = await exited;
        assert.equal(signal, null);
        assert.equal(code, 0);
      } finally {
        clearTimeout(force);
      }
    }
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
