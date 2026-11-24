'use strict';

(() => {
  const waveformWidth = 256;
  const waveformHistory = 100;
  const meterInterval = 50;

  function create({ draft, call, startRequest, changed, setup }) {
    const api = window.inferaStudio;
    const button = document.getElementById('dictate');
    const icon = button.querySelector('use');
    const panel = document.getElementById('voice-panel');
    const status = document.getElementById('voice-status');
    const elapsed = document.getElementById('voice-elapsed');
    const signal = document.getElementById('voice-signal');
    const waveform = document.getElementById('voice-waveform');
    const bars = document.getElementById('voice-waveform-bars');
    const meter = document.getElementById('voice-level');
    const cancelButton = document.getElementById('voice-cancel');
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let configuration;
    let job;
    let message = '';

    function stopTracks(stream) {
      stream?.getTracks().forEach((track) => track.stop());
    }

    function monitor(recording) {
      const started = performance.now();
      let context;
      let analyser;
      try {
        context = new AudioContext();
        analyser = context.createAnalyser();
        analyser.fftSize = 2048;
        context.createMediaStreamSource(recording.stream).connect(analyser);
        void context.resume().catch(() => {});
      } catch {
        analyser = null;
      }

      const samples = new Float32Array(analyser?.fftSize ?? 0);
      const history = Array(waveformHistory).fill(0);
      let lastSample = -Infinity;
      let lastMeter = -Infinity;
      let meterPeak = 0;
      let frame;
      elapsed.textContent = '0:00';
      bars.removeAttribute('d');
      meter.value = 0;

      function update(now) {
        frame = requestAnimationFrame(update);
        if (now - lastSample < meterInterval) return;
        lastSample = now;
        const seconds = Math.max(0, Math.min(recording.maxSeconds, Math.floor((now - started) / 1000)));
        const time = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
        if (elapsed.textContent !== time) elapsed.textContent = time;

        analyser?.getFloatTimeDomainData(samples);
        let energy = 0;
        for (const sample of samples) energy += sample * sample;
        // Keep room noise near the baseline and give quieter speech a visible range.
        const rms = samples.length ? Math.sqrt(energy / samples.length) : 0;
        const level = Math.sqrt(1 - Math.exp(-3 * Math.max(0, rms - 0.003)));
        history.shift();
        history.push(level);
        if (reducedMotion.matches) {
          meterPeak = Math.max(meterPeak, level);
          if (now - lastMeter >= 200) {
            meter.value = Math.round(meterPeak * 5);
            meterPeak = 0;
            lastMeter = now;
          }
          return;
        }

        const count = Math.min(history.length, Math.max(1, Math.floor(waveform.clientWidth / 5)));
        const step = waveformWidth / count;
        const path = history.slice(-count).map((value, index) => {
          const x = ((index + 0.5) * step).toFixed(1);
          const half = 0.4 + value * 12.6;
          return `M${x} ${(15 - half).toFixed(1)}V${(15 + half).toFixed(1)}`;
        });
        bars.setAttribute('d', path.join(' '));
      }

      frame = requestAnimationFrame(update);
      return () => {
        cancelAnimationFrame(frame);
        if (context) void context.close().catch(() => {});
      };
    }

    function closeCapture(recording) {
      clearTimeout(recording.timer);
      recording.stopMonitoring?.();
      recording.stopMonitoring = null;
      stopTracks(recording.stream);
      recording.capture?.port.close();
      if (recording.context) {
        void recording.context.close().catch(() => {});
        recording.context = null;
      }
    }

    function replaceTranscript(recording, text) {
      const start = recording.draftText.length;
      const replacement = text ? recording.separator + text : '';
      if (draft.value.slice(start) === replacement) return;
      draft.setRangeText(replacement, start, draft.value.length, 'preserve');
      if (text) {
        draft.scrollTop = draft.scrollHeight;
        window.InferaStudioTextReveal.play(draft);
      }
    }

    function clear(finalText = '') {
      const stopped = job;
      job = null;
      if (!stopped) return;
      closeCapture(stopped);
      if (stopped.request) api.cancel(stopped.request.id);
      window.InferaStudioTextReveal.cancel(draft);
      replaceTranscript(stopped, finalText);
    }

    function sync(enabled) {
      if (!enabled) {
        clear();
        message = '';
      }
      const phase = job?.phase;
      const recording = phase === 'recording';
      const label = recording ? 'Stop recording and transcribe'
        : phase ? 'Cancel voice input' : configuration ? 'Dictate message' : 'Set up voice input';
      draft.readOnly = Boolean(job);
      button.disabled = !enabled || configuration === undefined;
      button.dataset.state = phase || '';
      button.setAttribute('aria-label', label);
      button.title = label;
      if (!phase && !configuration) button.setAttribute('aria-haspopup', 'dialog');
      else button.removeAttribute('aria-haspopup');
      icon.setAttribute('href', recording ? '#icon-record-stop' : phase ? '#icon-close' : '#icon-mic');
      if (phase) panel.dataset.phase = phase;
      else delete panel.dataset.phase;
      elapsed.hidden = signal.hidden = cancelButton.hidden = !recording;
      const text = { starting: 'Opening microphone…', recording: 'Recording',
        transcribing: 'Transcribing locally…' }[phase] || message;
      if (status.textContent !== text) {
        status.textContent = text;
        status.title = text;
        if (text) window.InferaStudioTextReveal.play(status);
      }
      return Boolean(job);
    }

    function finish(statusText, finalText) {
      clear(finalText);
      message = statusText;
      changed();
    }

    async function refresh() {
      configuration = await call('voice_status').catch(() => null);
      changed();
      return configuration?.source ?? null;
    }

    async function configure() {
      try {
        await api.configureVoice();
        await refresh();
      } catch (error) {
        throw new Error(error.message.replace(/^Error invoking remote method '[^']+': Error: /, ''));
      }
    }

    async function applyTranscript(recording) {
      const response = await recording.request.response;
      if (job !== recording) return;
      recording.request = null;
      if (!response.ok) {
        finish(response.error || 'Dictation canceled.');
        return;
      }
      const text = response.value;
      if (!text) {
        finish('No speech detected. Try recording again.');
        return;
      }
      const focusDraft = document.activeElement === button;
      finish('Dictation added to your draft.', text);
      draft.dispatchEvent(new Event('input', { bubbles: true }));
      if (focusDraft) {
        draft.focus();
        const end = draft.value.length;
        draft.setSelectionRange(end, end);
      }
    }

    function stop(recording) {
      if (recording !== job || recording.phase !== 'recording') return;
      const refocus = document.activeElement === cancelButton;
      recording.phase = 'transcribing';
      clearTimeout(recording.timer);
      recording.stopMonitoring?.();
      recording.stopMonitoring = null;
      stopTracks(recording.stream);
      recording.capture.port.postMessage('flush');
      recording.timer = setTimeout(() => {
        if (recording === job) finish('Could not finish capturing the recording. Try again.');
      }, 2000);
      changed();
      if (refocus) button.focus();
    }

    async function start() {
      const { sample_rate: sampleRate, batch_ms: batchMs, max_seconds: maxSeconds } = configuration;
      const draftText = draft.value;
      const separator = draftText && !/\s$/.test(draftText) ? ' ' : '';
      const recording = { phase: 'starting', draftText, separator, maxSeconds };
      job = recording;
      message = '';
      changed();
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
        if (job !== recording) {
          stopTracks(stream);
          return;
        }
        recording.stream = stream;
        for (const track of stream.getAudioTracks()) track.addEventListener('ended', () => {
          if (job === recording && recording.phase !== 'transcribing') finish('The microphone disconnected.');
        });
        const context = new AudioContext({ sampleRate });
        recording.context = context;
        await context.audioWorklet.addModule('./voice-capture.js');
        if (job !== recording) return;
        await context.resume();
        if (job !== recording) return;
        const capture = new AudioWorkletNode(context, 'voice-capture', {
          numberOfOutputs: 0, channelCount: 1, channelCountMode: 'explicit',
          processorOptions: { batchFrames: Math.round(context.sampleRate * batchMs / 1000) },
        });
        recording.capture = capture;
        capture.addEventListener('processorerror', () => {
          if (job === recording) finish('Could not capture the microphone. Try again.');
        });
        recording.request = startRequest('voice_start', {}, (event, value) => {
          if (job === recording && event === 'transcript') replaceTranscript(recording, value);
        });
        capture.port.onmessage = ({ data }) => {
          if (job !== recording) return;
          api.writeVoice(recording.request.id, data);
          if (data === null) closeCapture(recording);
        };
        void applyTranscript(recording).catch(() => {
          if (job === recording) finish('The local transcription service closed. Try again.');
        });
        context.createMediaStreamSource(stream).connect(capture);
        recording.phase = 'recording';
        recording.timer = setTimeout(() => stop(recording), maxSeconds * 1000);
        recording.stopMonitoring = monitor(recording);
        changed();
      } catch (error) {
        if (job !== recording) return;
        finish(error.name === 'NotAllowedError' ? 'Microphone access was denied. Check your system permissions.'
          : error.name === 'NotFoundError' ? 'No microphone found. Connect one and try again.'
            : 'Could not capture the microphone. Check your sound input settings.');
      }
    }

    function cancel() {
      if (!job) return false;
      const refocus = document.activeElement === cancelButton;
      finish('Dictation canceled.');
      if (refocus) button.focus();
      return true;
    }

    button.addEventListener('click', () => {
      if (job?.phase === 'recording') stop(job);
      else if (job) cancel();
      else if (configuration) void start();
      else setup();
    });
    cancelButton.addEventListener('click', cancel);
    window.addEventListener('pagehide', () => clear());
    return { sync, refresh, configure, cancel, source: () => configuration?.source ?? null,
      draftText: () => job?.draftText ?? draft.value };
  }

  window.InferaStudioVoice = { create };
})();
