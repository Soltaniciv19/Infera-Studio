'use strict';

class VoiceCapture extends AudioWorkletProcessor {
  constructor({ processorOptions: { batchFrames } }) {
    super();
    this.batchSize = batchFrames;
    this.samples = new Float32Array(this.batchSize);
    this.used = 0;
    this.finished = false;
    this.port.onmessage = () => {
      this.finished = true;
      this.sendBatch();
      this.port.postMessage(null);
    };
  }

  sendBatch() {
    if (!this.used) return;
    const batch = this.used === this.batchSize ? this.samples : this.samples.slice(0, this.used);
    this.port.postMessage(batch.buffer, [batch.buffer]);
    this.samples = new Float32Array(this.batchSize);
    this.used = 0;
  }

  process(inputs) {
    if (this.finished) return false;
    const mono = inputs[0][0];
    if (!mono) return true;
    let offset = 0;
    while (offset < mono.length) {
      const length = Math.min(mono.length - offset, this.batchSize - this.used);
      this.samples.set(mono.subarray(offset, offset + length), this.used);
      this.used += length;
      offset += length;
      if (this.used === this.batchSize) this.sendBatch();
    }
    return true;
  }
}

registerProcessor('voice-capture', VoiceCapture);
