// Polyfill MUST be imported before resonators.js — wasm-bindgen's module-level
// init constructs a TextDecoder, which AudioWorkletGlobalScope doesn't provide.
// ES module imports execute in declaration order, so this works.
import './polyfill.js';
import init, { ResonatorBank } from './pkg/resonators.js';

// Chrome's AudioWorkletGlobalScope doesn't expose `performance`. Date.now() is
// integer-ms only, but accumulating across many quanta gives a meaningful average
// since DSP work per quantum is sub-ms.
const NOW = typeof performance !== 'undefined' ? () => performance.now() : () => Date.now();

class ResonatorsProcessor extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    this.linBank = null;
    this.logBank = null;
    this.totalDspMs = 0;
    this.frameCount = 0;

    init({ module_or_path: processorOptions.wasmModule }).then(() => {
      const sr = processorOptions.sampleRate;
      this.linBank = new ResonatorBank(processorOptions.linFreqs, sr);
      this.logBank = new ResonatorBank(processorOptions.logFreqs, sr);
      this.port.postMessage({ ready: true });
    });
  }

  process(inputs) {
    if (!this.linBank) return true;
    const input = inputs[0]?.[0];
    if (!input) return true;

    const t0 = NOW();
    this.linBank.process_samples(input);
    this.logBank.process_samples(input);
    this.totalDspMs += NOW() - t0;
    this.frameCount += 1;
    const dspUs = this.totalDspMs * 1000 / this.frameCount;

    const linMags = this.linBank.magnitudes();
    const logMags = this.logBank.magnitudes();
    // Track peak magnitude across both banks for live "Peak: -X dB" stat.
    let peak = 0;
    for (let i = 0; i < linMags.length; i++) if (linMags[i] > peak) peak = linMags[i];
    for (let i = 0; i < logMags.length; i++) if (logMags[i] > peak) peak = logMags[i];
    this.port.postMessage(
      { linMags, logMags, dspUs, peak },
      [linMags.buffer, logMags.buffer],
    );
    return true;
  }
}

registerProcessor('resonators-processor', ResonatorsProcessor);
