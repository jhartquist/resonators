// Polyfill MUST be imported before resonators.js — wasm-bindgen's module-level
// init constructs a TextDecoder, which AudioWorkletGlobalScope doesn't provide.
// ES module imports execute in declaration order, so this works.
import './polyfill.js';
import init, { ResonatorBank, heuristic_alpha } from './pkg/resonators.js';

// Chrome's AudioWorkletGlobalScope doesn't expose `performance`. Date.now() is
// integer-ms only, but accumulating across many quanta gives a meaningful average
// since DSP work per quantum is sub-ms.
const NOW = typeof performance !== 'undefined' ? () => performance.now() : () => Date.now();

// Apply a multiplicative scale to the heuristic tau of each resonator.
// Identity: alpha = 1 - exp(-dt/tau), so scaling tau by k gives
// alpha' = 1 - (1 - alpha)^(1/k). Returns null when scale is 1.0
// so the Rust default heuristic is used directly.
function scaleAlphas(freqs, sr, scale) {
  if (scale === 1.0) return null;
  const out = new Float32Array(freqs.length);
  for (let i = 0; i < freqs.length; i++) {
    const a = heuristic_alpha(freqs[i], sr);
    out[i] = 1 - Math.pow(1 - a, 1 / scale);
  }
  return out;
}

class ResonatorsProcessor extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    this.linBank = null;
    this.logBank = null;
    this.totalDspMs = 0;
    this.frameCount = 0;

    init({ module_or_path: processorOptions.wasmModule }).then(() => {
      const sr = processorOptions.sampleRate;
      const tauScale = processorOptions.tauScale ?? 1.0;
      const linAlphas = scaleAlphas(processorOptions.linFreqs, sr, tauScale);
      const logAlphas = scaleAlphas(processorOptions.logFreqs, sr, tauScale);
      this.linBank = new ResonatorBank(processorOptions.linFreqs, sr, linAlphas);
      this.logBank = new ResonatorBank(processorOptions.logFreqs, sr, logAlphas);
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
