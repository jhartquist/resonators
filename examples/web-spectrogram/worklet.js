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
    this.bank = null;
    this.totalDspMs = 0;
    this.frameCount = 0;

    init({ module_or_path: processorOptions.wasmModule }).then(() => {
      const sr = processorOptions.sampleRate;
      const tauScale = processorOptions.tauScale ?? 1.0;
      const alphas = scaleAlphas(processorOptions.freqs, sr, tauScale);
      this.bank = new ResonatorBank(processorOptions.freqs, sr, alphas);
      this.port.postMessage({ ready: true });
    });
  }

  process(inputs) {
    if (!this.bank) return true;
    const input = inputs[0]?.[0];
    if (!input) return true;

    const t0 = NOW();
    this.bank.process_samples(input);
    this.totalDspMs += NOW() - t0;
    this.frameCount += 1;
    const dspUs = this.totalDspMs * 1000 / this.frameCount;

    const mags = this.bank.magnitudes();
    let peak = 0;
    for (let i = 0; i < mags.length; i++) if (mags[i] > peak) peak = mags[i];
    this.port.postMessage({ mags, dspUs, peak }, [mags.buffer]);
    return true;
  }
}

registerProcessor('resonators-processor', ResonatorsProcessor);
