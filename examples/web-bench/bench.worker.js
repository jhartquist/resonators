import init, { ResonatorBank } from './pkg/resonators.js';

const SAMPLE_RATE = 48000;
const N_SAMPLES = SAMPLE_RATE;
const WARMUP_ITERATIONS = 5;
const ITERATIONS = 20;
const BIN_COUNTS = [88, 264, 440, 880];
const QUANTUM = 128;
const QUANTUM_BUDGET_US = (QUANTUM / SAMPLE_RATE) * 1e6;

function logSpacedFreqs(n) {
  const freqs = new Float32Array(n);
  const lo = Math.log(30);
  const hi = Math.log(8000);
  const denom = Math.max(1, n - 1);
  for (let i = 0; i < n; i++) {
    freqs[i] = Math.exp(lo + (hi - lo) * i / denom);
  }
  return freqs;
}

function makeNoise(n) {
  const sig = new Float32Array(n);
  let x = 0xDEADBEEF | 0;
  for (let i = 0; i < n; i++) {
    x = (Math.imul(x, 1664525) + 1013904223) | 0;
    sig[i] = x / 2147483648;
  }
  return sig;
}

async function run() {
  await init();
  const signal = makeNoise(N_SAMPLES);

  for (let i = 0; i < BIN_COUNTS.length; i++) {
    const nBins = BIN_COUNTS[i];
    postMessage({ type: 'start', index: i, bins: nBins });

    const freqs = logSpacedFreqs(nBins);
    const bank = new ResonatorBank(freqs, SAMPLE_RATE);
    for (let w = 0; w < WARMUP_ITERATIONS; w++) {
      bank.process_samples(signal);
    }

    const t0 = performance.now();
    for (let it = 0; it < ITERATIONS; it++) {
      bank.process_samples(signal);
    }
    const elapsed = performance.now() - t0;
    bank.free();

    const nsPerSample = elapsed * 1e6 / ITERATIONS / N_SAMPLES;
    const usPerQuantum = nsPerSample * QUANTUM / 1000;
    const pct = (usPerQuantum / QUANTUM_BUDGET_US) * 100;
    postMessage({ type: 'row', index: i, bins: nBins, nsPerSample, usPerQuantum, pct });
  }

  postMessage({ type: 'done' });
}

run().catch((e) => postMessage({ type: 'error', message: String(e?.stack || e) }));
