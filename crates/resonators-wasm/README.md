# resonators (WebAssembly)

WebAssembly bindings for [resonators](https://github.com/jhartquist/resonators), a Rust implementation of Alexandre François's [Resonate algorithm][paper] for low-latency spectral analysis.

See the [main repository](https://github.com/jhartquist/resonators) for details, demos, benchmarks, and attribution.

[paper]: https://alexandrefrancois.org/assets/publications/FrancoisARJ-ICMC2025.pdf

## Install

```sh
npm install resonators
```

## Quickstart

```javascript
import init, { ResonatorBank } from "resonators";
await init();

const sampleRate = 44100;
const freqs = new Float32Array([110, 220, 440, 880]);
const bank = new ResonatorBank(freqs, sampleRate);

const signal = new Float32Array(sampleRate);
for (let i = 0; i < signal.length; i++) {
  signal[i] = Math.sin(2 * Math.PI * 440 * i / sampleRate);
}
const spectrogram = bank.resonate(signal, 256); // Float32Array, interleaved [re, im, ...]
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
