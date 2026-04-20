# resonators

A Rust implementation of Alexandre François's [Resonate algorithm][paper] for low-latency spectral analysis.

See the [main repository](https://github.com/jhartquist/resonators) for details, demos, benchmarks, and attribution.

[paper]: https://alexandrefrancois.org/assets/publications/FrancoisARJ-ICMC2025.pdf

## Install

```sh
cargo add resonators
```

## Quickstart

```rust
use resonators::{ResonatorBank, ResonatorConfig, heuristic_alpha};
use std::f32::consts::PI;

let sample_rate = 44_100.0;
let freqs = [110.0, 220.0, 440.0, 880.0];
let configs: Vec<ResonatorConfig> = freqs
    .iter()
    .map(|&f| {
        let a = heuristic_alpha(f, sample_rate);
        ResonatorConfig::new(f, a, a)
    })
    .collect();
let mut bank = ResonatorBank::new(&configs, sample_rate);

let signal: Vec<f32> = (0..sample_rate as usize)
    .map(|i| (2.0 * PI * 440.0 * i as f32 / sample_rate).sin())
    .collect();
let spectrogram = bank.resonate(&signal, 256); // flat Vec<Complex32>, n_frames × n_bins
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
