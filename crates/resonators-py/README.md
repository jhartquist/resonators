# resonators (Python)

Python bindings for [resonators](https://github.com/jhartquist/resonators), a Rust implementation of Alexandre François's [Resonate algorithm][paper] for low-latency spectral analysis.

See the [main repository](https://github.com/jhartquist/resonators) for details, demos, benchmarks, and attribution.

[paper]: https://alexandrefrancois.org/assets/publications/FrancoisARJ-ICMC2025.pdf

## Install

```sh
pip install resonators
```

## Quickstart

```python
import numpy as np
from resonators import ResonatorBank

sample_rate = 44_100.0
freqs = np.array([110, 220, 440, 880], dtype=np.float32)
bank = ResonatorBank(freqs, sample_rate)  # alphas default to a per-frequency heuristic

t = np.arange(sample_rate, dtype=np.float32) / sample_rate
signal = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
spectrogram = bank.resonate(signal, hop=256)  # shape (n_frames, n_bins), complex64
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
