"""Generate reference test vectors using noFFT C++ (macOS only).

Run with: uv run python generate_reference.py

Outputs are saved to ../tests/fixtures/ as .npy files and can be
loaded on any platform for cross-validation.
"""

import numpy as np
from pathlib import Path
from noFFT import resonate

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

SR = 44100.0
HOP = 256

# Test case 1: 88 bins (MIDI piano), 440 Hz sine, 1 second
freqs_88 = (440.0 * 2.0 ** ((np.arange(21, 109) - 69) / 12.0)).astype(np.float32)
alphas_88 = (1.0 - np.exp(-(1.0 / SR) * freqs_88 / np.log10(1.0 + freqs_88))).astype(np.float32)
signal_sine = np.cos(2 * np.pi * 440.0 * np.arange(int(SR)) / SR).astype(np.float32)

ref_sine = resonate(signal_sine, SR, freqs_88, alphas_88, alphas_88, HOP)

# Test case 2: 88 bins, log chirp A0-C8, 1 second
f0, f1 = freqs_88[0], freqs_88[-1]
t = np.linspace(0, 1.0, int(SR), dtype=np.float64)
signal_chirp = np.cos(
    2 * np.pi * f0 * 1.0 / np.log(f1 / f0) * (np.exp(t / 1.0 * np.log(f1 / f0)) - 1)
).astype(np.float32)

ref_chirp = resonate(signal_chirp, SR, freqs_88, alphas_88, alphas_88, HOP)

# Test case 3: 88 bins, silence, 0.5 seconds
signal_silence = np.zeros(int(SR * 0.5), dtype=np.float32)
ref_silence = resonate(signal_silence, SR, freqs_88, alphas_88, alphas_88, HOP)

# Save everything
np.save(FIXTURE_DIR / "freqs_88.npy", freqs_88)
np.save(FIXTURE_DIR / "alphas_88.npy", alphas_88)
np.save(FIXTURE_DIR / "signal_sine_440.npy", signal_sine)
np.save(FIXTURE_DIR / "signal_chirp.npy", signal_chirp)
np.save(FIXTURE_DIR / "signal_silence.npy", signal_silence)
np.save(FIXTURE_DIR / "ref_sine_440.npy", ref_sine)
np.save(FIXTURE_DIR / "ref_chirp.npy", ref_chirp)
np.save(FIXTURE_DIR / "ref_silence.npy", ref_silence)

# Save metadata
metadata = {
    "sample_rate": SR,
    "hop_length": HOP,
    "n_bins": len(freqs_88),
    "sine_frequency": 440.0,
    "sine_duration": 1.0,
    "chirp_duration": 1.0,
    "silence_duration": 0.5,
}
np.save(FIXTURE_DIR / "metadata.npy", metadata)

print(f"Generated fixtures in {FIXTURE_DIR}:")
for f in sorted(FIXTURE_DIR.glob("*.npy")):
    data = np.load(f, allow_pickle=True)
    if data.ndim == 0:
        print(f"  {f.name}: metadata dict")
    else:
        print(f"  {f.name}: {data.shape} {data.dtype}")
