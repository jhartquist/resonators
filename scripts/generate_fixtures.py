# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy", "noFFT", "matplotlib"]
# ///

import numpy as np
from pathlib import Path
import noFFT
import matplotlib.pyplot as plt

SR = 44100.0
HOP = 256
DURATION = 1.0
OUT = Path(__file__).resolve().parent.parent / "fixtures"

# 88 piano keys: MIDI 21 (A0) through 108 (C8)
midi = np.arange(21, 109)
freqs = (440.0 * 2.0 ** ((midi - 69) / 12.0)).astype(np.float32)
alphas = (1.0 - np.exp(-(1.0 / SR) * freqs / np.log10(1.0 + freqs))).astype(np.float32)

# Log chirp A0 → C8, 0.5 amplitude
n = int(SR * DURATION)
t = np.arange(n, dtype=np.float32) / SR
f0, f1 = freqs[0], freqs[-1]
signal = (
    0.5
    * np.cos(
        2
        * np.pi
        * f0
        * DURATION
        / np.log(f1 / f0)
        * (np.exp(t / DURATION * np.log(f1 / f0)) - 1)
    )
).astype(np.float32)

# Reference output — resonate returns flat (num_frames * 2 * num_resonators,)
raw = noFFT.resonate(signal, SR, freqs, alphas, alphas, HOP)
ref = raw.reshape(-1, 2, len(freqs))

OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "chirp_88.npz"
np.savez(out_path, freqs=freqs, alphas=alphas, signal=signal, ref=ref)

print(f"Saved to {out_path}")
print(f"  freqs:  {freqs.shape}")
print(f"  alphas: {alphas.shape}")
print(f"  signal: {signal.shape}")
print(f"  ref:    {ref.shape}")

# Plot magnitude spectrogram
magnitude = np.sqrt(ref[:, 0, :] ** 2 + ref[:, 1, :] ** 2)
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(
    magnitude.T,
    aspect="auto",
    origin="lower",
    extent=[0, DURATION, freqs[0], freqs[-1]],
    interpolation="nearest",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Chirp A0→C8 resonator magnitude")
ax.set_yscale("log")
fig.colorbar(ax.images[0], ax=ax, label="Magnitude")
fig.tight_layout()
plot_path = OUT / "chirp_88.png"
fig.savefig(plot_path, dpi=150)
print(f"  plot:   {plot_path}")
