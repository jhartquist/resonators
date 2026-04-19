# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy", "matplotlib", "librosa", "resonators"]
#
# [tool.uv.sources]
# resonators = { path = "../crates/resonators-py" }
# ///
"""Demo: run resonators over the same audio with two different bin layouts.

Writes images/spectrograms.png with linear-spaced and log-spaced bins
side by side.
"""
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np

from resonators import ResonatorBank

SAMPLE_RATE = 44100
HOP_SIZE = 256


def run_bank(freqs: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Run a bank with default heuristic alphas, return complex spectrogram (n_frames, n_bins)."""
    bank = ResonatorBank(freqs, SAMPLE_RATE)
    return bank.resonate(signal, HOP_SIZE)


def plot_panel(ax, spec: np.ndarray, freqs: np.ndarray, title: str, show_ylabel: bool = True):
    magnitudes = np.abs(spec)  # (n_frames, n_bins)
    peak = max(magnitudes.max(), 1e-10)
    db = 20.0 * np.log10(np.maximum(magnitudes, 1e-10) / peak)
    duration = spec.shape[0] * HOP_SIZE / SAMPLE_RATE
    img = ax.imshow(
        db.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=-60,
        vmax=0,
        extent=[0, duration, 0, len(freqs)],
    )
    # Label y-axis ticks with the actual bin frequencies.
    n_ticks = 6
    tick_idx = np.linspace(0, len(freqs) - 1, n_ticks).astype(int)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_idx])
    ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return img


def main():
    # Load a librosa sample (trumpet — has clean harmonics)
    signal, _ = librosa.load(librosa.ex("trumpet"), sr=SAMPLE_RATE, mono=True)
    signal = signal.astype(np.float32)

    # Same frequency range (20 Hz – 20 kHz), two different bin-spacing strategies.
    n_bins = 256
    linear_freqs = np.linspace(20.0, 20000.0, n_bins, dtype=np.float32)
    log_freqs = np.geomspace(20.0, 20000.0, n_bins, dtype=np.float32)

    linear_spec = run_bank(linear_freqs, signal)
    log_spec = run_bank(log_freqs, signal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_panel(ax1, linear_spec, linear_freqs, f"Linear-spaced ({n_bins} bins)")
    img2 = plot_panel(ax2, log_spec, log_freqs, f"Log-spaced ({n_bins} bins)", show_ylabel=False)
    fig.colorbar(img2, ax=[ax1, ax2], label="dB", format="%+2.0f")

    Path("images").mkdir(exist_ok=True)
    out = Path("images") / "spectrograms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
