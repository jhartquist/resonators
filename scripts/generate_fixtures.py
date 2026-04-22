# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy", "noFFT", "matplotlib"]
# ///
"""Generate the reference fixture for Rust integration tests.

Runs noFFT over an 88-bin piano-range log-sweep chirp and saves the complex
output to fixtures/chirp_88.npz alongside a magnitude spectrogram plot. The
fixture is the ground truth our Rust tests compare against to catch drift.
"""
import numpy as np
from pathlib import Path
import noFFT
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100.0
HOP_SIZE = 256
DURATION = 1.0
OUT = Path(__file__).resolve().parent.parent / "fixtures"


def main():
    # 88 piano keys: MIDI 21 (A0) through 108 (C8)
    midi = np.arange(21, 109)
    freqs = (440.0 * 2.0 ** ((midi - 69) / 12.0)).astype(np.float32)
    alphas = (
        1.0 - np.exp(-(1.0 / SAMPLE_RATE) * freqs / np.log10(1.0 + freqs))
    ).astype(np.float32)

    # Log-sweep chirp A0 -> C8, 0.5 amplitude
    n_samples = int(SAMPLE_RATE * DURATION)
    t = np.arange(n_samples, dtype=np.float32) / SAMPLE_RATE
    start_freq, end_freq = freqs[0], freqs[-1]
    signal = (
        0.5
        * np.cos(
            2
            * np.pi
            * start_freq
            * DURATION
            / np.log(end_freq / start_freq)
            * (np.exp(t / DURATION * np.log(end_freq / start_freq)) - 1)
        )
    ).astype(np.float32)

    # Reference output. noFFT.resonate returns a flat float array of shape
    # (num_frames * 2 * num_resonators,), interleaved as [re, im] pairs per frame.
    raw = noFFT.resonate(signal, SAMPLE_RATE, freqs, alphas, alphas, HOP_SIZE)
    assert raw.size % (2 * len(freqs)) == 0, (
        f"noFFT output size {raw.size} is not a multiple of 2 * {len(freqs)}; "
        "the output layout may have changed."
    )
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
    ax.set_title("Chirp A0 to C8 resonator magnitude")
    ax.set_yscale("log")
    fig.colorbar(ax.images[0], ax=ax, label="Magnitude")
    fig.tight_layout()
    plot_path = OUT / "chirp_88.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  plot:   {plot_path}")


if __name__ == "__main__":
    main()
