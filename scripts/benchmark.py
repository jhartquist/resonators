# /// script
# requires-python = "==3.13.*"
# dependencies = ["numpy", "noFFT", "resonators"]
#
# [tool.uv.sources]
# resonators = { path = "../crates/resonators-py" }
# ///
"""Throughput comparison: resonators (Rust via PyO3) vs noFFT (C++).

Prints a markdown table to stdout.
"""
import numpy as np
from time import perf_counter

import noFFT
from resonators import ResonatorBank, heuristic_alphas

SAMPLE_RATE = 44100
HOP_SIZE = 256
N_SECONDS = 5
BIN_COUNTS = [88, 264, 440, 880]
WARMUPS = 3
RUNS = 5


def make_configs(n_bins: int):
    """Return (frequencies, alphas) for a bank of n_bins log-spaced over the piano range."""
    freqs = np.geomspace(27.5, 4186.0, n_bins).astype(np.float32)
    alphas = heuristic_alphas(freqs, SAMPLE_RATE).astype(np.float32)
    return freqs, alphas


def make_signal() -> np.ndarray:
    """Generate a log-sweep sine from 27.5 Hz (A0) to 4186 Hz (C8) over N_SECONDS."""
    t = np.arange(SAMPLE_RATE * N_SECONDS, dtype=np.float32) / SAMPLE_RATE
    start_freq, end_freq = 27.5, 4186.0
    phase = (
        2 * np.pi * start_freq * N_SECONDS / np.log(end_freq / start_freq)
        * ((end_freq / start_freq) ** (t / N_SECONDS) - 1)
    )
    return (0.5 * np.cos(phase)).astype(np.float32)


def time_resonators(freqs, alphas, signal):
    """Return median wall-clock seconds to run the resonators bank over signal."""
    bank = ResonatorBank(freqs, SAMPLE_RATE, alphas=alphas)

    def run():
        bank.reset()
        return bank.resonate(signal, HOP_SIZE)

    for _ in range(WARMUPS):
        run()
    times = []
    for _ in range(RUNS):
        t0 = perf_counter()
        run()
        times.append(perf_counter() - t0)
    return float(np.median(times))


def time_nofft(freqs, alphas, signal):
    """Return median wall-clock seconds to run noFFT over signal."""
    def run():
        return noFFT.resonate(signal, SAMPLE_RATE, freqs, alphas, alphas, HOP_SIZE)

    for _ in range(WARMUPS):
        run()
    times = []
    for _ in range(RUNS):
        t0 = perf_counter()
        run()
        times.append(perf_counter() - t0)
    return float(np.median(times))


def main():
    signal = make_signal()
    rows = []
    for n_bins in BIN_COUNTS:
        freqs, alphas = make_configs(n_bins)
        seconds_resonators = time_resonators(freqs, alphas, signal)
        seconds_nofft = time_nofft(freqs, alphas, signal)
        throughput_resonators = len(signal) / seconds_resonators / 1e6
        throughput_nofft = len(signal) / seconds_nofft / 1e6
        ratio = throughput_resonators / throughput_nofft
        rows.append((n_bins, throughput_resonators, throughput_nofft, ratio))

    print("| bins | resonators       | noFFT            | ratio  |")
    print("|------|------------------|------------------|--------|")
    for n_bins, throughput_resonators, throughput_nofft, ratio in rows:
        print(
            f"| {n_bins:4d} "
            f"| {throughput_resonators:6.2f} Msamples/s "
            f"| {throughput_nofft:6.2f} Msamples/s "
            f"| {ratio:.2f}× |"
        )


if __name__ == "__main__":
    main()
