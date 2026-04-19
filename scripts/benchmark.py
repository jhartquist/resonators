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

SR = 44100
HOP = 256
N_SECONDS = 5
BIN_COUNTS = [88, 264, 440, 880]
RUNS = 5


def make_configs(n_bins: int):
    freqs = np.geomspace(27.5, 4186.0, n_bins).astype(np.float32)
    alphas = heuristic_alphas(freqs, SR).astype(np.float32)
    return freqs, alphas


def make_signal() -> np.ndarray:
    t = np.arange(SR * N_SECONDS, dtype=np.float32) / SR
    f0, f1 = 27.5, 4186.0
    phase = 2 * np.pi * f0 * N_SECONDS / np.log(f1 / f0) * ((f1 / f0) ** (t / N_SECONDS) - 1)
    return (0.5 * np.cos(phase)).astype(np.float32)


def time_resonators(freqs, alphas, signal):
    bank = ResonatorBank(freqs, SR, alphas=alphas)

    def run():
        bank.reset()
        return bank.resonate(signal, HOP)

    run()  # warmup
    times = []
    for _ in range(RUNS):
        t0 = perf_counter()
        run()
        times.append(perf_counter() - t0)
    return float(np.median(times))


def time_nofft(freqs, alphas, signal):
    def run():
        return noFFT.resonate(signal, SR, freqs, alphas, alphas, HOP)

    run()  # warmup
    times = []
    for _ in range(RUNS):
        t0 = perf_counter()
        run()
        times.append(perf_counter() - t0)
    return float(np.median(times))


def main():
    signal = make_signal()
    rows = []
    for n in BIN_COUNTS:
        freqs, alphas = make_configs(n)
        t_res = time_resonators(freqs, alphas, signal)
        t_nof = time_nofft(freqs, alphas, signal)
        m_res = len(signal) / t_res / 1e6
        m_nof = len(signal) / t_nof / 1e6
        rows.append((n, m_res, m_nof, m_res / m_nof))

    print("| bins | resonators | noFFT     | ratio  |")
    print("|------|------------|-----------|--------|")
    for n, m_res, m_nof, ratio in rows:
        print(f"| {n:4d} | {m_res:6.2f} M/s | {m_nof:6.2f} M/s | {ratio:.2f}× |")


if __name__ == "__main__":
    main()
