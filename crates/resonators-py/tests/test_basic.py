import numpy as np
import pytest
from pathlib import Path

from resonators import (
    ResonatorBank,
    alpha_from_tau,
    alpha_heuristic,
    midi_to_hz,
    tau_from_alpha,
)

SAMPLE_RATE = 44100.0
HOP_SIZE = 256
TOLERANCE = 1e-3
FIXTURES = Path(__file__).resolve().parent.parent.parent.parent / "fixtures"


def test_midi_to_hz_a4():
    assert abs(midi_to_hz(69.0, 440.0) - 440.0) < 1e-4


def test_alpha_tau_roundtrip():
    tau = 0.05
    alpha = alpha_from_tau(tau, SAMPLE_RATE)
    assert abs(tau_from_alpha(alpha, SAMPLE_RATE) - tau) < 1e-6


def test_alpha_heuristic_in_range():
    for f in [27.5, 440.0, 4186.0]:
        a = alpha_heuristic(f, SAMPLE_RATE)
        assert 0.0 < a < 1.0


def test_bank_basic_construction():
    freqs = np.array([220.0, 440.0, 880.0], dtype=np.float32)
    alphas = np.array([alpha_heuristic(f, SAMPLE_RATE) for f in freqs], dtype=np.float32)
    bank = ResonatorBank(freqs, alphas, alphas, SAMPLE_RATE)
    assert len(bank) == 3
    np.testing.assert_allclose(bank.frequencies(), freqs)


def test_bank_peaks_at_matched_bin():
    freqs = np.array([220.0, 440.0, 880.0], dtype=np.float32)
    alphas = np.array([alpha_heuristic(f, SAMPLE_RATE) for f in freqs], dtype=np.float32)
    bank = ResonatorBank(freqs, alphas, alphas, SAMPLE_RATE)

    n = int(SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    signal = np.cos(2 * np.pi * 440.0 * t).astype(np.float32)
    bank.process_samples(signal)

    powers = bank.powers()
    assert powers[1] > powers[0] * 10
    assert powers[1] > powers[2] * 10


def test_bank_matches_fixture():
    data = np.load(FIXTURES / "chirp_88.npz")
    freqs = data["freqs"].astype(np.float32)
    alphas = data["alphas"].astype(np.float32)
    signal = data["signal"].astype(np.float32)
    ref = data["ref"]  # shape (n_frames, 2, n_bins)

    bank = ResonatorBank(freqs, alphas, alphas, SAMPLE_RATE)
    n_frames = ref.shape[0]

    for frame in range(n_frames):
        start = frame * HOP_SIZE
        bank.process_samples(signal[start:start + HOP_SIZE])
        for bin_idx in range(len(bank)):
            re, im = bank.complex(bin_idx)
            assert abs(re - ref[frame, 0, bin_idx]) < TOLERANCE, (
                f"frame {frame} bin {bin_idx} re: {re} vs {ref[frame, 0, bin_idx]}"
            )
            assert abs(im - ref[frame, 1, bin_idx]) < TOLERANCE


def test_reset_clears_state():
    freqs = np.array([440.0], dtype=np.float32)
    alphas = np.array([0.01], dtype=np.float32)
    bank = ResonatorBank(freqs, alphas, alphas, SAMPLE_RATE)
    bank.process_samples(np.full(1000, 0.5, dtype=np.float32))
    assert bank.magnitude(0) > 0.0
    bank.reset()
    assert bank.complex(0) == (0.0, 0.0)
