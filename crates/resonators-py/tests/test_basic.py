import numpy as np
from pathlib import Path

from resonators import (
    ResonatorBank,
    alpha_from_tau,
    heuristic_alpha,
    heuristic_alphas,
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


def test_heuristic_alpha_in_range():
    for f in [27.5, 440.0, 4186.0]:
        a = heuristic_alpha(f, SAMPLE_RATE)
        assert 0.0 < a < 1.0


def test_heuristic_alphas_matches_scalar():
    freqs = np.array([27.5, 440.0, 4186.0], dtype=np.float32)
    alphas = heuristic_alphas(freqs, SAMPLE_RATE)
    for i, f in enumerate(freqs):
        assert alphas[i] == heuristic_alpha(float(f), SAMPLE_RATE)


def test_bank_default_alphas():
    # no alphas kwarg, uses heuristic_alphas internally
    freqs = np.array([220.0, 440.0, 880.0], dtype=np.float32)
    bank = ResonatorBank(freqs, SAMPLE_RATE)
    assert len(bank) == 3
    np.testing.assert_allclose(bank.frequencies(), freqs)


def test_bank_peaks_at_matched_bin():
    freqs = np.array([220.0, 440.0, 880.0], dtype=np.float32)
    bank = ResonatorBank(freqs, SAMPLE_RATE)

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

    bank = ResonatorBank(freqs, SAMPLE_RATE, alphas=alphas)
    n_frames = ref.shape[0]

    for frame in range(n_frames):
        start = frame * HOP_SIZE
        bank.process_samples(signal[start:start + HOP_SIZE])
        for bin_idx in range(len(bank)):
            c = bank.complex(bin_idx)
            assert abs(c.real - ref[frame, 0, bin_idx]) < TOLERANCE, (
                f"frame {frame} bin {bin_idx} re: {c.real} vs {ref[frame, 0, bin_idx]}"
            )
            assert abs(c.imag - ref[frame, 1, bin_idx]) < TOLERANCE


def test_bank_explicit_alphas_betas():
    freqs = np.array([440.0], dtype=np.float32)
    alphas = np.array([0.01], dtype=np.float32)
    betas = np.array([0.02], dtype=np.float32)
    bank = ResonatorBank(freqs, SAMPLE_RATE, alphas=alphas, betas=betas)
    assert len(bank) == 1


def test_reset_clears_state():
    freqs = np.array([440.0], dtype=np.float32)
    bank = ResonatorBank(freqs, SAMPLE_RATE, alphas=np.array([0.01], dtype=np.float32))
    bank.process_samples(np.full(1000, 0.5, dtype=np.float32))
    assert bank.magnitude(0) > 0.0
    bank.reset()
    assert bank.complex(0) == 0.0 + 0.0j
