import numpy as np
import numpy.typing as npt

def resonate(
    input: npt.NDArray[np.float32] | list[float],
    sample_rate: float,
    frequencies: npt.NDArray[np.float32] | list[float],
    alphas: npt.NDArray[np.float32] | list[float],
    betas: npt.NDArray[np.float32] | list[float],
    hop_length: int,
) -> npt.NDArray[np.float32]: ...

def alpha_heuristic(
    frequencies: npt.NDArray[np.float32] | list[float],
    sample_rate: float,
) -> npt.NDArray[np.float32]: ...

def midi_piano_frequencies(tuning: float = 440.0) -> npt.NDArray[np.float32]: ...

def log_frequencies(fmin: float, n_bins: int, bins_per_octave: int) -> npt.NDArray[np.float32]: ...
