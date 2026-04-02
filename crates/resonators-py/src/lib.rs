use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use resonators::{ResonatorBank, ResonatorConfig};

/// Extract a Vec<f32> from any Python sequence (list, numpy array, tuple, etc.)
fn to_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(arr) = obj.extract::<Vec<f32>>() {
        return Ok(arr);
    }
    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        return Ok(arr.iter().map(|&v| v as f32).collect());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a sequence of numbers (list, tuple, or numpy array)",
    ))
}

/// Process a signal through a ResonatorBank.
///
/// Returns split-complex output: [re_0..re_N, im_0..im_N] per slice,
/// flattened into a single numpy array.
#[pyfunction]
fn resonate<'py>(
    py: Python<'py>,
    input: &Bound<'py, PyAny>,
    sample_rate: f64,
    frequencies: &Bound<'py, PyAny>,
    alphas: &Bound<'py, PyAny>,
    betas: &Bound<'py, PyAny>,
    hop_length: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let signal = to_f32_vec(input)?;
    let freqs = to_f32_vec(frequencies)?;
    let alphas = to_f32_vec(alphas)?;
    let betas = to_f32_vec(betas)?;
    let sr = sample_rate as f32;
    let n_bins = freqs.len();
    let n_slices = signal.len() / hop_length;
    let two_n = 2 * n_bins;

    let configs: Vec<ResonatorConfig> = freqs
        .iter()
        .zip(alphas.iter())
        .zip(betas.iter())
        .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
        .collect();

    let mut bank = ResonatorBank::new(&configs, sr);
    let mut result = Array1::<f32>::zeros(n_slices * two_n);

    for s in 0..n_slices {
        let frame = &signal[s * hop_length..(s + 1) * hop_length];
        bank.update_frame(frame);
        let (re, im) = bank.complex();
        let offset = s * two_n;
        let out = result.as_slice_mut().unwrap();
        out[offset..offset + n_bins].copy_from_slice(re);
        out[offset + n_bins..offset + two_n].copy_from_slice(im);
    }

    Ok(result.into_pyarray(py))
}

/// Compute the paper's alpha heuristic for an array of frequencies (Equation 7).
#[pyfunction]
fn alpha_heuristic<'py>(
    py: Python<'py>,
    frequencies: &Bound<'py, PyAny>,
    sample_rate: f64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let freqs = to_f32_vec(frequencies)?;
    let alphas: Vec<f32> = freqs
        .iter()
        .map(|&f| resonators::dynamics::alpha_heuristic(f, sample_rate as f32))
        .collect();
    Ok(PyArray1::from_vec(py, alphas))
}

/// Generate MIDI piano frequencies (88 keys).
#[pyfunction]
#[pyo3(signature = (tuning=440.0))]
fn midi_piano_frequencies<'py>(py: Python<'py>, tuning: f64) -> Bound<'py, PyArray1<f32>> {
    let freqs = resonators::frequencies::midi_piano(tuning as f32);
    PyArray1::from_vec(py, freqs)
}

/// Generate log-spaced frequencies (CQT-style).
#[pyfunction]
fn log_frequencies<'py>(
    py: Python<'py>,
    fmin: f64,
    n_bins: usize,
    bins_per_octave: usize,
) -> Bound<'py, PyArray1<f32>> {
    let freqs = resonators::frequencies::log_spaced(fmin as f32, n_bins, bins_per_octave);
    PyArray1::from_vec(py, freqs)
}

#[pymodule]
fn _resonators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(resonate, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_heuristic, m)?)?;
    m.add_function(wrap_pyfunction!(midi_piano_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(log_frequencies, m)?)?;
    Ok(())
}
