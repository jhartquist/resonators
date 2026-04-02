use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use resonators::{Resonator, ResonatorBank, ResonatorConfig, SimdResonatorBank};

/// Extract a Vec<f32> from any Python sequence (list, numpy array, tuple, etc.)
fn to_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try fast path: numpy f32 array
    if let Ok(arr) = obj.extract::<Vec<f32>>() {
        return Ok(arr);
    }
    // Fallback: extract as f64 and convert
    if let Ok(arr) = obj.extract::<Vec<f64>>() {
        return Ok(arr.iter().map(|&v| v as f32).collect());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a sequence of numbers (list, tuple, or numpy array)",
    ))
}

#[pyclass]
struct PyResonator {
    inner: Resonator,
}

#[pymethods]
impl PyResonator {
    #[new]
    fn new(frequency: f64, alpha: f64, beta: f64, sample_rate: f64) -> Self {
        let config = ResonatorConfig::new(frequency as f32, alpha as f32, beta as f32);
        Self {
            inner: Resonator::new(config, sample_rate as f32),
        }
    }

    fn update_sample(&mut self, sample: f32) {
        self.inner.update_sample(sample);
    }

    fn update_frame(&mut self, frame: &Bound<'_, PyAny>) -> PyResult<()> {
        let samples = to_f32_vec(frame)?;
        self.inner.update_frame(&samples);
        Ok(())
    }

    fn power(&self) -> f32 {
        self.inner.power()
    }

    fn amplitude(&self) -> f32 {
        self.inner.amplitude()
    }

    fn complex(&self) -> (f32, f32) {
        self.inner.complex()
    }

    fn frequency(&self) -> f32 {
        self.inner.frequency()
    }
}

/// Process a signal through a bank of individual resonators.
///
/// This is the "naive" approach: N separate Resonator instances,
/// each updated per sample. No SIMD. Useful as a reference.
///
/// Returns split-complex output: flat array of
/// [re_0, re_1, ..., re_N, im_0, im_1, ..., im_N] per slice.
#[pyfunction]
fn resonate_naive<'py>(
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

    // Create N individual resonators
    let mut resonators: Vec<Resonator> = freqs
        .iter()
        .zip(alphas.iter())
        .zip(betas.iter())
        .map(|((&f, &a), &b)| Resonator::new(ResonatorConfig::new(f, a, b), sr))
        .collect();

    // Process and collect output in noFFT-compatible format
    let mut result = Array1::<f32>::zeros(n_slices * two_n);

    for s in 0..n_slices {
        let frame = &signal[s * hop_length..(s + 1) * hop_length];
        for res in resonators.iter_mut() {
            res.update_frame(frame);
        }
        let offset = s * two_n;
        for (k, res) in resonators.iter().enumerate() {
            let (re, im) = res.complex();
            result[offset + k] = re;
            result[offset + n_bins + k] = im;
        }
    }

    Ok(result.into_pyarray(py))
}

/// Compute the paper's alpha heuristic for an array of frequencies.
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

/// Process a signal through a ResonatorBank (SoA layout, auto-vectorizable).
///
/// Returns split-complex output in noFFT-compatible format.
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
        result.as_slice_mut().unwrap()[offset..offset + n_bins].copy_from_slice(re);
        result.as_slice_mut().unwrap()[offset + n_bins..offset + two_n].copy_from_slice(im);
    }

    Ok(result.into_pyarray(py))
}

/// Process a signal through a SimdResonatorBank (wide f32x4 SIMD).
///
/// Returns split-complex output in noFFT-compatible format.
#[pyfunction]
fn resonate_simd<'py>(
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

    let mut bank = SimdResonatorBank::new(&configs, sr);
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

#[pymodule]
fn _resonators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonator>()?;
    m.add_function(wrap_pyfunction!(resonate, m)?)?;
    m.add_function(wrap_pyfunction!(resonate_simd, m)?)?;
    m.add_function(wrap_pyfunction!(resonate_naive, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_heuristic, m)?)?;
    m.add_function(wrap_pyfunction!(midi_piano_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(log_frequencies, m)?)?;
    Ok(())
}
