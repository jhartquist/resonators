use ::resonators as core;
use core::ResonatorConfig;
use numpy::{Complex32, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyclass(name = "ResonatorBank")]
struct PyResonatorBank {
    inner: core::ResonatorBank,
}

#[pymethods]
impl PyResonatorBank {
    #[new]
    #[pyo3(signature = (freqs, sample_rate, *, alphas=None, betas=None))]
    fn new(
        freqs: PyReadonlyArray1<'_, f32>,
        sample_rate: f32,
        alphas: Option<PyReadonlyArray1<'_, f32>>,
        betas: Option<PyReadonlyArray1<'_, f32>>,
    ) -> PyResult<Self> {
        let freqs_slice = freqs.as_slice()?;

        let alphas_vec: Vec<f32> = match alphas {
            Some(arr) => arr.as_slice()?.to_vec(),
            None => core::heuristic_alphas(freqs_slice, sample_rate),
        };
        let betas_vec: Vec<f32> = match betas {
            Some(arr) => arr.as_slice()?.to_vec(),
            None => alphas_vec.clone(),
        };

        if freqs_slice.len() != alphas_vec.len() || freqs_slice.len() != betas_vec.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "freqs, alphas, and betas must have the same length",
            ));
        }

        let configs: Vec<ResonatorConfig> = freqs_slice
            .iter()
            .zip(&alphas_vec)
            .zip(&betas_vec)
            .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
            .collect();

        Ok(Self {
            inner: core::ResonatorBank::new(&configs, sample_rate),
        })
    }

    fn process_sample(&mut self, sample: f32) {
        self.inner.process_sample(sample);
    }

    fn process_samples(
        &mut self,
        py: Python<'_>,
        samples: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<()> {
        let slice = samples.as_slice()?;
        py.detach(|| self.inner.process_samples(slice));
        Ok(())
    }

    fn resonate<'py>(
        &mut self,
        py: Python<'py>,
        signal: PyReadonlyArray1<'_, f32>,
        hop: usize,
    ) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
        let slice = signal.as_slice()?;
        let n_bins = self.inner.len();
        let frames = py.detach(|| self.inner.resonate(slice, hop));
        let n_frames = frames.len() / n_bins;
        Ok(frames.into_pyarray(py).reshape([n_frames, n_bins])?)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn freq(&self, i: usize) -> f32 {
        self.inner.freq(i)
    }

    fn magnitude(&self, i: usize) -> f32 {
        self.inner.magnitude(i)
    }

    fn phase(&self, i: usize) -> f32 {
        self.inner.phase(i)
    }

    fn power(&self, i: usize) -> f32 {
        self.inner.power(i)
    }

    fn complex(&self, i: usize) -> Complex32 {
        self.inner.complex(i)
    }

    fn frequencies<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.frequencies().into_pyarray(py)
    }

    fn magnitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.magnitudes().into_pyarray(py)
    }

    fn phases<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.phases().into_pyarray(py)
    }

    fn powers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.powers().into_pyarray(py)
    }
}

#[pyfunction]
fn heuristic_alpha(freq: f32, sample_rate: f32) -> f32 {
    core::heuristic_alpha(freq, sample_rate)
}

#[pyfunction]
fn heuristic_alphas<'py>(
    py: Python<'py>,
    freqs: PyReadonlyArray1<'_, f32>,
    sample_rate: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let slice = freqs.as_slice()?;
    Ok(core::heuristic_alphas(slice, sample_rate).into_pyarray(py))
}

#[pyfunction]
fn alpha_from_tau(tau: f32, sample_rate: f32) -> f32 {
    core::alpha_from_tau(tau, sample_rate)
}

#[pyfunction]
fn tau_from_alpha(alpha: f32, sample_rate: f32) -> f32 {
    core::tau_from_alpha(alpha, sample_rate)
}

#[pyfunction]
fn midi_to_hz(midi: f32, tuning: f32) -> f32 {
    core::midi_to_hz(midi, tuning)
}

#[pymodule]
fn resonators(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyResonatorBank>()?;
    m.add_function(wrap_pyfunction!(heuristic_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(heuristic_alphas, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_from_tau, m)?)?;
    m.add_function(wrap_pyfunction!(tau_from_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(midi_to_hz, m)?)?;
    Ok(())
}
