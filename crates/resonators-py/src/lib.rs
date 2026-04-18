use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use ::resonators as core;
use core::ResonatorConfig;

#[pyclass(name = "ResonatorBank")]
struct PyResonatorBank {
    inner: core::ResonatorBank,
}

#[pymethods]
impl PyResonatorBank {
    #[new]
    fn new(
        freqs: PyReadonlyArray1<'_, f32>,
        alphas: PyReadonlyArray1<'_, f32>,
        betas: PyReadonlyArray1<'_, f32>,
        sample_rate: f32,
    ) -> PyResult<Self> {
        let freqs = freqs.as_slice()?;
        let alphas = alphas.as_slice()?;
        let betas = betas.as_slice()?;
        if freqs.len() != alphas.len() || freqs.len() != betas.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "freqs, alphas, and betas must have the same length",
            ));
        }
        let configs: Vec<ResonatorConfig> = freqs
            .iter()
            .zip(alphas)
            .zip(betas)
            .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
            .collect();
        Ok(Self {
            inner: core::ResonatorBank::new(&configs, sample_rate),
        })
    }

    fn process_sample(&mut self, sample: f32) {
        self.inner.process_sample(sample);
    }

    fn process_samples(&mut self, py: Python<'_>, samples: PyReadonlyArray1<'_, f32>) -> PyResult<()> {
        let slice = samples.as_slice()?;
        py.detach(|| self.inner.process_samples(slice));
        Ok(())
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

    fn complex(&self, i: usize) -> (f32, f32) {
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
fn alpha_heuristic(freq: f32, sample_rate: f32) -> f32 {
    core::alpha_heuristic(freq, sample_rate)
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
    m.add_function(wrap_pyfunction!(alpha_heuristic, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_from_tau, m)?)?;
    m.add_function(wrap_pyfunction!(tau_from_alpha, m)?)?;
    m.add_function(wrap_pyfunction!(midi_to_hz, m)?)?;
    Ok(())
}
