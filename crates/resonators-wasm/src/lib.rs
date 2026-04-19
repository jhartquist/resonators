use ::resonators as core;
use core::ResonatorConfig;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ResonatorBank {
    inner: core::ResonatorBank,
}

#[wasm_bindgen]
impl ResonatorBank {
    #[wasm_bindgen(constructor)]
    pub fn new(
        freqs: Vec<f32>,
        sample_rate: f32,
        alphas: Option<Vec<f32>>,
        betas: Option<Vec<f32>>,
    ) -> Result<ResonatorBank, JsError> {
        let alphas = alphas.unwrap_or_else(|| core::heuristic_alphas(&freqs, sample_rate));
        let betas = betas.unwrap_or_else(|| alphas.clone());

        if freqs.len() != alphas.len() || freqs.len() != betas.len() {
            return Err(JsError::new(
                "freqs, alphas, and betas must have the same length",
            ));
        }

        let configs: Vec<ResonatorConfig> = freqs
            .iter()
            .zip(&alphas)
            .zip(&betas)
            .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
            .collect();

        Ok(Self {
            inner: core::ResonatorBank::new(&configs, sample_rate),
        })
    }

    pub fn process_sample(&mut self, sample: f32) {
        self.inner.process_sample(sample);
    }

    pub fn process_samples(&mut self, samples: &[f32]) {
        self.inner.process_samples(samples);
    }

    pub fn resonate(&mut self, signal: &[f32], hop: usize) -> Box<[f32]> {
        let frames = self.inner.resonate(signal, hop);
        let mut out = Vec::with_capacity(frames.len() * 2);
        for c in frames {
            out.push(c.re);
            out.push(c.im);
        }
        out.into_boxed_slice()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    pub fn freq(&self, i: usize) -> f32 {
        self.inner.freq(i)
    }

    pub fn magnitude(&self, i: usize) -> f32 {
        self.inner.magnitude(i)
    }

    pub fn phase(&self, i: usize) -> f32 {
        self.inner.phase(i)
    }

    pub fn power(&self, i: usize) -> f32 {
        self.inner.power(i)
    }

    pub fn frequencies(&self) -> Box<[f32]> {
        self.inner.frequencies().into_boxed_slice()
    }

    pub fn magnitudes(&self) -> Box<[f32]> {
        self.inner.magnitudes().into_boxed_slice()
    }

    pub fn phases(&self) -> Box<[f32]> {
        self.inner.phases().into_boxed_slice()
    }

    pub fn powers(&self) -> Box<[f32]> {
        self.inner.powers().into_boxed_slice()
    }
}

#[wasm_bindgen]
pub fn heuristic_alpha(freq: f32, sample_rate: f32) -> f32 {
    core::heuristic_alpha(freq, sample_rate)
}

#[wasm_bindgen]
pub fn heuristic_alphas(freqs: &[f32], sample_rate: f32) -> Box<[f32]> {
    core::heuristic_alphas(freqs, sample_rate).into_boxed_slice()
}

#[wasm_bindgen]
pub fn alpha_from_tau(tau: f32, sample_rate: f32) -> f32 {
    core::alpha_from_tau(tau, sample_rate)
}

#[wasm_bindgen]
pub fn tau_from_alpha(alpha: f32, sample_rate: f32) -> f32 {
    core::tau_from_alpha(alpha, sample_rate)
}

#[wasm_bindgen]
pub fn midi_to_hz(midi: f32, tuning: f32) -> f32 {
    core::midi_to_hz(midi, tuning)
}
