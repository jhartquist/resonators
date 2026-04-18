use std::f32::consts::PI;

use crate::STABILIZE_EVERY;
use crate::config::ResonatorConfig;

pub struct ResonatorBank {
    n_resonators: usize,
    frequencies: Vec<f32>,
    alphas: Vec<f32>,
    betas: Vec<f32>,

    // phasor state, rotates by phasor angle (w) each sample
    z_re: Vec<f32>,
    z_im: Vec<f32>,

    // phasor angle, constant
    w_re: Vec<f32>,
    w_im: Vec<f32>,

    // raw output of EWMA
    r_re: Vec<f32>,
    r_im: Vec<f32>,

    // smoothed output of EWMA
    rr_re: Vec<f32>,
    rr_im: Vec<f32>,

    // tracked for stabilization
    sample_count: u64,
}

impl ResonatorBank {
    pub fn new(configs: &[ResonatorConfig], sample_rate: f32) -> Self {
        let n_resonators = configs.len();

        let mut frequencies = Vec::with_capacity(n_resonators);
        let mut alphas = Vec::with_capacity(n_resonators);
        let mut betas = Vec::with_capacity(n_resonators);
        let mut w_re = Vec::with_capacity(n_resonators);
        let mut w_im = Vec::with_capacity(n_resonators);

        for &ResonatorConfig { freq, alpha, beta } in configs {
            let phasor_angle = -2.0 * PI * freq / sample_rate;
            frequencies.push(freq);
            alphas.push(alpha);
            betas.push(beta);
            w_re.push(phasor_angle.cos());
            w_im.push(phasor_angle.sin());
        }

        Self {
            n_resonators,
            sample_count: 0,
            frequencies,
            alphas,
            betas,
            w_re,
            w_im,
            z_re: vec![1.0; n_resonators],
            z_im: vec![0.0; n_resonators],
            r_re: vec![0.0; n_resonators],
            r_im: vec![0.0; n_resonators],
            rr_re: vec![0.0; n_resonators],
            rr_im: vec![0.0; n_resonators],
        }
    }

    pub fn process_sample(&mut self, sample: f32) {
        for k in 0..self.n_resonators {
            let alpha = self.alphas[k];
            let beta = self.betas[k];
            let alpha_sample = alpha * sample;

            // EWMA accumulation
            self.r_re[k] = (1.0 - alpha) * self.r_re[k] + alpha_sample * self.z_re[k];
            self.r_im[k] = (1.0 - alpha) * self.r_im[k] + alpha_sample * self.z_im[k];

            // output smoothing
            self.rr_re[k] = (1.0 - beta) * self.rr_re[k] + beta * self.r_re[k];
            self.rr_im[k] = (1.0 - beta) * self.rr_im[k] + beta * self.r_im[k];

            // rotate phasor
            let zr = self.z_re[k];
            let zi = self.z_im[k];
            self.z_re[k] = zr * self.w_re[k] - zi * self.w_im[k];
            self.z_im[k] = zr * self.w_im[k] + zi * self.w_re[k];
        }

        self.sample_count += 1;
        if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
            self.stabilize();
        }
    }

    fn stabilize(&mut self) {
        for k in 0..self.n_resonators {
            let mag = (self.z_re[k] * self.z_re[k] + self.z_im[k] * self.z_im[k]).sqrt();
            self.z_re[k] /= mag;
            self.z_im[k] /= mag;
        }
    }

    pub fn len(&self) -> usize {
        self.n_resonators
    }

    pub fn is_empty(&self) -> bool {
        self.n_resonators == 0
    }

    pub fn freq(&self, i: usize) -> f32 {
        self.frequencies[i]
    }

    pub fn power(&self, i: usize) -> f32 {
        self.rr_re[i] * self.rr_re[i] + self.rr_im[i] * self.rr_im[i]
    }

    pub fn magnitude(&self, i: usize) -> f32 {
        self.power(i).sqrt()
    }

    pub fn phase(&self, i: usize) -> f32 {
        self.rr_im[i].atan2(self.rr_re[i])
    }

    pub fn complex(&self, i: usize) -> (f32, f32) {
        (self.rr_re[i], self.rr_im[i])
    }

    pub fn frequencies(&self) -> Vec<f32> {
        self.frequencies.clone()
    }

    pub fn magnitudes(&self) -> Vec<f32> {
        (0..self.n_resonators).map(|i| self.magnitude(i)).collect()
    }

    pub fn phases(&self) -> Vec<f32> {
        (0..self.n_resonators).map(|i| self.phase(i)).collect()
    }

    pub fn powers(&self) -> Vec<f32> {
        (0..self.n_resonators).map(|i| self.power(i)).collect()
    }
}
