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

    #[inline]
    pub fn process_sample(&mut self, sample: f32) {
        for k in 0..self.n_resonators {
            let alpha = self.alphas[k];
            let beta = self.betas[k];
            let alpha_sample = alpha * sample;

            // EWMA accumulation
            self.r_re[k] = (1.0 - alpha).mul_add(self.r_re[k], alpha_sample * self.z_re[k]);
            self.r_im[k] = (1.0 - alpha).mul_add(self.r_im[k], alpha_sample * self.z_im[k]);

            // output smoothing
            self.rr_re[k] = (1.0 - beta).mul_add(self.rr_re[k], beta * self.r_re[k]);
            self.rr_im[k] = (1.0 - beta).mul_add(self.rr_im[k], beta * self.r_im[k]);

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
            let inv_mag = 1.0 / (self.z_re[k] * self.z_re[k] + self.z_im[k] * self.z_im[k]).sqrt();
            self.z_re[k] *= inv_mag;
            self.z_im[k] *= inv_mag;
        }
    }

    pub fn reset(&mut self) {
        self.z_re.fill(1.0);
        self.z_im.fill(0.0);
        self.r_re.fill(0.0);
        self.r_im.fill(0.0);
        self.rr_re.fill(0.0);
        self.rr_im.fill(0.0);
        self.sample_count = 0;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha_heuristic;

    #[test]
    fn matched_sine_power_converges_near_one_quarter() {
        let sr = 44100.0;
        let freq = 440.0;
        let alpha = alpha_heuristic(freq, sr);
        let configs = vec![ResonatorConfig::new(freq, alpha, alpha)];
        let mut bank = ResonatorBank::new(&configs, sr);
        for i in 0..2 * sr as usize {
            let t = i as f32 / sr;
            bank.process_sample((2.0 * PI * freq * t).cos());
        }
        assert!(
            (bank.power(0) - 0.25).abs() < 0.01,
            "power should be ~0.25, got {}",
            bank.power(0)
        );
    }

    #[test]
    fn peaks_at_matched_bin() {
        let sr = 44100.0;
        let freqs = [220.0, 440.0, 880.0];
        let configs: Vec<_> = freqs
            .iter()
            .map(|&f| {
                let a = alpha_heuristic(f, sr);
                ResonatorConfig::new(f, a, a)
            })
            .collect();
        let mut bank = ResonatorBank::new(&configs, sr);
        for i in 0..sr as usize {
            let t = i as f32 / sr;
            bank.process_sample((2.0 * PI * 440.0 * t).cos());
        }
        let p = bank.powers();
        assert!(p[1] > p[0] * 10.0, "440 should dominate 220: {p:?}");
        assert!(p[1] > p[2] * 10.0, "440 should dominate 880: {p:?}");
    }

    #[test]
    fn reset_clears_state() {
        let configs = vec![ResonatorConfig::new(440.0, 0.01, 0.01)];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        for _ in 0..1000 {
            bank.process_sample(0.5);
        }
        assert!(bank.magnitude(0) > 0.0);
        bank.reset();
        assert_eq!(bank.complex(0), (0.0, 0.0));
    }
}
