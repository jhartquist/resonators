use std::f32::consts::PI;

use num_complex::Complex32;

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

#[allow(clippy::len_without_is_empty)]
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

    #[inline]
    pub fn process_samples(&mut self, samples: &[f32]) {
        for &s in samples {
            self.process_sample(s);
        }
    }

    pub fn resonate(&mut self, signal: &[f32], hop: usize) -> Vec<Complex32> {
        let n_frames = signal.len() / hop;
        let mut out = Vec::with_capacity(n_frames * self.n_resonators);
        for frame in 0..n_frames {
            self.process_samples(&signal[frame * hop..(frame + 1) * hop]);
            for (&r, &i) in self.rr_re.iter().zip(&self.rr_im) {
                out.push(Complex32::new(r, i));
            }
        }
        out
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

    pub fn complex(&self, i: usize) -> Complex32 {
        Complex32::new(self.rr_re[i], self.rr_im[i])
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
    use crate::heuristic_alpha;

    #[test]
    fn matched_sine_power_converges_near_one_quarter() {
        let sr = 44100.0;
        let freq = 440.0;
        let alpha = heuristic_alpha(freq, sr);
        let configs = vec![ResonatorConfig::new(freq, alpha, alpha)];
        let mut bank = ResonatorBank::new(&configs, sr);
        let signal: Vec<f32> = (0..2 * sr as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).cos())
            .collect();
        bank.process_samples(&signal);
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
                let a = heuristic_alpha(f, sr);
                ResonatorConfig::new(f, a, a)
            })
            .collect();
        let mut bank = ResonatorBank::new(&configs, sr);
        let signal: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).cos())
            .collect();
        bank.process_samples(&signal);
        let p = bank.powers();
        assert!(p[1] > p[0] * 10.0, "440 should dominate 220: {p:?}");
        assert!(p[1] > p[2] * 10.0, "440 should dominate 880: {p:?}");
    }

    #[test]
    fn resonate_matches_streaming() {
        let sr = 44100.0;
        let hop = 256;
        let configs = vec![
            ResonatorConfig::new(440.0, 0.01, 0.01),
            ResonatorConfig::new(880.0, 0.01, 0.01),
        ];
        let signal: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).cos())
            .collect();

        // batch
        let mut bank = ResonatorBank::new(&configs, sr);
        let batch = bank.resonate(&signal, hop);

        // streaming equivalent
        let mut bank2 = ResonatorBank::new(&configs, sr);
        let n_frames = signal.len() / hop;
        let mut streamed = Vec::with_capacity(batch.len());
        for frame in 0..n_frames {
            bank2.process_samples(&signal[frame * hop..(frame + 1) * hop]);
            for (&r, &i) in bank2.rr_re.iter().zip(&bank2.rr_im) {
                streamed.push(Complex32::new(r, i));
            }
        }
        assert_eq!(batch, streamed);
    }

    #[test]
    fn reset_clears_state() {
        let configs = vec![ResonatorConfig::new(440.0, 0.01, 0.01)];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        bank.process_samples(&vec![0.5; 1000]);
        assert!(bank.magnitude(0) > 0.0);
        bank.reset();
        assert_eq!(bank.complex(0), Complex32::new(0.0, 0.0));
    }
}
