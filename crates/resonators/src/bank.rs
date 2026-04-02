use crate::config::ResonatorConfig;
use crate::dynamics;

/// A bank of N resonators with SoA (Structure of Arrays) layout.
///
/// All resonators are updated together per sample, enabling
/// vectorization across resonators.
pub struct ResonatorBank {
    sample_rate: f32,
    num: usize,

    frequencies: Vec<f32>,

    // EWMA coefficients
    alphas: Vec<f32>,
    om_alphas: Vec<f32>,
    betas: Vec<f32>,
    om_betas: Vec<f32>,

    // Precomputed phasor multipliers (split complex)
    w_re: Vec<f32>,
    w_im: Vec<f32>,

    // Phasor state (split complex)
    z_re: Vec<f32>,
    z_im: Vec<f32>,

    // Unsmoothed accumulation (split complex)
    r_re: Vec<f32>,
    r_im: Vec<f32>,

    // Smoothed output (split complex)
    rr_re: Vec<f32>,
    rr_im: Vec<f32>,
}

impl ResonatorBank {
    pub fn new(configs: &[ResonatorConfig], sample_rate: f32) -> Self {
        let num = configs.len();

        let mut frequencies = Vec::with_capacity(num);
        let mut alphas = Vec::with_capacity(num);
        let mut om_alphas = Vec::with_capacity(num);
        let mut betas = Vec::with_capacity(num);
        let mut om_betas = Vec::with_capacity(num);
        let mut w_re = Vec::with_capacity(num);
        let mut w_im = Vec::with_capacity(num);

        for c in configs {
            frequencies.push(c.frequency);
            alphas.push(c.alpha);
            om_alphas.push(1.0 - c.alpha);
            betas.push(c.beta);
            om_betas.push(1.0 - c.beta);

            let angle = dynamics::phasor_angle(c.frequency, sample_rate);
            w_re.push(angle.cos());
            w_im.push(angle.sin());
        }

        Self {
            sample_rate,
            num,
            frequencies,
            alphas,
            om_alphas,
            betas,
            om_betas,
            w_re,
            w_im,
            z_re: vec![1.0; num],
            z_im: vec![0.0; num],
            r_re: vec![0.0; num],
            r_im: vec![0.0; num],
            rr_re: vec![0.0; num],
            rr_im: vec![0.0; num],
        }
    }

    /// Process a single sample across all resonators.
    #[inline]
    pub fn update_sample(&mut self, sample: f32) {
        for k in 0..self.num {
            // EWMA accumulation: R = (1-α)*R + α*x*Z
            self.r_re[k] = self.om_alphas[k] * self.r_re[k] + self.alphas[k] * sample * self.z_re[k];
            self.r_im[k] = self.om_alphas[k] * self.r_im[k] + self.alphas[k] * sample * self.z_im[k];

            // Output smoothing: R̃ = (1-β)*R̃ + β*R
            self.rr_re[k] = self.om_betas[k] * self.rr_re[k] + self.betas[k] * self.r_re[k];
            self.rr_im[k] = self.om_betas[k] * self.rr_im[k] + self.betas[k] * self.r_im[k];

            // Phasor rotation: Z *= W
            let zr = self.z_re[k];
            let zi = self.z_im[k];
            self.z_re[k] = zr * self.w_re[k] - zi * self.w_im[k];
            self.z_im[k] = zr * self.w_im[k] + zi * self.w_re[k];
        }
    }

    /// Process a frame of samples, stabilizing phasors at the end.
    pub fn update_frame(&mut self, frame: &[f32]) {
        for &sample in frame {
            self.update_sample(sample);
        }
        self.stabilize();
    }

    /// Normalize phasors back to unit magnitude.
    fn stabilize(&mut self) {
        for k in 0..self.num {
            let mag_sq = self.z_re[k] * self.z_re[k] + self.z_im[k] * self.z_im[k];
            let inv_mag = 1.0 / mag_sq.sqrt();
            self.z_re[k] *= inv_mag;
            self.z_im[k] *= inv_mag;
        }
    }

    /// Borrow the smoothed output as split-complex slices.
    #[inline]
    pub fn complex(&self) -> (&[f32], &[f32]) {
        (&self.rr_re, &self.rr_im)
    }

    /// Power at a single bin.
    #[inline]
    pub fn power(&self, i: usize) -> f32 {
        self.rr_re[i] * self.rr_re[i] + self.rr_im[i] * self.rr_im[i]
    }

    /// Amplitude at a single bin.
    #[inline]
    pub fn amplitude(&self, i: usize) -> f32 {
        self.power(i).sqrt()
    }

    /// Complex value at a single bin.
    #[inline]
    pub fn complex_at(&self, i: usize) -> (f32, f32) {
        (self.rr_re[i], self.rr_im[i])
    }

    pub fn num_resonators(&self) -> usize {
        self.num
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn frequencies(&self) -> &[f32] {
        &self.frequencies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn test_silence_produces_zero() {
        let sr = 44100.0;
        let configs = vec![ResonatorConfig::with_alpha(440.0, 0.01)];
        let mut bank = ResonatorBank::new(&configs, sr);

        bank.update_frame(&vec![0.0; 1024]);

        assert_eq!(bank.power(0), 0.0);
    }

    #[test]
    fn test_peak_at_matching_frequency() {
        let sr = 44100.0;
        let freqs = [220.0, 440.0, 880.0];
        let configs: Vec<ResonatorConfig> = freqs
            .iter()
            .map(|&f| {
                let alpha = crate::dynamics::alpha_heuristic(f, sr);
                ResonatorConfig::with_alpha(f, alpha)
            })
            .collect();

        let mut bank = ResonatorBank::new(&configs, sr);

        // Feed 440 Hz sine for 1 second
        let n = sr as usize;
        for chunk in (0..n)
            .map(|i| (TAU * 440.0 * i as f32 / sr).cos())
            .collect::<Vec<f32>>()
            .chunks(256)
        {
            bank.update_frame(chunk);
        }

        // 440 Hz bin (index 1) should be strongest
        let powers: Vec<f32> = (0..3).map(|i| bank.power(i)).collect();
        assert!(
            powers[1] > powers[0] * 10.0,
            "440 Hz bin should dominate 220 Hz: {:?}",
            powers
        );
        assert!(
            powers[1] > powers[2] * 10.0,
            "440 Hz bin should dominate 880 Hz: {:?}",
            powers
        );
    }

    #[test]
    fn test_power_converges_to_theoretical_max() {
        let sr = 44100.0;
        let freq = 440.0;
        let alpha = crate::dynamics::alpha_heuristic(freq, sr);
        let configs = vec![ResonatorConfig::with_alpha(freq, alpha)];
        let mut bank = ResonatorBank::new(&configs, sr);

        // Feed matching sine for 2 seconds (well past convergence)
        let n = 2 * sr as usize;
        for chunk in (0..n)
            .map(|i| (TAU * freq * i as f32 / sr).cos())
            .collect::<Vec<f32>>()
            .chunks(256)
        {
            bank.update_frame(chunk);
        }

        // Theoretical max power for cosine input is 0.25
        let power = bank.power(0);
        assert!(
            (power - 0.25).abs() < 0.01,
            "power should converge near 0.25, got {}",
            power
        );
    }

    #[test]
    fn test_accessors() {
        let sr = 44100.0;
        let configs = vec![
            ResonatorConfig::with_alpha(440.0, 0.01),
            ResonatorConfig::new(880.0, 0.02, 0.03),
        ];
        let bank = ResonatorBank::new(&configs, sr);

        assert_eq!(bank.num_resonators(), 2);
        assert_eq!(bank.sample_rate(), sr);
        assert_eq!(bank.frequencies(), &[440.0, 880.0]);
    }
}
