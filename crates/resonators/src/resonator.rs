use crate::config::ResonatorConfig;
use crate::dynamics;

/// A single resonator implementing the Resonate algorithm (François, ICMC 2025).
///
/// Updates per sample:
/// ```text
/// R(t)  = (1-α) * R(t-1)  + α * x(t) * P(t)    // EWMA accumulation
/// R̃(t) = (1-β) * R̃(t-1) + β * R(t)             // output smoothing
/// P(t+1) = P(t) * W                               // phasor rotation
/// ```
///
/// Where W = exp(-i * 2π * f / sr) is precomputed at construction.
pub struct Resonator {
    frequency: f32,
    sample_rate: f32,

    alpha: f32,
    om_alpha: f32, // 1 - alpha
    beta: f32,
    om_beta: f32, // 1 - beta

    // Precomputed phasor multiplier
    w_re: f32,
    w_im: f32,

    // Phasor state (unit complex number, rotates each sample)
    z_re: f32,
    z_im: f32,

    // Unsmoothed accumulation (internal)
    r_re: f32,
    r_im: f32,

    // Smoothed output
    rr_re: f32,
    rr_im: f32,
}

impl Resonator {
    pub fn new(config: ResonatorConfig, sample_rate: f32) -> Self {
        let angle = dynamics::phasor_angle(config.frequency, sample_rate);
        Self {
            frequency: config.frequency,
            sample_rate,
            alpha: config.alpha,
            om_alpha: 1.0 - config.alpha,
            beta: config.beta,
            om_beta: 1.0 - config.beta,
            w_re: angle.cos(),
            w_im: angle.sin(),
            z_re: 1.0,
            z_im: 0.0,
            r_re: 0.0,
            r_im: 0.0,
            rr_re: 0.0,
            rr_im: 0.0,
        }
    }

    /// Process a single sample.
    #[inline]
    pub fn update_sample(&mut self, sample: f32) {
        // EWMA accumulation: R = (1-α)*R + α*x*Z
        self.r_re = self.om_alpha * self.r_re + self.alpha * sample * self.z_re;
        self.r_im = self.om_alpha * self.r_im + self.alpha * sample * self.z_im;

        // Output smoothing: R̃ = (1-β)*R̃ + β*R
        self.rr_re = self.om_beta * self.rr_re + self.beta * self.r_re;
        self.rr_im = self.om_beta * self.rr_im + self.beta * self.r_im;

        // Phasor rotation: Z *= W
        let zr = self.z_re;
        let zi = self.z_im;
        self.z_re = zr * self.w_re - zi * self.w_im;
        self.z_im = zr * self.w_im + zi * self.w_re;
    }

    /// Process a frame of samples, stabilizing the phasor at the end.
    pub fn update_frame(&mut self, frame: &[f32]) {
        for &sample in frame {
            self.update_sample(sample);
        }
        self.stabilize();
    }

    /// Normalize the phasor back to unit magnitude to prevent drift.
    fn stabilize(&mut self) {
        let mag_sq = self.z_re * self.z_re + self.z_im * self.z_im;
        let inv_mag = 1.0 / mag_sq.sqrt();
        self.z_re *= inv_mag;
        self.z_im *= inv_mag;
    }

    /// Smoothed output as (re, im).
    #[inline]
    pub fn complex(&self) -> (f32, f32) {
        (self.rr_re, self.rr_im)
    }

    /// Smoothed output power: re² + im².
    #[inline]
    pub fn power(&self) -> f32 {
        self.rr_re * self.rr_re + self.rr_im * self.rr_im
    }

    /// Smoothed output amplitude: sqrt(re² + im²).
    #[inline]
    pub fn amplitude(&self) -> f32 {
        self.power().sqrt()
    }

    pub fn frequency(&self) -> f32 {
        self.frequency
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    pub fn beta(&self) -> f32 {
        self.beta
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    #[test]
    fn test_silence_produces_zero_output() {
        let config = ResonatorConfig::with_alpha(440.0, 0.01);
        let mut res = Resonator::new(config, 44100.0);

        for _ in 0..1000 {
            res.update_sample(0.0);
        }

        assert_eq!(res.power(), 0.0);
    }

    #[test]
    fn test_resonator_responds_to_matching_frequency() {
        let freq = 440.0;
        let sr = 44100.0;
        let alpha = crate::dynamics::alpha_heuristic(freq, sr);
        let config = ResonatorConfig::with_alpha(freq, alpha);
        let mut res = Resonator::new(config, sr);

        // Feed a 440 Hz sine for 1 second
        let n = sr as usize;
        for i in 0..n {
            let sample = (TAU * freq * i as f32 / sr).cos();
            res.update_sample(sample);
        }

        // Should have significant power
        assert!(res.power() > 0.01, "power = {}", res.power());
    }

    #[test]
    fn test_resonator_rejects_distant_frequency() {
        let res_freq = 440.0;
        let sr = 44100.0;
        let alpha = crate::dynamics::alpha_heuristic(res_freq, sr);
        let config = ResonatorConfig::with_alpha(res_freq, alpha);
        let mut on_freq = Resonator::new(config, sr);
        let mut off_freq = Resonator::new(config, sr);

        let n = sr as usize;
        for i in 0..n {
            let sample = (TAU * res_freq * i as f32 / sr).cos();
            on_freq.update_sample(sample);

            // Feed a very different frequency
            let off_sample = (TAU * 1000.0 * i as f32 / sr).cos();
            off_freq.update_sample(off_sample);
        }

        // On-frequency should be much stronger
        assert!(
            on_freq.power() > off_freq.power() * 10.0,
            "on={} off={}",
            on_freq.power(),
            off_freq.power()
        );
    }

    #[test]
    fn test_phasor_stability() {
        let config = ResonatorConfig::with_alpha(440.0, 0.01);
        let mut res = Resonator::new(config, 44100.0);

        // Run for a while without stabilizing
        for i in 0..10000 {
            res.update_sample((TAU * 440.0 * i as f32 / 44100.0).cos());
        }
        res.stabilize();

        let mag_sq = res.z_re * res.z_re + res.z_im * res.z_im;
        assert!(
            (mag_sq - 1.0).abs() < 1e-6,
            "phasor magnitude squared = {}",
            mag_sq
        );
    }
}
