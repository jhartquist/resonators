use std::f32::consts::PI;

use num_complex::Complex32;

use crate::STABILIZE_EVERY;
use crate::config::ResonatorConfig;

/// A single resonator tuned to a fixed frequency.
///
/// For most applications, prefer [`ResonatorBank`](crate::ResonatorBank),
/// which processes many resonators together. `Resonator` is useful for
/// single-bin analysis or composing custom bank-like structures.
pub struct Resonator {
    freq: f32,
    alpha: f32,
    beta: f32,

    // phasor state, rotates by phasor angle (w) each sample
    z_re: f32,
    z_im: f32,

    // phasor angle, constant
    w_re: f32,
    w_im: f32,

    // raw output of EWMA
    r_re: f32,
    r_im: f32,

    // smoothed output of EWMA
    rr_re: f32,
    rr_im: f32,

    // tracked for stabilization
    sample_count: u64,
}

impl Resonator {
    /// Creates a new resonator from the given config and sample rate.
    pub fn new(config: ResonatorConfig, sample_rate: f32) -> Self {
        let ResonatorConfig { freq, alpha, beta } = config;
        let phasor_angle = -2.0 * PI * freq / sample_rate;
        Self {
            freq,
            alpha,
            beta,
            z_re: 1.0,
            z_im: 0.0,
            w_re: phasor_angle.cos(),
            w_im: phasor_angle.sin(),
            r_re: 0.0,
            r_im: 0.0,
            rr_re: 0.0,
            rr_im: 0.0,
            sample_count: 0,
        }
    }

    /// Updates the resonator with a single input sample.
    pub fn process_sample(&mut self, sample: f32) {
        // update raw output via EWMA
        self.r_re = (1.0 - self.alpha) * self.r_re + self.alpha * sample * self.z_re;
        self.r_im = (1.0 - self.alpha) * self.r_im + self.alpha * sample * self.z_im;

        // update smoothed output via second EMWA
        self.rr_re = (1.0 - self.beta) * self.rr_re + self.beta * self.r_re;
        self.rr_im = (1.0 - self.beta) * self.rr_im + self.beta * self.r_im;

        // rotate phasor (complex multiply)
        let z_re = self.z_re;
        let z_im = self.z_im;
        self.z_re = z_re * self.w_re - z_im * self.w_im;
        self.z_im = z_re * self.w_im + z_im * self.w_re;

        // occasional phasor stabilization
        self.sample_count += 1;
        if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
            self.stabilize();
        }
    }

    /// Updates the resonator with a block of input samples, in order.
    #[inline]
    pub fn process_samples(&mut self, samples: &[f32]) {
        for &s in samples {
            self.process_sample(s);
        }
    }

    fn stabilize(&mut self) {
        let mag = (self.z_re * self.z_re + self.z_im * self.z_im).sqrt();
        self.z_re /= mag;
        self.z_im /= mag;
    }

    /// Clears all accumulated state. Frequency and time constants are
    /// preserved.
    pub fn reset(&mut self) {
        self.z_re = 1.0;
        self.z_im = 0.0;
        self.r_re = 0.0;
        self.r_im = 0.0;
        self.rr_re = 0.0;
        self.rr_im = 0.0;
        self.sample_count = 0;
    }

    /// Returns the resonant frequency, in Hz.
    pub fn freq(&self) -> f32 {
        self.freq
    }

    /// Returns the current power (squared magnitude).
    pub fn power(&self) -> f32 {
        self.rr_re * self.rr_re + self.rr_im * self.rr_im
    }

    /// Returns the current magnitude.
    pub fn magnitude(&self) -> f32 {
        self.power().sqrt()
    }

    /// Returns the current phase, in radians.
    pub fn phase(&self) -> f32 {
        self.rr_im.atan2(self.rr_re)
    }

    /// Returns the current complex value.
    pub fn complex(&self) -> Complex32 {
        Complex32::new(self.rr_re, self.rr_im)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic_alpha;

    #[test]
    fn power_is_magnitude_squared() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.rr_re = 3.0;
        r.rr_im = 4.0;
        assert_eq!(r.power(), 25.0);
    }

    #[test]
    fn magnitude_is_sqrt_of_power() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.rr_re = 3.0;
        r.rr_im = 4.0;
        assert_eq!(r.magnitude(), 5.0);
    }

    #[test]
    fn phase_uses_atan2() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.rr_re = 1.0;
        r.rr_im = 0.0;
        assert_eq!(r.phase(), 0.0);

        r.rr_re = 0.0;
        r.rr_im = 1.0;
        assert!((r.phase() - std::f32::consts::FRAC_PI_2).abs() < 1e-6);

        r.rr_re = -1.0;
        r.rr_im = 0.0;
        assert!((r.phase() - PI).abs() < 1e-6);
    }

    #[test]
    fn matched_sine_power_converges_near_one_quarter() {
        let sr = 44100.0;
        let freq = 440.0;
        let alpha = heuristic_alpha(freq, sr);
        let mut r = Resonator::new(ResonatorConfig::new(freq, alpha, alpha), sr);
        let signal: Vec<f32> = (0..2 * sr as usize)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).cos())
            .collect();
        r.process_samples(&signal);
        assert!(
            (r.power() - 0.25).abs() < 0.01,
            "power should be ~0.25, got {}",
            r.power()
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 0.01, 0.01), 44100.0);
        r.process_samples(&vec![0.5; 1000]);
        assert!(r.magnitude() > 0.0);
        r.reset();
        assert_eq!(r.complex(), Complex32::new(0.0, 0.0));
    }

    #[test]
    fn stabilize_restores_unit_magnitude() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.z_re = 3.0;
        r.z_im = 4.0;
        r.stabilize();
        assert!((r.z_re - 0.6).abs() < 1e-6);
        assert!((r.z_im - 0.8).abs() < 1e-6);
    }
}
