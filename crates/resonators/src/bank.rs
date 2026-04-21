use std::f32::consts::PI;

use num_complex::Complex32;

use crate::STABILIZE_EVERY;
use crate::config::ResonatorConfig;
use crate::dynamics::heuristic_alphas;

/// A bank of independent resonators, each tuned to a fixed frequency.
///
/// Construct with [`from_frequencies`](ResonatorBank::from_frequencies) for
/// the common case, or [`new`](ResonatorBank::new) for custom per-resonator
/// parameters. Feed samples in one at a time with
/// [`process_sample`](ResonatorBank::process_sample) or in chunks with
/// [`process_samples`](ResonatorBank::process_samples). Read per-bin
/// magnitudes, powers, phases, or complex values at any time. For one-shot
/// processing of a full signal into a spectrogram-like output, use
/// [`resonate`](ResonatorBank::resonate).
#[derive(Debug)]
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
    /// Creates a new bank from a slice of frequencies, with
    /// [`heuristic_alpha`](crate::heuristic_alpha) used for each resonator's
    /// `alpha` and `beta`. For custom per-resonator parameters, use
    /// [`new`](ResonatorBank::new) with an explicit slice of
    /// [`ResonatorConfig`].
    pub fn from_frequencies(freqs: &[f32], sample_rate: f32) -> Self {
        let alphas = heuristic_alphas(freqs, sample_rate);
        let configs: Vec<ResonatorConfig> = freqs
            .iter()
            .zip(&alphas)
            .map(|(&f, &a)| ResonatorConfig::new(f, a, a))
            .collect();
        Self::new(&configs, sample_rate)
    }

    /// Creates a new bank with one resonator per config, all sharing the
    /// given sample rate.
    pub fn new(configs: &[ResonatorConfig], sample_rate: f32) -> Self {
        debug_assert!(
            sample_rate.is_finite() && sample_rate > 0.0,
            "sample_rate must be positive"
        );
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

    /// Updates every resonator with a single input sample.
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

    /// Updates every resonator with a block of input samples, in order.
    #[inline]
    pub fn process_samples(&mut self, samples: &[f32]) {
        for &s in samples {
            self.process_sample(s);
        }
    }

    /// Processes `signal` in hops and returns the complex state of every
    /// resonator at the end of each hop.
    ///
    /// The output is laid out row-major with shape `(n_frames, n_bins)`, where
    /// `n_frames = signal.len() / hop` and `n_bins = self.len()`. Any trailing
    /// samples (fewer than `hop`) are dropped.
    ///
    /// # Panics
    ///
    /// Panics if `hop` is `0`.
    pub fn resonate(&mut self, signal: &[f32], hop: usize) -> Vec<Complex32> {
        let n_frames = signal.len() / hop;
        let mut out = Vec::with_capacity(n_frames * self.n_resonators);
        for chunk in signal.chunks_exact(hop) {
            self.process_samples(chunk);
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

    /// Clears all accumulated state. Frequencies and time constants are
    /// preserved.
    pub fn reset(&mut self) {
        self.z_re.fill(1.0);
        self.z_im.fill(0.0);
        self.r_re.fill(0.0);
        self.r_im.fill(0.0);
        self.rr_re.fill(0.0);
        self.rr_im.fill(0.0);
        self.sample_count = 0;
    }

    /// Returns the number of resonators in the bank.
    pub fn len(&self) -> usize {
        self.n_resonators
    }

    /// Returns the resonant frequency of bin `i`, in Hz.
    pub fn freq(&self, i: usize) -> f32 {
        self.frequencies[i]
    }

    /// Returns the current power (squared magnitude) at bin `i`.
    pub fn power(&self, i: usize) -> f32 {
        self.rr_re[i] * self.rr_re[i] + self.rr_im[i] * self.rr_im[i]
    }

    /// Returns the current magnitude at bin `i`.
    pub fn magnitude(&self, i: usize) -> f32 {
        self.power(i).sqrt()
    }

    /// Returns the current phase at bin `i`, in radians.
    pub fn phase(&self, i: usize) -> f32 {
        self.rr_im[i].atan2(self.rr_re[i])
    }

    /// Returns the current complex value at bin `i`.
    pub fn complex(&self, i: usize) -> Complex32 {
        Complex32::new(self.rr_re[i], self.rr_im[i])
    }

    /// Returns a copy of every resonator's resonant frequency, in Hz.
    pub fn frequencies(&self) -> Vec<f32> {
        self.frequencies.clone()
    }

    /// Returns the current magnitude of every bin.
    pub fn magnitudes(&self) -> Vec<f32> {
        (0..self.n_resonators).map(|i| self.magnitude(i)).collect()
    }

    /// Returns the current phase of every bin, in radians.
    pub fn phases(&self) -> Vec<f32> {
        (0..self.n_resonators).map(|i| self.phase(i)).collect()
    }

    /// Returns the current power of every bin.
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
        let mut streamed = Vec::with_capacity(batch.len());
        for chunk in signal.chunks_exact(hop) {
            bank2.process_samples(chunk);
            for i in 0..bank2.len() {
                streamed.push(bank2.complex(i));
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
