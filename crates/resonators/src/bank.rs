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
        self.process_sample_inner(sample);
        self.sample_count += 1;
        if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
            self.stabilize();
        }
    }

    /// Updates every resonator with a block of input samples, in order.
    #[inline]
    pub fn process_samples(&mut self, samples: &[f32]) {
        let mut remaining = samples;
        while !remaining.is_empty() {
            let chunk_len = remaining.len().min(self.samples_until_stabilize());
            let (chunk, rest) = remaining.split_at(chunk_len);

            for &sample in chunk {
                self.process_sample_inner(sample);
            }

            self.sample_count += chunk_len as u64;
            if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
                self.stabilize();
            }
            remaining = rest;
        }
    }

    #[inline(always)]
    fn process_sample_inner(&mut self, sample: f32) {
        // hoisted to locals so LLVM can drop bounds checks and vectorize cleanly.
        let n = self.n_resonators;
        let alphas = &self.alphas[..n];
        let betas = &self.betas[..n];
        let w_re = &self.w_re[..n];
        let w_im = &self.w_im[..n];
        let r_re = &mut self.r_re[..n];
        let r_im = &mut self.r_im[..n];
        let rr_re = &mut self.rr_re[..n];
        let rr_im = &mut self.rr_im[..n];
        let z_re = &mut self.z_re[..n];
        let z_im = &mut self.z_im[..n];

        for k in 0..n {
            let alpha = alphas[k];
            let beta = betas[k];
            let alpha_sample = alpha * sample;

            // EWMA accumulation
            r_re[k] = mul_add(1.0 - alpha, r_re[k], alpha_sample * z_re[k]);
            r_im[k] = mul_add(1.0 - alpha, r_im[k], alpha_sample * z_im[k]);

            // output smoothing
            rr_re[k] = mul_add(1.0 - beta, rr_re[k], beta * r_re[k]);
            rr_im[k] = mul_add(1.0 - beta, rr_im[k], beta * r_im[k]);

            // rotate phasor
            let zr = z_re[k];
            let zi = z_im[k];
            z_re[k] = zr * w_re[k] - zi * w_im[k];
            z_im[k] = zr * w_im[k] + zi * w_re[k];
        }
    }

    fn samples_until_stabilize(&self) -> usize {
        let offset = (self.sample_count % STABILIZE_EVERY) as usize;
        STABILIZE_EVERY as usize - offset
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

// Unfused on wasm32+simd128: `f32::mul_add` kills autovectorization there.
#[inline(always)]
fn mul_add(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        a * b + c
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        a.mul_add(b, c)
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

    #[test]
    fn single_bin_bank_matches_scalar_resonator() {
        // Bank uses mul_add in the hot loop; Resonator uses separate mul + add.
        // Results agree to within f32 rounding, not bit-for-bit.
        use crate::Resonator;

        let sr = 44100.0;
        let freq = 440.0;
        let alpha = heuristic_alpha(freq, sr);
        let config = ResonatorConfig::new(freq, alpha, alpha);
        let signal: Vec<f32> = (0..2000)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).cos())
            .collect();

        let mut r = Resonator::new(config, sr);
        r.process_samples(&signal);
        let mut bank = ResonatorBank::new(&[config], sr);
        bank.process_samples(&signal);

        let rc = r.complex();
        let bc = bank.complex(0);
        assert!(
            (rc.re - bc.re).abs() < 1e-5,
            "re drift: scalar={} bank={}",
            rc.re,
            bc.re
        );
        assert!(
            (rc.im - bc.im).abs() < 1e-5,
            "im drift: scalar={} bank={}",
            rc.im,
            bc.im
        );
    }

    #[test]
    fn resonate_empty_signal() {
        let configs = vec![ResonatorConfig::new(440.0, 0.01, 0.01)];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        assert!(bank.resonate(&[], 256).is_empty());
    }

    #[test]
    fn resonate_signal_shorter_than_hop() {
        let configs = vec![ResonatorConfig::new(440.0, 0.01, 0.01)];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        let signal = vec![0.5f32; 100];
        assert!(bank.resonate(&signal, 256).is_empty());
    }

    #[test]
    fn resonate_drops_trailing_samples() {
        let configs = vec![
            ResonatorConfig::new(440.0, 0.01, 0.01),
            ResonatorConfig::new(880.0, 0.01, 0.01),
        ];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        let hop = 256;
        let signal = vec![0.5f32; 3 * hop + 50];
        let out = bank.resonate(&signal, hop);
        assert_eq!(out.len(), 3 * bank.len());
    }

    #[test]
    #[should_panic]
    fn resonate_panics_on_zero_hop() {
        let configs = vec![ResonatorConfig::new(440.0, 0.01, 0.01)];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        let _ = bank.resonate(&[0.0; 100], 0);
    }
}
