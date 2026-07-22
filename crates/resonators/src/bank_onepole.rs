//! Reformulated resonator bank: two cascaded complex one-poles, no running
//! phasor, no periodic stabilization.
//!
//! The reference [`ResonatorBank`](crate::ResonatorBank) maintains, per
//! resonator, a unit phasor `z` that it rotates by `e^{-iωΔt}` every sample and
//! must periodically renormalize (because `|w| != 1` exactly in `f32`, so `z`
//! drifts). The EWMA accumulates `α·x·z`.
//!
//! Moving into the de-rotated frame `u_n = r_n · e^{+iωnΔt}` collapses the
//! "rotate phasor + EWMA" pair into a single complex one-pole with a constant
//! complex coefficient:
//!
//! ```text
//! u_n = c·u_{n-1} + α·x_n        c = (1-α)·e^{+iωΔt}
//! v_n = d·v_{n-1} + β·u_n        d = (1-β)·e^{+iωΔt}   (output smoothing)
//! ```
//!
//! Because `|e^{-iωnΔt}| = 1`, power and magnitude are identical to the
//! reference: `|v_n| ≡ |rr_n|`. The absolute-frame complex value (and hence
//! phase) is recovered lazily at readout by multiplying by `e^{-iωnΔt}` — only
//! at hop boundaries, never per sample.
//!
//! Numerically this is strictly contractive: `|c| = (1-α) < 1`, so round-off
//! decays instead of accumulating. There is nothing to stabilize.

use std::f32::consts::PI;

use num_complex::Complex32;

use crate::config::ResonatorConfig;
use crate::dynamics::heuristic_alphas;

#[derive(Debug)]
pub struct OnePoleBank {
    n_resonators: usize,
    frequencies: Vec<f32>,

    // input gains
    alphas: Vec<f32>,
    betas: Vec<f32>,

    // first-stage one-pole coefficient c = (1-alpha) e^{+i w dt}
    c_re: Vec<f32>,
    c_im: Vec<f32>,
    // second-stage (output smoothing) one-pole coefficient d = (1-beta) e^{+i w dt}
    d_re: Vec<f32>,
    d_im: Vec<f32>,

    // first-stage state (de-rotated frame)
    u_re: Vec<f32>,
    u_im: Vec<f32>,
    // second-stage state (de-rotated frame)
    v_re: Vec<f32>,
    v_im: Vec<f32>,

    // sample index of the most recently processed sample, needed only to
    // de-rotate at readout. Not used in the hot loop.
    sample_count: u64,
    sample_rate: f32,
}

#[allow(clippy::len_without_is_empty)]
impl OnePoleBank {
    pub fn from_frequencies(freqs: &[f32], sample_rate: f32) -> Self {
        let alphas = heuristic_alphas(freqs, sample_rate);
        let configs: Vec<ResonatorConfig> = freqs
            .iter()
            .zip(&alphas)
            .map(|(&f, &a)| ResonatorConfig::new(f, a, a))
            .collect();
        Self::new(&configs, sample_rate)
    }

    pub fn new(configs: &[ResonatorConfig], sample_rate: f32) -> Self {
        let n_resonators = configs.len();
        let mut frequencies = Vec::with_capacity(n_resonators);
        let mut alphas = Vec::with_capacity(n_resonators);
        let mut betas = Vec::with_capacity(n_resonators);
        let mut c_re = Vec::with_capacity(n_resonators);
        let mut c_im = Vec::with_capacity(n_resonators);
        let mut d_re = Vec::with_capacity(n_resonators);
        let mut d_im = Vec::with_capacity(n_resonators);

        for &ResonatorConfig { freq, alpha, beta } in configs {
            // +ω·Δt: the de-rotated frame rotates opposite to the reference phasor.
            let w = 2.0 * PI * freq / sample_rate;
            let (ws, wc) = w.sin_cos();
            frequencies.push(freq);
            alphas.push(alpha);
            betas.push(beta);
            c_re.push((1.0 - alpha) * wc);
            c_im.push((1.0 - alpha) * ws);
            d_re.push((1.0 - beta) * wc);
            d_im.push((1.0 - beta) * ws);
        }

        Self {
            n_resonators,
            frequencies,
            alphas,
            betas,
            c_re,
            c_im,
            d_re,
            d_im,
            u_re: vec![0.0; n_resonators],
            u_im: vec![0.0; n_resonators],
            v_re: vec![0.0; n_resonators],
            v_im: vec![0.0; n_resonators],
            sample_count: 0,
            sample_rate,
        }
    }

    #[inline]
    pub fn process_sample(&mut self, sample: f32) {
        self.process_sample_inner(sample);
        self.sample_count += 1;
    }

    #[inline]
    pub fn process_samples(&mut self, samples: &[f32]) {
        for &s in samples {
            self.process_sample_inner(s);
        }
        self.sample_count += samples.len() as u64;
    }

    #[inline(always)]
    fn process_sample_inner(&mut self, sample: f32) {
        let n = self.n_resonators;
        let alphas = &self.alphas[..n];
        let betas = &self.betas[..n];
        let c_re = &self.c_re[..n];
        let c_im = &self.c_im[..n];
        let d_re = &self.d_re[..n];
        let d_im = &self.d_im[..n];
        let u_re = &mut self.u_re[..n];
        let u_im = &mut self.u_im[..n];
        let v_re = &mut self.v_re[..n];
        let v_im = &mut self.v_im[..n];

        for k in 0..n {
            // first one-pole: u = c*u + alpha*x   (alpha*x is real)
            let ur = u_re[k];
            let ui = u_im[k];
            let nur = mul_add(c_re[k], ur, mul_add(-c_im[k], ui, alphas[k] * sample));
            let nui = mul_add(c_re[k], ui, c_im[k] * ur);
            u_re[k] = nur;
            u_im[k] = nui;

            // second one-pole (output smoothing): v = d*v + beta*u
            let vr = v_re[k];
            let vi = v_im[k];
            v_re[k] = mul_add(d_re[k], vr, mul_add(-d_im[k], vi, betas[k] * nur));
            v_im[k] = mul_add(d_re[k], vi, mul_add(d_im[k], vr, betas[k] * nui));
        }
    }

    pub fn resonate(&mut self, signal: &[f32], hop: usize) -> Vec<Complex32> {
        let n_frames = signal.len() / hop;
        let mut out = Vec::with_capacity(n_frames * self.n_resonators);
        for chunk in signal.chunks_exact(hop) {
            self.process_samples(chunk);
            for i in 0..self.n_resonators {
                out.push(self.complex(i));
            }
        }
        out
    }

    pub fn reset(&mut self) {
        self.u_re.fill(0.0);
        self.u_im.fill(0.0);
        self.v_re.fill(0.0);
        self.v_im.fill(0.0);
        self.sample_count = 0;
    }

    pub fn len(&self) -> usize {
        self.n_resonators
    }

    pub fn freq(&self, i: usize) -> f32 {
        self.frequencies[i]
    }

    /// Power is frame-invariant: `|v_n|^2 == |rr_n|^2`. No de-rotation needed.
    pub fn power(&self, i: usize) -> f32 {
        self.v_re[i] * self.v_re[i] + self.v_im[i] * self.v_im[i]
    }

    pub fn magnitude(&self, i: usize) -> f32 {
        self.power(i).sqrt()
    }

    /// Absolute-frame complex value, recovered by de-rotating the stored state
    /// by `e^{-iωnΔt}` where `n` is the index of the last processed sample.
    /// This is the only place trig is evaluated, and only on readout.
    ///
    /// The de-rotation angle `2π·f·n/sr` grows without bound as `n` accumulates,
    /// so it is formed and reduced in `f64`. At `f32` the `sin`/`cos` argument
    /// degrades within minutes (ulp ≈ 0.2 rad after ~10 min at 440 Hz) and is
    /// meaningless after about an hour — long-running phase/`complex` readouts
    /// would drift into noise. `f64` keeps it accurate for many hours. The power
    /// path ([`power`](Self::power)) is frame-invariant and never touches this.
    pub fn complex(&self, i: usize) -> Complex32 {
        if self.sample_count == 0 {
            return Complex32::new(self.v_re[i], self.v_im[i]);
        }
        let n = (self.sample_count - 1) as f64;
        let theta =
            std::f64::consts::TAU * self.frequencies[i] as f64 * n / self.sample_rate as f64;
        // Reduce before the cast so the f32 trig argument stays in [0, 2π).
        let theta = theta.rem_euclid(std::f64::consts::TAU) as f32;
        let (s, cth) = theta.sin_cos();
        // (v_re + i v_im) * (cos - i sin)
        let re = self.v_re[i] * cth + self.v_im[i] * s;
        let im = self.v_im[i] * cth - self.v_re[i] * s;
        Complex32::new(re, im)
    }

    /// Current phase at bin `i`, in radians. Requires de-rotation, so it is
    /// computed from [`complex`](Self::complex).
    pub fn phase(&self, i: usize) -> f32 {
        let c = self.complex(i);
        c.im.atan2(c.re)
    }

    /// A copy of every resonator's resonant frequency, in Hz.
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
        let mut bank = OnePoleBank::new(&[ResonatorConfig::new(freq, alpha, alpha)], sr);
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
        let mut bank = OnePoleBank::new(&configs, sr);
        let signal: Vec<f32> = (0..sr as usize)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).cos())
            .collect();
        bank.process_samples(&signal);
        let p = bank.powers();
        assert!(p[1] > p[0] * 10.0, "440 should dominate 220: {p:?}");
        assert!(p[1] > p[2] * 10.0, "440 should dominate 880: {p:?}");
    }

    #[test]
    fn power_matches_reference_bank() {
        // The whole point: identical power to the phasor formulation, up to f32.
        use crate::ResonatorBank;
        let sr = 44100.0;
        let configs: Vec<_> = [110.0, 261.6, 440.0, 1000.0, 4186.0]
            .iter()
            .map(|&f| {
                let a = heuristic_alpha(f, sr);
                ResonatorConfig::new(f, a, a)
            })
            .collect();
        let signal: Vec<f32> = (0..sr as usize)
            .map(|i| {
                let t = i as f32 / sr;
                (2.0 * PI * 440.0 * t).sin() + 0.5 * (2.0 * PI * 110.0 * t).sin()
            })
            .collect();

        let mut reference = ResonatorBank::new(&configs, sr);
        let mut onepole = OnePoleBank::new(&configs, sr);
        reference.process_samples(&signal);
        onepole.process_samples(&signal);

        for i in 0..configs.len() {
            let pr = reference.power(i);
            let po = onepole.power(i);
            let rel = (pr - po).abs() / pr.max(1e-9);
            assert!(rel < 1e-3, "bin {i}: ref={pr} onepole={po} rel={rel:e}");
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut bank = OnePoleBank::new(&[ResonatorConfig::new(440.0, 0.01, 0.01)], 44100.0);
        bank.process_samples(&vec![0.5; 1000]);
        assert!(bank.magnitude(0) > 0.0);
        bank.reset();
        assert_eq!(bank.power(0), 0.0);
    }

    /// Phase readout must stay accurate after a long run, when the de-rotation
    /// index `n` is large. Detuning recovered from two phase samples 128 apart
    /// should still match the true offset — this fails outright if the
    /// de-rotation is done in f32 (the trig argument is ~2.5e6 rad here).
    #[test]
    fn phase_readout_stable_at_large_sample_count() {
        use std::f32::consts::TAU;
        let sr = 44100.0;
        let f_bin = 440.0;
        let f_in = 441.0; // +1 Hz
        let a = heuristic_alpha(f_bin, sr);
        let mut bank = OnePoleBank::new(&[ResonatorConfig::new(f_bin, a, a)], sr);

        // Generate the input from a wrapped phase accumulator so the *stimulus*
        // stays clean at large n (a naive sin(2π f n/sr) in f32 would itself rot).
        let dph = std::f64::consts::TAU * f_in / sr as f64;
        let mut ph = 0.0f64;
        let mut next = || {
            let s = ph.sin() as f32;
            ph += dph;
            if ph >= std::f64::consts::TAU {
                ph -= std::f64::consts::TAU;
            }
            s
        };

        // ~90 s of audio → n ≈ 4 million.
        let warm: Vec<f32> = (0..(90 * sr as usize)).map(|_| next()).collect();
        bank.process_samples(&warm);

        let p0 = bank.phase(0);
        let gap = 128usize;
        let probe: Vec<f32> = (0..gap).map(|_| next()).collect();
        bank.process_samples(&probe);
        let p1 = bank.phase(0);

        let dphi = (p1 - p0 + PI).rem_euclid(TAU) - PI;
        let detuning = dphi / (TAU * gap as f32 / sr);
        assert!(
            (detuning - 1.0).abs() < 0.1,
            "detuning {detuning} should be ~1.0 Hz even at large n"
        );
    }
}

/// Property-based parity check against the reference [`ResonatorBank`].
///
/// `power_matches_reference_bank` above pins parity at one hand-picked operating
/// point (`alpha == beta`, five fixed frequencies, one signal). This module
/// *fuzzes* the operating point the consuming app actually uses: independent
/// `alpha_scale` / `beta_scale` (so `alpha != beta`), the full MIDI 12..=84
/// range — including the high-Q low end where `heuristic_alpha` is tiny — and an
/// arbitrary multi-partial stimulus.
///
/// The identity `v_n = rr_n · e^{+iωnΔt}` (see the module header) makes power
/// algebraically *equal* for any `(alpha, beta, freq)`; only f32 round-off can
/// separate the two formulations. So a wide rel/abs envelope that still trips on
/// a real divergence is the right shape: if this stays green, swapping
/// `ResonatorBank` → `OnePoleBank` cannot change a magnitude/power display.
#[cfg(test)]
mod parity_proptest {
    use proptest::prelude::*;

    use super::OnePoleBank;
    use crate::{
        ResonatorBank,
        ResonatorConfig,
        heuristic_alpha,
        midi_to_hz,
    };

    /// Configs mirroring the consumer's `build_resonator_bank`: per-bin
    /// `heuristic_alpha` scaled and clamped to the valid `(0, 1]` window.
    ///
    /// One bin per semitone (not the app's 5) — parity is per-bin independent of
    /// neighbours, so the coarser grid covers the same `(freq, alpha, beta)`
    /// regimes at a fraction of the cost.
    fn app_configs(sr: f32, alpha_scale: f32, beta_scale: f32) -> Vec<ResonatorConfig> {
        (12u32..=84)
            .map(|midi| {
                let freq = midi_to_hz(midi as f32, 440.0);
                let h = heuristic_alpha(freq, sr);
                let alpha = (h * alpha_scale).clamp(0.0001, 1.0);
                let beta = (h * beta_scale).clamp(0.0001, 1.0);
                ResonatorConfig::new(freq, alpha, beta)
            })
            .collect()
    }

    /// Sum-of-sines stimulus. Parameterised by a handful of `(freq, amp)` pairs
    /// rather than a raw sample vector, so proptest shrinks toward a *simple
    /// tone* on failure instead of an unreadable noise buffer.
    fn synth(partials: &[(f32, f32)], n: usize, sr: f32) -> Vec<f32> {
        use std::f32::consts::TAU;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr;
                partials.iter().map(|&(f, a)| a * (TAU * f * t).sin()).sum()
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]
        #[test]
        fn onepole_power_matches_reference_bank(
            alpha_scale in 0.1f32..4.0,
            beta_scale in 0.1f32..4.0,
            partials in prop::collection::vec((30.0f32..6000.0, 0.05f32..1.0), 1..4),
        ) {
            let sr = 44_100.0;
            let configs = app_configs(sr, alpha_scale, beta_scale);
            // 0.5 s — long enough to leave the initial transient at audible Qs.
            let signal = synth(&partials, sr as usize / 2, sr);

            let mut reference = ResonatorBank::new(&configs, sr);
            let mut onepole = OnePoleBank::new(&configs, sr);
            reference.process_samples(&signal);
            onepole.process_samples(&signal);

            let pr = reference.powers();
            let po = onepole.powers();
            // Quiet bins carry only f32 noise, where *relative* error is unbounded
            // and meaningless; floor the tolerance to a fraction of the loudest
            // bin so we judge the bins that actually paint pixels.
            let peak = pr.iter().chain(&po).cloned().fold(0.0f32, f32::max).max(1e-12);

            for i in 0..configs.len() {
                let diff = (pr[i] - po[i]).abs();
                // 0.5 % relative on the bin itself + 0.1 % of the global peak.
                // A genuine formulation bug is whole-percent or worse; f32
                // round-off lives far below this.
                let tol = 5e-3 * pr[i].max(po[i]) + 1e-3 * peak;
                prop_assert!(
                    diff <= tol,
                    "bin {i} f={:.1}Hz a={:.4} b={:.4}: ref={} onepole={} diff={:e} tol={:e}",
                    configs[i].freq, configs[i].alpha, configs[i].beta, pr[i], po[i], diff, tol
                );
            }
        }
    }
}
