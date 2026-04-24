use std::f32::consts::PI;

use num_complex::Complex32;

use crate::STABILIZE_EVERY;
use crate::config::ResonatorConfig;
use crate::dynamics::heuristic_alphas;

// WASM SIMD128 is the only target where explicit SIMD meaningfully beats
// LLVM auto-vectorisation of the scalar loop. On x86_64 (SSE baseline)
// and aarch64 (NEON baseline) auto-vectorisation gets the same or better
// throughput; on wasm32 without `+simd128` there's no SIMD to emit. See
// `process_sample_inner` below for the two implementations.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use wide::f32x4;

// x86_64 explicit SIMD (AVX2 + FMA, AVX-512F) is exposed behind
// `#[target_feature]`-gated unsafe methods for benchmarking. Each method
// is a parallel implementation of `process_sample_inner` at a different
// vector width so that the `bank` bench can compare all three on a
// single binary. Auto-vectorisation-of-scalar may match these in
// practice; see bench comments.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Unaligned 128-bit load of four contiguous `f32`s starting at `buf[offset]`.
///
/// # Safety
/// Caller must ensure `offset + 4 <= buf.len()`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn load_f32x4(buf: &[f32], offset: usize) -> f32x4 {
    unsafe { core::ptr::read_unaligned(buf.as_ptr().add(offset) as *const f32x4) }
}

/// Unaligned 128-bit store of four contiguous `f32`s starting at `buf[offset]`.
///
/// # Safety
/// Caller must ensure `offset + 4 <= buf.len()`.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
unsafe fn store_f32x4(buf: &mut [f32], offset: usize, value: f32x4) {
    unsafe {
        core::ptr::write_unaligned(buf.as_mut_ptr().add(offset) as *mut f32x4, value);
    }
}

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
    ///
    /// Amortises the per-sample stabilisation check by batching up to
    /// `STABILIZE_EVERY` samples between stabilisations instead of running
    /// the modulo test inside the hot loop.
    #[inline]
    pub fn process_samples(&mut self, samples: &[f32]) {
        let mut i = 0;
        while i < samples.len() {
            let until_stabilize = STABILIZE_EVERY - (self.sample_count % STABILIZE_EVERY);
            let take = (samples.len() - i).min(until_stabilize as usize);
            for &s in &samples[i..i + take] {
                self.process_sample_inner(s);
            }
            self.sample_count += take as u64;
            if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
                self.stabilize();
            }
            i += take;
        }
    }

    /// Scalar per-sample update across all bins. LLVM auto-vectorises this
    /// well enough on x86_64 (SSE) and aarch64 (NEON) that explicit SIMD
    /// matches or slightly regresses it; this is also the fallback on
    /// wasm32 builds without `+simd128`.
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    #[inline(always)]
    fn process_sample_inner(&mut self, sample: f32) {
        for k in 0..self.n_resonators {
            let alpha = self.alphas[k];
            let beta = self.betas[k];
            let alpha_sample = alpha * sample;

            // EWMA accumulation
            self.r_re[k] = (1.0 - alpha).mul_add(self.r_re[k], alpha_sample * self.z_re[k]);
            self.r_im[k] = (1.0 - alpha).mul_add(self.r_im[k], alpha_sample * self.z_im[k]);

            // Output smoothing
            self.rr_re[k] = (1.0 - beta).mul_add(self.rr_re[k], beta * self.r_re[k]);
            self.rr_im[k] = (1.0 - beta).mul_add(self.rr_im[k], beta * self.r_im[k]);

            // Rotate phasor
            let zr = self.z_re[k];
            let zi = self.z_im[k];
            self.z_re[k] = zr * self.w_re[k] - zi * self.w_im[k];
            self.z_im[k] = zr * self.w_im[k] + zi * self.w_re[k];
        }
    }

    /// WASM-SIMD128 per-sample update: processes 4 bins per iteration via
    /// `wide::f32x4`, mapping to `v128.*` instructions. Loads / stores go
    /// through `ptr::read_unaligned` / `write_unaligned` so they lower to
    /// single `v128.load` / `v128.store` ops — the array-literal
    /// `f32x4::new([a,b,c,d])` path generates per-lane inserts and
    /// defeats vectorisation.
    ///
    /// Measured ~6-8x speedup over the scalar path in-browser (Firefox
    /// 130, Chrome 131) at 88–880 bins; see PR description for numbers.
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[inline(always)]
    fn process_sample_inner(&mut self, sample: f32) {
        let n = self.n_resonators;
        let vec_end = n & !3; // round down to multiple of 4

        let sample_v = f32x4::splat(sample);
        let one_v = f32x4::splat(1.0);

        // Safety: all 10 backing `Vec<f32>` buffers share `n_resonators`
        // elements; `vec_end` is `n & !3`, so every `k..k+4` subslice is
        // in-bounds for every buffer. `f32x4` is `#[repr(C, align(16))]`
        // with exactly 16 bytes — four contiguous f32s.
        let mut k = 0;
        unsafe {
            while k < vec_end {
                let alpha = load_f32x4(&self.alphas, k);
                let beta = load_f32x4(&self.betas, k);
                let z_re = load_f32x4(&self.z_re, k);
                let z_im = load_f32x4(&self.z_im, k);
                let r_re_old = load_f32x4(&self.r_re, k);
                let r_im_old = load_f32x4(&self.r_im, k);
                let rr_re_old = load_f32x4(&self.rr_re, k);
                let rr_im_old = load_f32x4(&self.rr_im, k);
                let w_re = load_f32x4(&self.w_re, k);
                let w_im = load_f32x4(&self.w_im, k);

                let one_m_alpha = one_v - alpha;
                let one_m_beta = one_v - beta;
                let alpha_sample = alpha * sample_v;

                // EWMA accumulation: r = (1 - alpha) * r_prev + alpha_sample * z
                let r_re = one_m_alpha.mul_add(r_re_old, alpha_sample * z_re);
                let r_im = one_m_alpha.mul_add(r_im_old, alpha_sample * z_im);

                // Output smoothing: rr = (1 - beta) * rr_prev + beta * r
                let rr_re = one_m_beta.mul_add(rr_re_old, beta * r_re);
                let rr_im = one_m_beta.mul_add(rr_im_old, beta * r_im);

                // Phasor rotation: z_new = z * w (complex multiply)
                let z_re_new = z_re * w_re - z_im * w_im;
                let z_im_new = z_re * w_im + z_im * w_re;

                store_f32x4(&mut self.r_re, k, r_re);
                store_f32x4(&mut self.r_im, k, r_im);
                store_f32x4(&mut self.rr_re, k, rr_re);
                store_f32x4(&mut self.rr_im, k, rr_im);
                store_f32x4(&mut self.z_re, k, z_re_new);
                store_f32x4(&mut self.z_im, k, z_im_new);

                k += 4;
            }
        }

        // Scalar tail for any remaining bins (n not a multiple of 4).
        while k < n {
            let alpha = self.alphas[k];
            let beta = self.betas[k];
            let alpha_sample = alpha * sample;

            self.r_re[k] = (1.0 - alpha).mul_add(self.r_re[k], alpha_sample * self.z_re[k]);
            self.r_im[k] = (1.0 - alpha).mul_add(self.r_im[k], alpha_sample * self.z_im[k]);

            self.rr_re[k] = (1.0 - beta).mul_add(self.rr_re[k], beta * self.r_re[k]);
            self.rr_im[k] = (1.0 - beta).mul_add(self.rr_im[k], beta * self.r_im[k]);

            let zr = self.z_re[k];
            let zi = self.z_im[k];
            self.z_re[k] = zr * self.w_re[k] - zi * self.w_im[k];
            self.z_im[k] = zr * self.w_im[k] + zi * self.w_re[k];

            k += 1;
        }
    }

    /// x86_64 AVX2+FMA per-sample update: processes 8 bins per iteration
    /// via 256-bit `__m256` vectors with `vfmadd231ps` for the EWMA /
    /// smoothing mul-adds. Exposed for benchmarking only (see `benches/
    /// bank.rs`). Unaligned loads because `Vec<f32>` is only 4-byte
    /// aligned; on modern x86_64 unaligned loads have matched aligned
    /// perf when they don't cross a cache line.
    ///
    /// # Safety
    /// CPU must support `avx2` and `fma` — check with
    /// `is_x86_feature_detected!("avx2")` and `..("fma")` before calling.
    #[cfg(target_arch = "x86_64")]
    #[doc(hidden)]
    #[inline]
    pub unsafe fn process_sample_avx2(&mut self, sample: f32) {
        unsafe { self.process_sample_inner_avx2(sample) };
        self.sample_count += 1;
        if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
            self.stabilize();
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn process_sample_inner_avx2(&mut self, sample: f32) {
        unsafe {
            let n = self.n_resonators;
            let vec_end = n & !7; // multiple of 8

            let sample_v = _mm256_set1_ps(sample);
            let one_v = _mm256_set1_ps(1.0);

            // Safety: all 10 backing buffers share `n_resonators`; every
            // `k..k+8` slice is in-bounds for every buffer.
            let mut k = 0;
            while k < vec_end {
                let alpha = _mm256_loadu_ps(self.alphas.as_ptr().add(k));
                let beta = _mm256_loadu_ps(self.betas.as_ptr().add(k));
                let z_re = _mm256_loadu_ps(self.z_re.as_ptr().add(k));
                let z_im = _mm256_loadu_ps(self.z_im.as_ptr().add(k));
                let r_re_old = _mm256_loadu_ps(self.r_re.as_ptr().add(k));
                let r_im_old = _mm256_loadu_ps(self.r_im.as_ptr().add(k));
                let rr_re_old = _mm256_loadu_ps(self.rr_re.as_ptr().add(k));
                let rr_im_old = _mm256_loadu_ps(self.rr_im.as_ptr().add(k));
                let w_re = _mm256_loadu_ps(self.w_re.as_ptr().add(k));
                let w_im = _mm256_loadu_ps(self.w_im.as_ptr().add(k));

                let one_m_alpha = _mm256_sub_ps(one_v, alpha);
                let one_m_beta = _mm256_sub_ps(one_v, beta);
                let alpha_sample = _mm256_mul_ps(alpha, sample_v);

                // r = (1 - alpha) * r_prev + alpha_sample * z
                let r_re = _mm256_fmadd_ps(one_m_alpha, r_re_old, _mm256_mul_ps(alpha_sample, z_re));
                let r_im = _mm256_fmadd_ps(one_m_alpha, r_im_old, _mm256_mul_ps(alpha_sample, z_im));

                // rr = (1 - beta) * rr_prev + beta * r
                let rr_re = _mm256_fmadd_ps(one_m_beta, rr_re_old, _mm256_mul_ps(beta, r_re));
                let rr_im = _mm256_fmadd_ps(one_m_beta, rr_im_old, _mm256_mul_ps(beta, r_im));

                // z_new = z * w  (complex multiply)
                let z_re_new = _mm256_sub_ps(
                    _mm256_mul_ps(z_re, w_re),
                    _mm256_mul_ps(z_im, w_im),
                );
                let z_im_new = _mm256_add_ps(
                    _mm256_mul_ps(z_re, w_im),
                    _mm256_mul_ps(z_im, w_re),
                );

                _mm256_storeu_ps(self.r_re.as_mut_ptr().add(k), r_re);
                _mm256_storeu_ps(self.r_im.as_mut_ptr().add(k), r_im);
                _mm256_storeu_ps(self.rr_re.as_mut_ptr().add(k), rr_re);
                _mm256_storeu_ps(self.rr_im.as_mut_ptr().add(k), rr_im);
                _mm256_storeu_ps(self.z_re.as_mut_ptr().add(k), z_re_new);
                _mm256_storeu_ps(self.z_im.as_mut_ptr().add(k), z_im_new);

                k += 8;
            }

            // Scalar tail for any remaining bins (n not a multiple of 8).
            while k < n {
                let alpha = self.alphas[k];
                let beta = self.betas[k];
                let alpha_sample = alpha * sample;

                self.r_re[k] = (1.0 - alpha).mul_add(self.r_re[k], alpha_sample * self.z_re[k]);
                self.r_im[k] = (1.0 - alpha).mul_add(self.r_im[k], alpha_sample * self.z_im[k]);

                self.rr_re[k] = (1.0 - beta).mul_add(self.rr_re[k], beta * self.r_re[k]);
                self.rr_im[k] = (1.0 - beta).mul_add(self.rr_im[k], beta * self.r_im[k]);

                let zr = self.z_re[k];
                let zi = self.z_im[k];
                self.z_re[k] = zr * self.w_re[k] - zi * self.w_im[k];
                self.z_im[k] = zr * self.w_im[k] + zi * self.w_re[k];

                k += 1;
            }
        }
    }

    /// x86_64 AVX-512F per-sample update: processes 16 bins per iteration
    /// via 512-bit `__m512` vectors. Same algorithm as
    /// `process_sample_avx2` at twice the vector width. Exposed for
    /// benchmarking only.
    ///
    /// # Safety
    /// CPU must support `avx512f` — check with
    /// `is_x86_feature_detected!("avx512f")` before calling.
    #[cfg(target_arch = "x86_64")]
    #[doc(hidden)]
    #[inline]
    pub unsafe fn process_sample_avx512(&mut self, sample: f32) {
        unsafe { self.process_sample_inner_avx512(sample) };
        self.sample_count += 1;
        if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
            self.stabilize();
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    unsafe fn process_sample_inner_avx512(&mut self, sample: f32) {
        unsafe {
            let n = self.n_resonators;
            let vec_end = n & !15; // multiple of 16

            let sample_v = _mm512_set1_ps(sample);
            let one_v = _mm512_set1_ps(1.0);

            let mut k = 0;
            while k < vec_end {
                let alpha = _mm512_loadu_ps(self.alphas.as_ptr().add(k));
                let beta = _mm512_loadu_ps(self.betas.as_ptr().add(k));
                let z_re = _mm512_loadu_ps(self.z_re.as_ptr().add(k));
                let z_im = _mm512_loadu_ps(self.z_im.as_ptr().add(k));
                let r_re_old = _mm512_loadu_ps(self.r_re.as_ptr().add(k));
                let r_im_old = _mm512_loadu_ps(self.r_im.as_ptr().add(k));
                let rr_re_old = _mm512_loadu_ps(self.rr_re.as_ptr().add(k));
                let rr_im_old = _mm512_loadu_ps(self.rr_im.as_ptr().add(k));
                let w_re = _mm512_loadu_ps(self.w_re.as_ptr().add(k));
                let w_im = _mm512_loadu_ps(self.w_im.as_ptr().add(k));

                let one_m_alpha = _mm512_sub_ps(one_v, alpha);
                let one_m_beta = _mm512_sub_ps(one_v, beta);
                let alpha_sample = _mm512_mul_ps(alpha, sample_v);

                let r_re = _mm512_fmadd_ps(one_m_alpha, r_re_old, _mm512_mul_ps(alpha_sample, z_re));
                let r_im = _mm512_fmadd_ps(one_m_alpha, r_im_old, _mm512_mul_ps(alpha_sample, z_im));

                let rr_re = _mm512_fmadd_ps(one_m_beta, rr_re_old, _mm512_mul_ps(beta, r_re));
                let rr_im = _mm512_fmadd_ps(one_m_beta, rr_im_old, _mm512_mul_ps(beta, r_im));

                let z_re_new = _mm512_sub_ps(
                    _mm512_mul_ps(z_re, w_re),
                    _mm512_mul_ps(z_im, w_im),
                );
                let z_im_new = _mm512_add_ps(
                    _mm512_mul_ps(z_re, w_im),
                    _mm512_mul_ps(z_im, w_re),
                );

                _mm512_storeu_ps(self.r_re.as_mut_ptr().add(k), r_re);
                _mm512_storeu_ps(self.r_im.as_mut_ptr().add(k), r_im);
                _mm512_storeu_ps(self.rr_re.as_mut_ptr().add(k), rr_re);
                _mm512_storeu_ps(self.rr_im.as_mut_ptr().add(k), rr_im);
                _mm512_storeu_ps(self.z_re.as_mut_ptr().add(k), z_re_new);
                _mm512_storeu_ps(self.z_im.as_mut_ptr().add(k), z_im_new);

                k += 16;
            }

            // Scalar tail.
            while k < n {
                let alpha = self.alphas[k];
                let beta = self.betas[k];
                let alpha_sample = alpha * sample;

                self.r_re[k] = (1.0 - alpha).mul_add(self.r_re[k], alpha_sample * self.z_re[k]);
                self.r_im[k] = (1.0 - alpha).mul_add(self.r_im[k], alpha_sample * self.z_im[k]);

                self.rr_re[k] = (1.0 - beta).mul_add(self.rr_re[k], beta * self.r_re[k]);
                self.rr_im[k] = (1.0 - beta).mul_add(self.rr_im[k], beta * self.r_im[k]);

                let zr = self.z_re[k];
                let zi = self.z_im[k];
                self.z_re[k] = zr * self.w_re[k] - zi * self.w_im[k];
                self.z_im[k] = zr * self.w_im[k] + zi * self.w_re[k];

                k += 1;
            }
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

    #[cfg(target_arch = "x86_64")]
    fn matches_scalar(backend: &str, process: unsafe fn(&mut ResonatorBank, f32)) {
        let sr = 44100.0;
        // Use bin counts that span both the vectorised body and a scalar
        // tail for both AVX2 (tail if n % 8 != 0) and AVX-512 (tail if
        // n % 16 != 0). 23 exercises both tails.
        for n_bins in [1usize, 8, 15, 16, 17, 23, 64, 88] {
            let freqs: Vec<f32> = (0..n_bins).map(|i| 100.0 + i as f32 * 37.0).collect();
            let signal: Vec<f32> = (0..1024)
                .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).cos() * 0.5)
                .collect();

            let mut scalar = ResonatorBank::from_frequencies(&freqs, sr);
            let mut simd = ResonatorBank::from_frequencies(&freqs, sr);
            for &s in &signal {
                scalar.process_sample(s);
                unsafe { process(&mut simd, s) };
            }

            for k in 0..n_bins {
                let s_re = scalar.rr_re[k];
                let s_im = scalar.rr_im[k];
                let v_re = simd.rr_re[k];
                let v_im = simd.rr_im[k];
                let tol = 1e-4 * (1.0 + s_re.abs() + s_im.abs());
                assert!(
                    (s_re - v_re).abs() < tol && (s_im - v_im).abs() < tol,
                    "{backend} bin {k}/{n_bins}: scalar=({s_re},{s_im}) simd=({v_re},{v_im})",
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
            eprintln!("skipping — CPU lacks avx2 or fma");
            return;
        }
        matches_scalar("avx2", ResonatorBank::process_sample_avx2);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx512_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            eprintln!("skipping — CPU lacks avx512f");
            return;
        }
        matches_scalar("avx512", ResonatorBank::process_sample_avx512);
    }

    #[test]
    #[should_panic]
    fn resonate_panics_on_zero_hop() {
        let configs = vec![ResonatorConfig::new(440.0, 0.01, 0.01)];
        let mut bank = ResonatorBank::new(&configs, 44100.0);
        let _ = bank.resonate(&[0.0; 100], 0);
    }
}
