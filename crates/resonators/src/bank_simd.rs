use wide::f32x4;

use crate::config::ResonatorConfig;
use crate::dynamics;

/// A bank of N resonators using explicit SIMD via `wide`.
///
/// Uses f32x4 (native 128-bit SIMD on ARM NEON, SSE on x86, WASM SIMD128).
/// Arrays are padded to a multiple of 4 internally.
/// Output is stored as flat Vec<f32> to avoid extraction overhead.
pub struct SimdResonatorBank {
    sample_rate: f32,
    num: usize,
    chunks: usize,

    frequencies: Vec<f32>,

    // EWMA coefficients (packed as f32x4)
    alphas: Vec<f32x4>,
    om_alphas: Vec<f32x4>,
    betas: Vec<f32x4>,
    om_betas: Vec<f32x4>,

    // Precomputed phasor multipliers
    w_re: Vec<f32x4>,
    w_im: Vec<f32x4>,

    // Phasor state
    z_re: Vec<f32x4>,
    z_im: Vec<f32x4>,

    // Unsmoothed accumulation
    r_re: Vec<f32x4>,
    r_im: Vec<f32x4>,

    // Smoothed output — stored as SIMD for compute, flat for reads
    rr_re: Vec<f32x4>,
    rr_im: Vec<f32x4>,

    // Flat output mirrors (written after each frame for zero-cost reads)
    out_re: Vec<f32>,
    out_im: Vec<f32>,
}

/// Pack a slice of f32 into Vec<f32x4>, padding with `pad_value`.
fn pack(values: &[f32], pad_value: f32) -> Vec<f32x4> {
    let chunks = (values.len() + 3) / 4;
    let mut result = Vec::with_capacity(chunks);
    for c in 0..chunks {
        let start = c * 4;
        let mut arr = [pad_value; 4];
        for i in 0..4 {
            if start + i < values.len() {
                arr[i] = values[start + i];
            }
        }
        result.push(f32x4::from(arr));
    }
    result
}

impl SimdResonatorBank {
    pub fn new(configs: &[ResonatorConfig], sample_rate: f32) -> Self {
        let num = configs.len();
        let chunks = (num + 3) / 4;

        let mut frequencies = Vec::with_capacity(num);
        let mut alphas_raw = Vec::with_capacity(num);
        let mut om_alphas_raw = Vec::with_capacity(num);
        let mut betas_raw = Vec::with_capacity(num);
        let mut om_betas_raw = Vec::with_capacity(num);
        let mut w_re_raw = Vec::with_capacity(num);
        let mut w_im_raw = Vec::with_capacity(num);

        for c in configs {
            frequencies.push(c.frequency);
            alphas_raw.push(c.alpha);
            om_alphas_raw.push(1.0 - c.alpha);
            betas_raw.push(c.beta);
            om_betas_raw.push(1.0 - c.beta);

            let angle = dynamics::phasor_angle(c.frequency, sample_rate);
            w_re_raw.push(angle.cos());
            w_im_raw.push(angle.sin());
        }

        Self {
            sample_rate,
            num,
            chunks,
            frequencies,
            alphas: pack(&alphas_raw, 0.0),
            om_alphas: pack(&om_alphas_raw, 1.0),
            betas: pack(&betas_raw, 0.0),
            om_betas: pack(&om_betas_raw, 1.0),
            w_re: pack(&w_re_raw, 1.0),
            w_im: pack(&w_im_raw, 0.0),
            z_re: vec![f32x4::splat(1.0); chunks],
            z_im: vec![f32x4::splat(0.0); chunks],
            r_re: vec![f32x4::splat(0.0); chunks],
            r_im: vec![f32x4::splat(0.0); chunks],
            rr_re: vec![f32x4::splat(0.0); chunks],
            rr_im: vec![f32x4::splat(0.0); chunks],
            out_re: vec![0.0; num],
            out_im: vec![0.0; num],
        }
    }

    /// Process a single sample across all resonators (4 at a time).
    #[inline]
    pub fn update_sample(&mut self, sample: f32) {
        let x = f32x4::splat(sample);

        for c in 0..self.chunks {
            let ax = self.alphas[c] * x;

            // EWMA accumulation: R = (1-α)*R + α*x*Z
            self.r_re[c] = self.om_alphas[c] * self.r_re[c] + ax * self.z_re[c];
            self.r_im[c] = self.om_alphas[c] * self.r_im[c] + ax * self.z_im[c];

            // Output smoothing: R̃ = (1-β)*R̃ + β*R
            self.rr_re[c] = self.om_betas[c] * self.rr_re[c] + self.betas[c] * self.r_re[c];
            self.rr_im[c] = self.om_betas[c] * self.rr_im[c] + self.betas[c] * self.r_im[c];

            // Phasor rotation: Z *= W (complex multiply)
            let zr = self.z_re[c];
            let zi = self.z_im[c];
            self.z_re[c] = zr * self.w_re[c] - zi * self.w_im[c];
            self.z_im[c] = zr * self.w_im[c] + zi * self.w_re[c];
        }
    }

    /// Process a frame of samples, stabilizing phasors and writing flat output.
    pub fn update_frame(&mut self, frame: &[f32]) {
        for &sample in frame {
            self.update_sample(sample);
        }
        self.stabilize();
        self.write_output();
    }

    /// Normalize phasors back to unit magnitude.
    fn stabilize(&mut self) {
        for c in 0..self.chunks {
            let mag_sq = self.z_re[c] * self.z_re[c] + self.z_im[c] * self.z_im[c];
            let inv_mag = f32x4::splat(1.0) / mag_sq.sqrt();
            self.z_re[c] *= inv_mag;
            self.z_im[c] *= inv_mag;
        }
    }

    /// Write SIMD output to flat arrays.
    #[inline]
    fn write_output(&mut self) {
        let re_ptr = self.out_re.as_mut_ptr();
        let im_ptr = self.out_im.as_mut_ptr();
        for c in 0..self.chunks {
            let re: [f32; 4] = self.rr_re[c].into();
            let im: [f32; 4] = self.rr_im[c].into();
            let offset = c * 4;
            let remaining = (self.num - offset).min(4);
            unsafe {
                std::ptr::copy_nonoverlapping(re.as_ptr(), re_ptr.add(offset), remaining);
                std::ptr::copy_nonoverlapping(im.as_ptr(), im_ptr.add(offset), remaining);
            }
        }
    }

    /// Borrow the smoothed output as split-complex slices (zero-copy).
    #[inline]
    pub fn complex(&self) -> (&[f32], &[f32]) {
        (&self.out_re, &self.out_im)
    }

    /// Power at a single bin.
    #[inline]
    pub fn power(&self, i: usize) -> f32 {
        self.out_re[i] * self.out_re[i] + self.out_im[i] * self.out_im[i]
    }

    /// Complex value at a single bin.
    #[inline]
    pub fn complex_at(&self, i: usize) -> (f32, f32) {
        (self.out_re[i], self.out_im[i])
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
    use crate::bank::ResonatorBank;
    use std::f32::consts::TAU;

    /// SIMD bank must match scalar bank exactly.
    #[test]
    fn test_simd_matches_scalar_bank() {
        let sr = 44100.0;
        let configs: Vec<ResonatorConfig> = crate::frequencies::midi_piano(440.0)
            .iter()
            .map(|&f| {
                let alpha = crate::dynamics::alpha_heuristic(f, sr);
                ResonatorConfig::with_alpha(f, alpha)
            })
            .collect();

        let mut scalar = ResonatorBank::new(&configs, sr);
        let mut simd = SimdResonatorBank::new(&configs, sr);

        // Process several frames
        for frame_idx in 0..10 {
            let frame: Vec<f32> = (0..256)
                .map(|i| {
                    let t = (frame_idx * 256 + i) as f32;
                    (TAU * 440.0 * t / sr).cos()
                })
                .collect();

            scalar.update_frame(&frame);
            simd.update_frame(&frame);
        }

        // Compare every bin
        let (scalar_re, scalar_im) = scalar.complex();
        let (simd_re, simd_im) = simd.complex();

        for k in 0..configs.len() {
            assert!(
                (scalar_re[k] - simd_re[k]).abs() < 1e-6,
                "bin {} re: scalar={} simd={}",
                k,
                scalar_re[k],
                simd_re[k]
            );
            assert!(
                (scalar_im[k] - simd_im[k]).abs() < 1e-6,
                "bin {} im: scalar={} simd={}",
                k,
                scalar_im[k],
                simd_im[k]
            );
        }
    }

    /// Test with non-multiple-of-4 bin count.
    #[test]
    fn test_simd_odd_bin_count() {
        let sr = 44100.0;
        let freqs = crate::frequencies::log_spaced(100.0, 13, 12);
        let configs: Vec<ResonatorConfig> = freqs
            .iter()
            .map(|&f| {
                let alpha = crate::dynamics::alpha_heuristic(f, sr);
                ResonatorConfig::with_alpha(f, alpha)
            })
            .collect();

        let mut scalar = ResonatorBank::new(&configs, sr);
        let mut simd = SimdResonatorBank::new(&configs, sr);

        let frame: Vec<f32> = (0..256)
            .map(|i| (TAU * 440.0 * i as f32 / sr).cos())
            .collect();

        scalar.update_frame(&frame);
        simd.update_frame(&frame);

        let (scalar_re, scalar_im) = scalar.complex();
        let (simd_re, simd_im) = simd.complex();

        for k in 0..13 {
            assert!(
                (scalar_re[k] - simd_re[k]).abs() < 1e-6,
                "bin {} re mismatch",
                k
            );
            assert!(
                (scalar_im[k] - simd_im[k]).abs() < 1e-6,
                "bin {} im mismatch",
                k
            );
        }
    }
}
