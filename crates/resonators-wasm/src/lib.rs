use resonators::{ResonatorBank, ResonatorConfig, SimdResonatorBank};
use wasm_bindgen::prelude::*;

/// A resonator bank for use from JavaScript.
///
/// Holds state between calls — create once, feed frames repeatedly.
#[wasm_bindgen]
pub struct WasmResonatorBank {
    bank: ResonatorBank,
    num: usize,
}

#[wasm_bindgen]
impl WasmResonatorBank {
    /// Create a new resonator bank.
    ///
    /// `frequencies`, `alphas`, and `betas` must be the same length.
    #[wasm_bindgen(constructor)]
    pub fn new(
        frequencies: &[f32],
        alphas: &[f32],
        betas: &[f32],
        sample_rate: f32,
    ) -> WasmResonatorBank {
        let configs: Vec<ResonatorConfig> = frequencies
            .iter()
            .zip(alphas.iter())
            .zip(betas.iter())
            .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
            .collect();
        let num = configs.len();
        WasmResonatorBank {
            bank: ResonatorBank::new(&configs, sample_rate),
            num,
        }
    }

    /// Process a frame of audio samples. Call this per hop.
    pub fn update_frame(&mut self, frame: &[f32]) {
        self.bank.update_frame(frame);
    }

    /// Get the current powers (magnitude squared) as a new Float32Array.
    pub fn powers(&self) -> Vec<f32> {
        let (re, im) = self.bank.complex();
        re.iter()
            .zip(im.iter())
            .map(|(r, i)| r * r + i * i)
            .collect()
    }

    /// Get the current amplitudes as a new Float32Array.
    pub fn amplitudes(&self) -> Vec<f32> {
        let (re, im) = self.bank.complex();
        re.iter()
            .zip(im.iter())
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Get smoothed output as split-complex: [re_0..re_N, im_0..im_N].
    pub fn complex(&self) -> Vec<f32> {
        let (re, im) = self.bank.complex();
        let mut out = Vec::with_capacity(self.num * 2);
        out.extend_from_slice(re);
        out.extend_from_slice(im);
        out
    }

    pub fn num_resonators(&self) -> usize {
        self.num
    }
}

/// A SIMD-optimized resonator bank for use from JavaScript.
///
/// Uses wide f32x4 SIMD. When compiled with RUSTFLAGS="-C target-feature=+simd128",
/// this uses WASM SIMD128 instructions.
#[wasm_bindgen]
pub struct WasmSimdResonatorBank {
    bank: SimdResonatorBank,
    num: usize,
}

#[wasm_bindgen]
impl WasmSimdResonatorBank {
    #[wasm_bindgen(constructor)]
    pub fn new(
        frequencies: &[f32],
        alphas: &[f32],
        betas: &[f32],
        sample_rate: f32,
    ) -> WasmSimdResonatorBank {
        let configs: Vec<ResonatorConfig> = frequencies
            .iter()
            .zip(alphas.iter())
            .zip(betas.iter())
            .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
            .collect();
        let num = configs.len();
        WasmSimdResonatorBank {
            bank: SimdResonatorBank::new(&configs, sample_rate),
            num,
        }
    }

    pub fn update_frame(&mut self, frame: &[f32]) {
        self.bank.update_frame(frame);
    }

    pub fn powers(&self) -> Vec<f32> {
        let (re, im) = self.bank.complex();
        re.iter()
            .zip(im.iter())
            .map(|(r, i)| r * r + i * i)
            .collect()
    }

    pub fn amplitudes(&self) -> Vec<f32> {
        let (re, im) = self.bank.complex();
        re.iter()
            .zip(im.iter())
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }

    pub fn complex(&self) -> Vec<f32> {
        let (re, im) = self.bank.complex();
        let mut out = Vec::with_capacity(self.num * 2);
        out.extend_from_slice(re);
        out.extend_from_slice(im);
        out
    }

    pub fn num_resonators(&self) -> usize {
        self.num
    }
}

/// Process a full signal through a resonator bank in one call.
///
/// Returns split-complex output: [re_0..re_N, im_0..im_N] per slice,
/// flattened into a single Float32Array.
#[wasm_bindgen]
pub fn resonate(
    input: &[f32],
    sample_rate: f32,
    frequencies: &[f32],
    alphas: &[f32],
    betas: &[f32],
    hop_length: usize,
) -> Vec<f32> {
    let configs: Vec<ResonatorConfig> = frequencies
        .iter()
        .zip(alphas.iter())
        .zip(betas.iter())
        .map(|((&f, &a), &b)| ResonatorConfig::new(f, a, b))
        .collect();

    let n_bins = configs.len();
    let n_slices = input.len() / hop_length;
    let two_n = 2 * n_bins;
    let mut bank = ResonatorBank::new(&configs, sample_rate);
    let mut result = vec![0.0f32; n_slices * two_n];

    for s in 0..n_slices {
        let frame = &input[s * hop_length..(s + 1) * hop_length];
        bank.update_frame(frame);
        let (re, im) = bank.complex();
        let offset = s * two_n;
        result[offset..offset + n_bins].copy_from_slice(re);
        result[offset + n_bins..offset + two_n].copy_from_slice(im);
    }

    result
}

/// Helper: compute alpha heuristic for an array of frequencies (Equation 7).
#[wasm_bindgen]
pub fn alpha_heuristic(frequencies: &[f32], sample_rate: f32) -> Vec<f32> {
    frequencies
        .iter()
        .map(|&f| resonators::dynamics::alpha_heuristic(f, sample_rate))
        .collect()
}

/// Helper: generate MIDI piano frequencies (88 keys).
#[wasm_bindgen]
pub fn midi_piano_frequencies(tuning: f32) -> Vec<f32> {
    resonators::frequencies::midi_piano(tuning)
}

/// Helper: generate log-spaced frequencies (CQT-style).
#[wasm_bindgen]
pub fn log_frequencies(fmin: f32, n_bins: usize, bins_per_octave: usize) -> Vec<f32> {
    resonators::frequencies::log_spaced(fmin, n_bins, bins_per_octave)
}
