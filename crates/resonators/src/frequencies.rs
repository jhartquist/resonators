/// Convert MIDI note number to frequency in Hz.
///
/// `f = tuning * 2^((midi - 69) / 12)`
pub fn midi_to_hz(midi: f32, tuning: f32) -> f32 {
    tuning * 2.0f32.powf((midi - 69.0) / 12.0)
}

/// Convert frequency in Hz to MIDI note number.
///
/// `midi = 69 + 12 * log2(hz / tuning)`
pub fn hz_to_midi(hz: f32, tuning: f32) -> f32 {
    69.0 + 12.0 * (hz / tuning).log2()
}

/// Generate logarithmically-spaced frequencies (CQT-style).
///
/// `f_i = fmin * 2^(i / bins_per_octave)` for `i` in `0..n_bins`.
pub fn log_spaced(fmin: f32, n_bins: usize, bins_per_octave: usize) -> Vec<f32> {
    (0..n_bins)
        .map(|i| fmin * 2.0f32.powf(i as f32 / bins_per_octave as f32))
        .collect()
}

/// Generate MIDI piano frequencies (88 keys, A0 to C8).
///
/// MIDI notes 21..=108 at the given tuning reference.
pub fn midi_piano(tuning: f32) -> Vec<f32> {
    (21..=108)
        .map(|midi| midi_to_hz(midi as f32, tuning))
        .collect()
}

/// Generate linearly-spaced frequencies (STFT-style).
///
/// `n_bins` frequencies from `fmin` to `fmax`, inclusive.
pub fn linear(fmin: f32, fmax: f32, n_bins: usize) -> Vec<f32> {
    if n_bins <= 1 {
        return vec![fmin];
    }
    let step = (fmax - fmin) / (n_bins - 1) as f32;
    (0..n_bins).map(|i| fmin + step * i as f32).collect()
}

/// Generate mel-spaced frequencies.
///
/// `n_mels` frequencies between `fmin` and `fmax` Hz, spaced uniformly on the mel scale.
pub fn mel(n_mels: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let step = (mel_max - mel_min) / (n_mels - 1) as f32;
    (0..n_mels)
        .map(|i| mel_to_hz(mel_min + step * i as f32))
        .collect()
}

/// Convert Hz to mel scale.
pub fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Convert mel scale to Hz.
pub fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
}
