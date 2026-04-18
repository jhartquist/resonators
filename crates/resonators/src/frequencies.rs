pub fn midi_to_hz(midi: f32, tuning: f32) -> f32 {
    tuning * 2.0f32.powf((midi - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn midi_to_hz_a4() {
        assert!((midi_to_hz(69.0, 440.0) - 440.0).abs() < 1e-4);
    }

    #[test]
    fn midi_to_hz_octave_doubles() {
        let a4 = midi_to_hz(69.0, 440.0);
        let a5 = midi_to_hz(81.0, 440.0);
        assert!((a5 - 2.0 * a4).abs() < 1e-4);
    }
}
