pub fn alpha_heuristic(freq: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt * freq / (1.0 + freq).log10()).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_heuristic_in_valid_range() {
        for &freq in &[27.5, 440.0, 4186.0] {
            let a = alpha_heuristic(freq, 44100.0);
            assert!(a > 0.0 && a < 1.0, "alpha({freq}) = {a}");
        }
    }

    #[test]
    fn alpha_heuristic_increases_with_freq() {
        let sr = 44100.0;
        assert!(alpha_heuristic(4000.0, sr) > alpha_heuristic(100.0, sr));
    }
}
