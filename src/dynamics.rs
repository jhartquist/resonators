pub fn alpha_heuristic(freq: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt * freq / (1.0 + freq).log10()).exp()
}

pub fn alpha_from_tau(tau: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt / tau).exp()
}

pub fn tau_from_alpha(alpha: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    -dt / (1.0 - alpha).ln()
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

    #[test]
    fn alpha_tau_roundtrip() {
        let sr = 44100.0;
        let tau = 0.05;
        let alpha = alpha_from_tau(tau, sr);
        let tau_back = tau_from_alpha(alpha, sr);
        assert!((tau - tau_back).abs() < 1e-6);
    }
}
