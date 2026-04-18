pub fn heuristic_alpha(freq: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt * freq / (1.0 + freq).log10()).exp()
}

pub fn heuristic_alphas(freqs: &[f32], sample_rate: f32) -> Vec<f32> {
    freqs
        .iter()
        .map(|&f| heuristic_alpha(f, sample_rate))
        .collect()
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
    fn heuristic_alpha_in_valid_range() {
        for &freq in &[27.5, 440.0, 4186.0] {
            let a = heuristic_alpha(freq, 44100.0);
            assert!(a > 0.0 && a < 1.0, "alpha({freq}) = {a}");
        }
    }

    #[test]
    fn heuristic_alpha_increases_with_freq() {
        let sr = 44100.0;
        assert!(heuristic_alpha(4000.0, sr) > heuristic_alpha(100.0, sr));
    }

    #[test]
    fn heuristic_alphas_matches_scalar() {
        let sr = 44100.0;
        let freqs = vec![27.5, 440.0, 4186.0];
        let alphas = heuristic_alphas(&freqs, sr);
        for (i, &f) in freqs.iter().enumerate() {
            assert_eq!(alphas[i], heuristic_alpha(f, sr));
        }
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
