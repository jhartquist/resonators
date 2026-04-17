pub struct Resonator {
    pub freq: f32,
    pub alpha: f32,
    pub beta: f32,
    pub sample_rate: f32,
}

impl Resonator {
    pub fn new(freq: f32, alpha: f32, beta: f32, sample_rate: f32) -> Self {
        Self {
            freq,
            alpha,
            beta,
            sample_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_new() {
        let config = Resonator::new(440.0, 1.0, 2.0, 44100.0);
        assert_eq!(config.freq, 440.0);
        assert_eq!(config.alpha, 1.0);
        assert_eq!(config.beta, 2.0);
        assert_eq!(config.sample_rate, 44100.0);
    }
}
