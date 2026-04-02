/// Configuration for a single resonator.
///
/// Describes the resonant frequency and EWMA parameters (alpha, beta)
/// as defined in François (ICMC 2025).
#[derive(Debug, Clone, Copy)]
pub struct ResonatorConfig {
    pub frequency: f32,
    pub alpha: f32,
    pub beta: f32,
}

impl ResonatorConfig {
    pub fn new(frequency: f32, alpha: f32, beta: f32) -> Self {
        Self {
            frequency,
            alpha,
            beta,
        }
    }

    /// Create a config where beta = alpha (common case).
    pub fn with_alpha(frequency: f32, alpha: f32) -> Self {
        Self::new(frequency, alpha, alpha)
    }
}
