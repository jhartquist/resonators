/// Parameters for a single resonator.
///
/// - `freq`: resonant frequency, in Hz.
/// - `alpha`: EWMA coefficient in `(0, 1]`. Smaller `alpha` means a longer
///   time constant; `1.0` is no smoothing.
/// - `beta`: coefficient of a second EWMA on the resonator's output, in
///   `(0, 1]`. Typically equal to `alpha`; smaller `beta` gives a smoother
///   output at the cost of additional latency.
///
/// See [`heuristic_alpha`](crate::heuristic_alpha) for a reasonable
/// per-frequency default.
#[derive(Debug, Clone, Copy)]
pub struct ResonatorConfig {
    pub freq: f32,
    pub alpha: f32,
    pub beta: f32,
}

impl ResonatorConfig {
    /// Creates a new config with the given frequency, alpha, and beta.
    pub fn new(freq: f32, alpha: f32, beta: f32) -> Self {
        Self { freq, alpha, beta }
    }
}
