use std::f32::consts::TAU;

/// Alpha heuristic from Equation 7, François (ICMC 2025).
///
/// `alpha_f = 1 - exp(-dt * f / log10(1 + f))`
///
/// where `dt = 1 / sample_rate`.
pub fn alpha_heuristic(frequency: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt * frequency / (1.0 + frequency).log10()).exp()
}

/// Compute alpha from time constant tau (Equation 6).
///
/// `alpha = 1 - exp(-dt / tau)`
pub fn alpha_from_tau(tau: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt / tau).exp()
}

/// Compute time constant tau from alpha (Equation 6).
///
/// `tau = -dt / ln(1 - alpha)`
pub fn tau_from_alpha(alpha: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    -dt / (1.0 - alpha).ln()
}

/// EWMA coefficient for smoothing `R` (output) with time constant `tau`.
///
/// Used in the self-tuning resonator (François 2026 extension): `R̃ ← (1−β)R̃ + β·R`
/// applied after the main `α`-accumulation. Shape matches [`alpha_from_tau`].
pub fn beta_from_tau(tau: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt / tau).exp()
}

/// EWMA coefficient for the frequency-tracking update with time constant `tau`.
///
/// Used in the self-tuning resonator (François 2026 extension):
/// `ω ← ω + γ·(ω_obs − ω)` where `ω_obs = arg(R̃·conj(R̃_prev))/Δt`.
/// Small γ (long τ_γ) = slow, stable tracking; large γ = aggressive slew.
pub fn gamma_from_tau(tau: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt / tau).exp()
}

/// Compute the phasor rotation angle for a given frequency and sample rate.
///
/// Returns `-2π * frequency / sample_rate`.
pub fn phasor_angle(frequency: f32, sample_rate: f32) -> f32 {
    -TAU * frequency / sample_rate
}
