pub(crate) const STABILIZE_EVERY: u64 = 256;

mod bank;
mod config;
mod dynamics;
mod frequencies;
mod resonator;

pub use bank::ResonatorBank;
pub use config::ResonatorConfig;
pub use dynamics::{alpha_from_tau, alpha_heuristic, tau_from_alpha};
pub use frequencies::midi_to_hz;
pub use resonator::Resonator;
