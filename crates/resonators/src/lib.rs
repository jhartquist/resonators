//! A Rust implementation of Alexandre François's [Resonate algorithm] for
//! low-latency spectral analysis.
//!
//! The primary type is [`ResonatorBank`], a collection of independent
//! resonators each tuned to a fixed frequency. Feed samples in one at a time
//! with [`process_sample`](ResonatorBank::process_sample) or in chunks with
//! [`process_samples`](ResonatorBank::process_samples). Read per-bin
//! magnitudes, powers, phases, or complex values at any time. For one-shot
//! processing of a full signal into a spectrogram-like output, use
//! [`resonate`](ResonatorBank::resonate).
//!
//! [`Resonator`] is the single-resonator analogue, useful when only one bin
//! is needed.
//!
//! In addition to its frequency, each resonator has two EWMA coefficients:
//! `alpha` sets the time constant, and `beta` smooths the output. See
//! [`ResonatorConfig`] for parameter details, or the [paper] for the full
//! algorithm.
//!
//! # Example
//!
//! The following example uses the common case, where both `alpha` and `beta`
//! default to a per-frequency heuristic.
//!
//! ```
//! use resonators::ResonatorBank;
//!
//! let sample_rate = 44_100.0;
//! let freqs = [110.0, 220.0, 440.0, 880.0];
//! let mut bank = ResonatorBank::from_frequencies(&freqs, sample_rate);
//!
//! bank.process_samples(&vec![0.0f32; 1024]);
//! let magnitudes = bank.magnitudes();
//! assert_eq!(magnitudes.len(), 4);
//! ```
//!
//! The following example shows the equivalent with explicit
//! [`ResonatorConfig`] values, useful when you need custom `alpha` or `beta`.
//!
//! ```
//! use resonators::{ResonatorBank, ResonatorConfig, heuristic_alpha};
//!
//! let sample_rate = 44_100.0;
//! let configs: Vec<_> = [110.0, 220.0, 440.0, 880.0]
//!     .iter()
//!     .map(|&freq| {
//!         let alpha = heuristic_alpha(freq, sample_rate);
//!         let beta = alpha;
//!         ResonatorConfig::new(freq, alpha, beta)
//!     })
//!     .collect();
//! let mut bank = ResonatorBank::new(&configs, sample_rate);
//!
//! bank.process_samples(&vec![0.0f32; 1024]);
//! ```
//!
//! [Resonate algorithm]: https://alexandrefrancois.org/Resonate/
//! [paper]: https://alexandrefrancois.org/assets/publications/FrancoisARJ-ICMC2025.pdf

pub(crate) const STABILIZE_EVERY: u64 = 256;

mod bank;
mod config;
mod dynamics;
mod frequencies;
mod resonator;

pub use bank::ResonatorBank;
pub use config::ResonatorConfig;
pub use dynamics::{alpha_from_tau, heuristic_alpha, heuristic_alphas, tau_from_alpha};
pub use frequencies::midi_to_hz;
pub use resonator::Resonator;
