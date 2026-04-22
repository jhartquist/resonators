# Changelog

All notable changes to this project will be documented in this file. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-04-21

### Added

- `ResonatorBank` and `Resonator` types implementing the Resonate algorithm for low-latency spectral analysis.
- Streaming (`process_samples`) and one-shot (`resonate`) processing modes.
- `heuristic_alpha` / `heuristic_alphas`, `alpha_from_tau` / `tau_from_alpha`, `midi_to_hz` helpers.
- Python bindings via PyO3.
- WebAssembly bindings via wasm-bindgen.
