# Changelog

All notable changes to this project will be documented in this file. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-04-24

### Changed

- WASM: ~13-15x speedup in `ResonatorBank::process_sample` / `process_samples` at 88-880 bins, by recovering autovectorization that was defeated by `f32::mul_add` lowering to per-lane `fmaf` calls on wasm32+simd128. Diagnosis by @pengowray (#1).

## [Python 0.1.1] - 2026-04-22

### Changed

- Linux and Windows x86_64 wheels now target `x86-64-v3` (Haswell, 2013+) to enable AVX2 + FMA auto-vectorization in the resonator hot loop. On Linux, this is typically a 20-50x speedup over the `0.1.0` wheel. Users on pre-Haswell CPUs should install from sdist with custom `RUSTFLAGS`.

## [0.1.0] - 2026-04-21

### Added

- `ResonatorBank` and `Resonator` types implementing the Resonate algorithm for low-latency spectral analysis.
- Streaming (`process_samples`) and one-shot (`resonate`) processing modes.
- `heuristic_alpha` / `heuristic_alphas`, `alpha_from_tau` / `tau_from_alpha`, `midi_to_hz` helpers.
- Python bindings via PyO3.
- WebAssembly bindings via wasm-bindgen.
