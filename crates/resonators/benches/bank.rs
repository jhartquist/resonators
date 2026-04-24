use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use resonators::{ResonatorBank, ResonatorConfig, heuristic_alpha, midi_to_hz};

const SAMPLE_RATE: f32 = 44100.0;
const TUNING: f32 = 440.0;
const MIDI_LOW: f32 = 21.0; // A0
const MIDI_HIGH: f32 = 108.0; // C8

fn log_spaced_configs(n_bins: usize) -> Vec<ResonatorConfig> {
    let f_low = midi_to_hz(MIDI_LOW, TUNING);
    let f_high = midi_to_hz(MIDI_HIGH, TUNING);
    let ratio = (f_high / f_low).ln();
    (0..n_bins)
        .map(|i| {
            let freq = f_low * (ratio * i as f32 / (n_bins - 1) as f32).exp();
            let alpha = heuristic_alpha(freq, SAMPLE_RATE);
            ResonatorConfig::new(freq, alpha, alpha)
        })
        .collect()
}

const BIN_COUNTS: &[usize] = &[88, 264, 440, 880];

fn bench_bank(c: &mut Criterion) {
    let n = SAMPLE_RATE as usize; // 1 second of a 440 Hz sine wave
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE).sin())
        .collect();

    // Scalar path — on x86_64 with -C target-cpu=native or equivalent,
    // LLVM will auto-vectorise this to whatever the build target
    // supports (SSE2 baseline, AVX2 if enabled, etc).
    {
        let mut group = c.benchmark_group("bank/scalar");
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(50);

        for &n_bins in BIN_COUNTS {
            let configs = log_spaced_configs(n_bins);
            group.bench_with_input(
                BenchmarkId::from_parameter(n_bins),
                &configs,
                |bencher, configs| {
                    let mut bank = ResonatorBank::new(configs, SAMPLE_RATE);
                    bencher.iter(|| {
                        bank.reset();
                        for &sample in &signal {
                            bank.process_sample(black_box(sample));
                        }
                    });
                },
            );
        }
        group.finish();
    }

    // Explicit AVX2 + FMA — 8 bins per iteration via __m256 + vfmadd231ps.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
            let mut group = c.benchmark_group("bank/avx2");
            group.throughput(Throughput::Elements(n as u64));
            group.sample_size(50);

            for &n_bins in BIN_COUNTS {
                let configs = log_spaced_configs(n_bins);
                group.bench_with_input(
                    BenchmarkId::from_parameter(n_bins),
                    &configs,
                    |bencher, configs| {
                        let mut bank = ResonatorBank::new(configs, SAMPLE_RATE);
                        bencher.iter(|| {
                            bank.reset();
                            // Safety: we've checked avx2+fma support above.
                            unsafe {
                                for &sample in &signal {
                                    bank.process_sample_avx2(black_box(sample));
                                }
                            }
                        });
                    },
                );
            }
            group.finish();
        } else {
            eprintln!("SKIPPED bank/avx2 — CPU lacks avx2 or fma");
        }
    }

    // Explicit AVX-512F — 16 bins per iteration via __m512.
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            let mut group = c.benchmark_group("bank/avx512");
            group.throughput(Throughput::Elements(n as u64));
            group.sample_size(50);

            for &n_bins in BIN_COUNTS {
                let configs = log_spaced_configs(n_bins);
                group.bench_with_input(
                    BenchmarkId::from_parameter(n_bins),
                    &configs,
                    |bencher, configs| {
                        let mut bank = ResonatorBank::new(configs, SAMPLE_RATE);
                        bencher.iter(|| {
                            bank.reset();
                            // Safety: we've checked avx512f support above.
                            unsafe {
                                for &sample in &signal {
                                    bank.process_sample_avx512(black_box(sample));
                                }
                            }
                        });
                    },
                );
            }
            group.finish();
        } else {
            eprintln!("SKIPPED bank/avx512 — CPU lacks avx512f");
        }
    }
}

criterion_group!(benches, bench_bank);
criterion_main!(benches);
