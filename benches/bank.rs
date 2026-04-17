use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use resonators::{ResonatorBank, ResonatorConfig};

const SAMPLE_RATE: f32 = 44100.0;
const F_LOW: f32 = 27.5; // A0
const F_HIGH: f32 = 4186.0; // C8

fn alpha_heuristic(freq: f32) -> f32 {
    1.0 - (-(1.0 / SAMPLE_RATE) * freq / (1.0 + freq).log10()).exp()
}

fn log_spaced_configs(n_bins: usize) -> Vec<ResonatorConfig> {
    let ratio = (F_HIGH / F_LOW).ln();
    (0..n_bins)
        .map(|i| {
            let freq = F_LOW * (ratio * i as f32 / (n_bins - 1) as f32).exp();
            let alpha = alpha_heuristic(freq);
            ResonatorConfig::new(freq, alpha, alpha)
        })
        .collect()
}

fn bench_bank(c: &mut Criterion) {
    let n = SAMPLE_RATE as usize; // 1 second of a 440 Hz sine wave
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE).sin())
        .collect();

    let mut group = c.benchmark_group("bank");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(50);

    for &n_bins in &[88, 264, 440, 880] {
        let configs = log_spaced_configs(n_bins);
        group.bench_with_input(
            BenchmarkId::from_parameter(n_bins),
            &configs,
            |bencher, configs| {
                bencher.iter(|| {
                    let mut bank = ResonatorBank::new(configs, SAMPLE_RATE);
                    for &sample in &signal {
                        bank.process_sample(black_box(sample));
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_bank);
criterion_main!(benches);
