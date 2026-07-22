// Apples-to-apples throughput: reference ResonatorBank (running phasor +
// periodic stabilization) vs the reformulated OnePoleBank (two cascaded complex
// one-poles, no phasor, no stabilization). Same SoA layout, same mul_add.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use resonators::{OnePoleBank, ResonatorBank, ResonatorConfig, heuristic_alpha, midi_to_hz};

const SAMPLE_RATE: f32 = 44100.0;
const TUNING: f32 = 440.0;
const MIDI_LOW: f32 = 21.0;
const MIDI_HIGH: f32 = 108.0;

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

fn bench_compare(c: &mut Criterion) {
    let n = SAMPLE_RATE as usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE).sin())
        .collect();

    let mut group = c.benchmark_group("phasor_vs_onepole");
    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(50);

    for &n_bins in &[88, 264, 440, 880] {
        let configs = log_spaced_configs(n_bins);

        group.bench_with_input(
            BenchmarkId::new("phasor", n_bins),
            &configs,
            |b, configs| {
                let mut bank = ResonatorBank::new(configs, SAMPLE_RATE);
                b.iter(|| {
                    bank.reset();
                    for &s in &signal {
                        bank.process_sample(black_box(s));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("onepole", n_bins),
            &configs,
            |b, configs| {
                let mut bank = OnePoleBank::new(configs, SAMPLE_RATE);
                b.iter(|| {
                    bank.reset();
                    for &s in &signal {
                        bank.process_sample(black_box(s));
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_compare);
criterion_main!(benches);
