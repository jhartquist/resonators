use resonators::{ResonatorBank, ResonatorConfig, heuristic_alpha, midi_to_hz};
use std::f32::consts::PI;

const SAMPLE_RATE: f32 = 44100.0;
const HOP_SIZE: usize = 256;

fn main() {
    // 88 MIDI piano bins, A0 to C8
    let configs: Vec<ResonatorConfig> = (21..=108)
        .map(|midi| {
            let freq = midi_to_hz(midi as f32, 440.0);
            let alpha = heuristic_alpha(freq, SAMPLE_RATE);
            ResonatorConfig::new(freq, alpha, alpha)
        })
        .collect();
    let mut bank = ResonatorBank::new(&configs, SAMPLE_RATE);

    // 1-second log chirp sweeping A0 -> C8
    let n = SAMPLE_RATE as usize;
    let (f0, f1) = (27.5_f32, 4186.0_f32);
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            let phase = 2.0 * PI * f0 / (f1 / f0).ln() * ((f1 / f0).powf(t) - 1.0);
            0.5 * phase.cos()
        })
        .collect();

    // feed chunks as if they were arriving from a live stream
    for (frame, chunk) in signal.chunks(HOP_SIZE).enumerate() {
        bank.process_samples(chunk);

        // read the bank's state after each chunk: find the peak bin
        let (peak, power) = bank
            .powers()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap();

        if frame % 20 == 0 {
            println!(
                "frame {frame:3}: peak bin {peak:2} ({:6.1} Hz), power {power:.4}",
                bank.freq(peak)
            );
        }
    }
}
