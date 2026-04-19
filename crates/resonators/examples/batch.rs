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

    // 1 second of a 440 Hz cosine (A4)
    let n = SAMPLE_RATE as usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * 440.0 * i as f32 / SAMPLE_RATE).cos())
        .collect();

    // one-shot: process the whole signal and get complex output per frame
    let spectrogram = bank.resonate(&signal, HOP_SIZE);
    let n_frames = spectrogram.len() / bank.len();
    let last_frame = &spectrogram[(n_frames - 1) * bank.len()..];

    // find the strongest bin in the final frame
    let (peak, peak_value) = last_frame
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.norm_sqr().partial_cmp(&b.1.norm_sqr()).unwrap())
        .unwrap();

    println!("processed {n} samples, {n_frames} frames");
    println!(
        "peak bin at final frame: {peak} ({:.1} Hz), magnitude {:.3}",
        bank.freq(peak),
        peak_value.norm()
    );
}
