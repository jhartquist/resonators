// Verifies the reformulated OnePoleBank against the same fixtures the
// reference ResonatorBank uses (generated from noFFT). The claim being tested:
// power/magnitude are identical, and the absolute complex value is recoverable
// by lazy de-rotation.

use npyz::npz::NpzArchive;
use resonators::{OnePoleBank, ResonatorConfig};

const SAMPLE_RATE: f32 = 44100.0;
const HOP_SIZE: usize = 256;

struct Fixture {
    signal: Vec<f32>,
    configs: Vec<ResonatorConfig>,
    ref_re: Vec<f32>,
    ref_im: Vec<f32>,
    n_frames: usize,
    n_bins: usize,
}

impl Fixture {
    fn load() -> Self {
        let npz_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/chirp_88.npz");
        let mut npz = NpzArchive::open(npz_path).unwrap();
        let signal: Vec<f32> = npz.by_name("signal").unwrap().unwrap().into_vec().unwrap();
        let freqs: Vec<f32> = npz.by_name("freqs").unwrap().unwrap().into_vec().unwrap();
        let alphas: Vec<f32> = npz.by_name("alphas").unwrap().unwrap().into_vec().unwrap();
        let ref_arr = npz.by_name("ref").unwrap().unwrap();
        let shape = ref_arr.shape().to_vec(); // [frames, 2, bins]
        let flat: Vec<f32> = ref_arr.into_vec().unwrap();
        let n_frames = shape[0] as usize;
        let n_bins = shape[2] as usize;
        let mut ref_re = vec![0.0f32; n_frames * n_bins];
        let mut ref_im = vec![0.0f32; n_frames * n_bins];
        for f in 0..n_frames {
            for b in 0..n_bins {
                ref_re[f * n_bins + b] = flat[f * 2 * n_bins + b];
                ref_im[f * n_bins + b] = flat[f * 2 * n_bins + n_bins + b];
            }
        }
        let configs = freqs
            .iter()
            .zip(&alphas)
            .map(|(&f, &a)| ResonatorConfig::new(f, a, a))
            .collect();
        Self {
            signal,
            configs,
            ref_re,
            ref_im,
            n_frames,
            n_bins,
        }
    }
}

#[test]
fn onepole_power_matches_reference() {
    let fx = Fixture::load();
    let mut bank = OnePoleBank::new(&fx.configs, SAMPLE_RATE);

    let mut max_pow_err: f32 = 0.0;
    let mut max_ref_pow: f32 = 0.0;
    for frame in 0..fx.n_frames {
        let start = frame * HOP_SIZE;
        bank.process_samples(&fx.signal[start..start + HOP_SIZE]);
        for bin in 0..fx.n_bins {
            let idx = frame * fx.n_bins + bin;
            let ref_pow = fx.ref_re[idx] * fx.ref_re[idx] + fx.ref_im[idx] * fx.ref_im[idx];
            max_ref_pow = max_ref_pow.max(ref_pow);
            max_pow_err = max_pow_err.max((bank.power(bin) - ref_pow).abs());
        }
    }
    let rel = max_pow_err / max_ref_pow;
    eprintln!("max|Δpower| = {max_pow_err:.3e}, relative = {rel:.3e}");
    // f32 round-off only; the two are mathematically identical.
    assert!(rel < 1e-4, "relative power error too large: {rel:e}");
}

#[test]
fn onepole_complex_matches_reference_via_derotation() {
    let fx = Fixture::load();
    let mut bank = OnePoleBank::new(&fx.configs, SAMPLE_RATE);

    let mut max_err: f32 = 0.0;
    for frame in 0..fx.n_frames {
        let start = frame * HOP_SIZE;
        bank.process_samples(&fx.signal[start..start + HOP_SIZE]);
        for bin in 0..fx.n_bins {
            let c = bank.complex(bin);
            let idx = frame * fx.n_bins + bin;
            max_err = max_err.max((c.re - fx.ref_re[idx]).abs());
            max_err = max_err.max((c.im - fx.ref_im[idx]).abs());
        }
    }
    eprintln!("max|Δcomplex| (de-rotated) = {max_err:.3e}");
    // Looser than power: de-rotation reintroduces f32 trig at large sample
    // indices, so absolute phase drifts a little. Power is the robust quantity.
    assert!(max_err < 5e-2, "complex error too large: {max_err:e}");
}
