use npyz::npz::NpzArchive;
use resonators::{Resonator, ResonatorBank, ResonatorConfig};

const SAMPLE_RATE: f32 = 44100.0;
const HOP_SIZE: usize = 256;
const TOLERANCE: f32 = 1e-3;

struct Fixture {
    signal: Vec<f32>,
    configs: Vec<ResonatorConfig>,
    ref_re: Vec<Vec<f32>>, // [frame][bin]
    ref_im: Vec<Vec<f32>>,
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

        let mut ref_re = vec![vec![0.0; n_bins]; n_frames];
        let mut ref_im = vec![vec![0.0; n_bins]; n_frames];
        for f in 0..n_frames {
            for b in 0..n_bins {
                ref_re[f][b] = flat[f * 2 * n_bins + b];
                ref_im[f][b] = flat[f * 2 * n_bins + n_bins + b];
            }
        }

        let configs: Vec<ResonatorConfig> = freqs
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
fn bank_matches_reference() {
    let fx = Fixture::load();
    let mut bank = ResonatorBank::new(&fx.configs, SAMPLE_RATE);

    for frame in 0..fx.n_frames {
        let start = frame * HOP_SIZE;
        bank.process_samples(&fx.signal[start..start + HOP_SIZE]);

        for bin in 0..fx.n_bins {
            let (re, im) = bank.complex(bin);
            assert!(
                (re - fx.ref_re[frame][bin]).abs() < TOLERANCE,
                "frame {frame} bin {bin} re: {re} vs {}",
                fx.ref_re[frame][bin]
            );
            assert!(
                (im - fx.ref_im[frame][bin]).abs() < TOLERANCE,
                "frame {frame} bin {bin} im: {im} vs {}",
                fx.ref_im[frame][bin]
            );
        }
    }
}

#[test]
fn resonator_matches_reference() {
    let fx = Fixture::load();

    for bin in 0..fx.n_bins {
        let mut r = Resonator::new(fx.configs[bin], SAMPLE_RATE);
        for frame in 0..fx.n_frames {
            let start = frame * HOP_SIZE;
            r.process_samples(&fx.signal[start..start + HOP_SIZE]);
            let (re, im) = r.complex();
            assert!(
                (re - fx.ref_re[frame][bin]).abs() < TOLERANCE,
                "frame {frame} bin {bin} re: {re} vs {}",
                fx.ref_re[frame][bin]
            );
            assert!(
                (im - fx.ref_im[frame][bin]).abs() < TOLERANCE,
                "frame {frame} bin {bin} im: {im} vs {}",
                fx.ref_im[frame][bin]
            );
        }
    }
}
