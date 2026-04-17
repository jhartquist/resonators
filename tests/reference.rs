use npyz::npz::NpzArchive;
use resonators::{ResonatorBank, ResonatorConfig};

#[test]
fn matches_reference() {
    let npz_path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chirp_88.npz");
    let mut npz = NpzArchive::open(npz_path).unwrap();

    let signal: Vec<f32> = npz.by_name("signal").unwrap().unwrap().into_vec().unwrap();
    let freqs: Vec<f32> = npz.by_name("freqs").unwrap().unwrap().into_vec().unwrap();
    let alphas: Vec<f32> = npz.by_name("alphas").unwrap().unwrap().into_vec().unwrap();

    let ref_arr = npz.by_name("ref").unwrap().unwrap();
    let ref_shape = ref_arr.shape().to_vec(); // [num_frames, num_parts (complex number: 2), num_bins]
    let ref_flat: Vec<f32> = ref_arr.into_vec().unwrap();

    let n_frames = ref_shape[0] as usize;
    let n_bins = ref_shape[2] as usize;

    const SAMPLE_RATE: f32 = 44100.0;
    const HOP_SIZE: usize = 256;

    let idx = |frame: usize, part: usize, bin: usize| (frame * 2 * n_bins) + (part * n_bins) + bin;

    let configs: Vec<ResonatorConfig> = freqs
        .iter()
        .zip(&alphas)
        .map(|(&f, &a)| ResonatorConfig::new(f, a, a))
        .collect();
    let mut bank = ResonatorBank::new(&configs, SAMPLE_RATE);

    for frame in 0..n_frames {
        for i in 0..HOP_SIZE {
            bank.process_sample(signal[frame * HOP_SIZE + i]);
        }

        for bin in 0..bank.len() {
            let (re, im) = bank.complex(bin);

            let expected_re = ref_flat[idx(frame, 0, bin)];
            let expected_im = ref_flat[idx(frame, 1, bin)];

            assert!(
                (re - expected_re).abs() < 1e-3,
                "frame {frame} bin {bin} re: {re} vs {expected_re}"
            );
            assert!(
                (im - expected_im).abs() < 1e-3,
                "frame {frame} bin {bin} im: {im} vs {expected_im}"
            );
        }
    }
}
