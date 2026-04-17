pub struct Resonator {
    freq: f32,
    alpha: f32,
    beta: f32,
    sample_rate: f32,
}

impl Resonator {
    pub fn new(freq: f32, alpha: f32, beta: f32, sample_rate: f32) -> Self {
        Self {
            freq,
            alpha,
            beta,
            sample_rate,
        }
    }

    pub fn process_sample(&mut self, sample: f32) {
        todo!()
    }

    pub fn complex(&self) -> (f32, f32) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use npyz::npz::NpzArchive;

    use super::*;

    #[test]
    fn resonator_new() {
        let resonator = Resonator::new(440.0, 1.0, 2.0, 44100.0);
        assert_eq!(resonator.freq, 440.0);
        assert_eq!(resonator.alpha, 1.0);
        assert_eq!(resonator.beta, 2.0);
        assert_eq!(resonator.sample_rate, 44100.0);
    }

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

        let idx =
            |frame: usize, part: usize, bin: usize| (frame * 2 * n_bins) + (part * n_bins) + bin;

        for bin in 0..n_bins {
            let mut resonator = Resonator::new(
                freqs[bin],
                alphas[bin],
                alphas[bin], // beta = alpha
                SAMPLE_RATE,
            );

            for frame in 0..n_frames {
                for i in 0..HOP_SIZE {
                    let sample = signal[frame * HOP_SIZE + i];
                    resonator.process_sample(sample);
                }
                let (re, im) = resonator.complex();

                let expected_re = ref_flat[idx(frame, 0, bin)];
                let expected_im = ref_flat[idx(frame, 1, bin)];

                assert!(
                    (re - expected_re).abs() < 1e-5,
                    "frame {frame} re: {re} vs {expected_re}"
                );
                assert!(
                    (im - expected_im).abs() < 1e-5,
                    "frame {frame} im: {im} vs {expected_im}"
                );
            }
        }
    }
}
