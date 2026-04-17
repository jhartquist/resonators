use std::f32::consts::PI;

const STABILIZE_EVERY: u64 = 256;

pub struct Resonator {
    freq: f32,
    alpha: f32,
    beta: f32,

    // phasor state, rotates by phasor angle (w) each sample
    z_re: f32,
    z_im: f32,

    // phasor angle, constant
    w_re: f32,
    w_im: f32,

    // raw output of EWMA
    r_re: f32,
    r_im: f32,

    // smoothed output of EWMA
    rr_re: f32,
    rr_im: f32,

    // tracked for stabilization
    sample_count: u64,
}

impl Resonator {
    pub fn new(freq: f32, alpha: f32, beta: f32, sample_rate: f32) -> Self {
        let phasor_angle = -2.0 * PI * freq / sample_rate;
        Self {
            freq,
            alpha,
            beta,
            z_re: 1.0,
            z_im: 0.0,
            w_re: phasor_angle.cos(),
            w_im: phasor_angle.sin(),
            r_re: 0.0,
            r_im: 0.0,
            rr_re: 0.0,
            rr_im: 0.0,
            sample_count: 0,
        }
    }

    pub fn process_sample(&mut self, sample: f32) {
        // update raw output via EWMA
        self.r_re = (1.0 - self.alpha) * self.r_re + self.alpha * sample * self.z_re;
        self.r_im = (1.0 - self.alpha) * self.r_im + self.alpha * sample * self.z_im;

        // update smoothed output via second EMWA
        self.rr_re = (1.0 - self.beta) * self.rr_re + self.beta * self.r_re;
        self.rr_im = (1.0 - self.beta) * self.rr_im + self.beta * self.r_im;

        // rotate phasor (complex multiply)
        let z_re = self.z_re;
        let z_im = self.z_im;
        self.z_re = z_re * self.w_re - z_im * self.w_im;
        self.z_im = z_re * self.w_im + z_im * self.w_re;

        // occasional phasor stabilization
        self.sample_count += 1;
        if self.sample_count.is_multiple_of(STABILIZE_EVERY) {
            let mag = (self.z_re * self.z_re + self.z_im * self.z_im).sqrt();
            self.z_re /= mag;
            self.z_im /= mag;
        }
    }

    pub fn complex(&self) -> (f32, f32) {
        (self.rr_re, self.rr_im)
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
                    (re - expected_re).abs() < 1e-3,
                    "frame {frame} re: {re} vs {expected_re}"
                );
                assert!(
                    (im - expected_im).abs() < 1e-3,
                    "frame {frame} im: {im} vs {expected_im}"
                );
            }
        }
    }
}
