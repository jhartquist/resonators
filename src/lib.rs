use std::f32::consts::PI;

const STABILIZE_EVERY: u64 = 256;

pub fn alpha_heuristic(freq: f32, sample_rate: f32) -> f32 {
    let dt = 1.0 / sample_rate;
    1.0 - (-dt * freq / (1.0 + freq).log10()).exp()
}

pub fn midi_to_hz(midi: f32, tuning: f32) -> f32 {
    tuning * 2.0f32.powf((midi - 69.0) / 12.0)
}

#[derive(Debug, Clone, Copy)]
pub struct ResonatorConfig {
    pub freq: f32,
    pub alpha: f32,
    pub beta: f32,
}

impl ResonatorConfig {
    pub fn new(freq: f32, alpha: f32, beta: f32) -> Self {
        Self { freq, alpha, beta }
    }
}

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
    pub fn new(config: ResonatorConfig, sample_rate: f32) -> Self {
        let ResonatorConfig { freq, alpha, beta } = config;
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
            self.stabilize();
        }
    }

    fn stabilize(&mut self) {
        let mag = (self.z_re * self.z_re + self.z_im * self.z_im).sqrt();
        self.z_re /= mag;
        self.z_im /= mag;
    }

    pub fn freq(&self) -> f32 {
        self.freq
    }

    pub fn power(&self) -> f32 {
        self.rr_re * self.rr_re + self.rr_im * self.rr_im
    }

    pub fn magnitude(&self) -> f32 {
        self.power().sqrt()
    }

    pub fn phase(&self) -> f32 {
        self.rr_im.atan2(self.rr_re)
    }

    pub fn complex(&self) -> (f32, f32) {
        (self.rr_re, self.rr_im)
    }
}

pub struct ResonatorBank {
    resonators: Vec<Resonator>,
}

impl ResonatorBank {
    pub fn new(configs: &[ResonatorConfig], sample_rate: f32) -> Self {
        Self {
            resonators: configs
                .iter()
                .map(|&config| Resonator::new(config, sample_rate))
                .collect(),
        }
    }

    pub fn process_sample(&mut self, sample: f32) {
        for r in &mut self.resonators {
            r.process_sample(sample);
        }
    }

    pub fn len(&self) -> usize {
        self.resonators.len()
    }

    pub fn freq(&self, i: usize) -> f32 {
        self.resonators[i].freq()
    }

    pub fn magnitude(&self, i: usize) -> f32 {
        self.resonators[i].magnitude()
    }

    pub fn phase(&self, i: usize) -> f32 {
        self.resonators[i].phase()
    }

    pub fn power(&self, i: usize) -> f32 {
        self.resonators[i].power()
    }

    pub fn complex(&self, i: usize) -> (f32, f32) {
        self.resonators[i].complex()
    }

    pub fn frequencies(&self) -> Vec<f32> {
        self.resonators.iter().map(|r| r.freq()).collect()
    }

    pub fn magnitudes(&self) -> Vec<f32> {
        self.resonators.iter().map(|r| r.magnitude()).collect()
    }

    pub fn phases(&self) -> Vec<f32> {
        self.resonators.iter().map(|r| r.phase()).collect()
    }

    pub fn powers(&self) -> Vec<f32> {
        self.resonators.iter().map(|r| r.power()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_is_magnitude_squared() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.rr_re = 3.0;
        r.rr_im = 4.0;
        assert_eq!(r.power(), 25.0);
    }

    #[test]
    fn magnitude_is_sqrt_of_power() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.rr_re = 3.0;
        r.rr_im = 4.0;
        assert_eq!(r.magnitude(), 5.0);
    }

    #[test]
    fn phase_uses_atan2() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.rr_re = 1.0;
        r.rr_im = 0.0;
        assert_eq!(r.phase(), 0.0);

        r.rr_re = 0.0;
        r.rr_im = 1.0;
        assert!((r.phase() - std::f32::consts::FRAC_PI_2).abs() < 1e-6);

        r.rr_re = -1.0;
        r.rr_im = 0.0;
        assert!((r.phase() - PI).abs() < 1e-6);
    }

    #[test]
    fn midi_to_hz_a4() {
        assert!((midi_to_hz(69.0, 440.0) - 440.0).abs() < 1e-4);
    }

    #[test]
    fn midi_to_hz_octave_doubles() {
        let a4 = midi_to_hz(69.0, 440.0);
        let a5 = midi_to_hz(81.0, 440.0);
        assert!((a5 - 2.0 * a4).abs() < 1e-4);
    }

    #[test]
    fn alpha_heuristic_in_valid_range() {
        for &freq in &[27.5, 440.0, 4186.0] {
            let a = alpha_heuristic(freq, 44100.0);
            assert!(a > 0.0 && a < 1.0, "alpha({freq}) = {a}");
        }
    }

    #[test]
    fn alpha_heuristic_increases_with_freq() {
        let sr = 44100.0;
        assert!(alpha_heuristic(4000.0, sr) > alpha_heuristic(100.0, sr));
    }

    #[test]
    fn stabilize_restores_unit_magnitude() {
        let mut r = Resonator::new(ResonatorConfig::new(440.0, 1.0, 1.0), 44100.0);
        r.z_re = 3.0;
        r.z_im = 4.0;
        r.stabilize();
        assert!((r.z_re - 0.6).abs() < 1e-6);
        assert!((r.z_im - 0.8).abs() < 1e-6);
    }
}
