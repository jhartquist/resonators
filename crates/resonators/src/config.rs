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
