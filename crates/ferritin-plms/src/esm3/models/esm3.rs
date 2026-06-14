//! ESM3 model stub.
//!
//! Full port tracked by ferritin-iln (EncodeInputs + OutputHeads)
//! and ferritin-zcf (TransformerStack).

/// Configuration for the ESM3 open model (esm3-sm-open-v1).
#[derive(Debug, Clone)]
pub struct ESM3Config {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_sequence_vocab: usize,
    pub d_structure_vocab: usize,
}

impl ESM3Config {
    /// Config for `esm3-sm-open-v1` (1.4B parameters).
    pub fn sm_open() -> Self {
        Self {
            d_model: 1536,
            n_heads: 24,
            n_layers: 48,
            d_sequence_vocab: 64,
            d_structure_vocab: 4096,
        }
    }
}

/// ESM3 multimodal generative model (stub).
pub struct ESM3 {
    pub config: ESM3Config,
}
