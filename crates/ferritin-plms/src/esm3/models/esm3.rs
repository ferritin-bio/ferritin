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
    /// Number of transformer layers that use geometric (affine3d) attention.
    pub n_layers_geom: usize,
    /// Number of heads for geometric attention (v_heads).
    pub v_head_transformer: usize,
    /// FFN expansion ratio.
    pub expansion_ratio: f64,
    /// Whether to scale residual connections by sqrt(n_layers / 36).
    pub scale_residue: bool,
    /// Whether to mask and zero frameless positions in geometric attention.
    pub mask_and_zero_frameless: bool,
    /// Whether to apply per-head LayerNorm to Q and K.
    pub qk_layernorm: bool,
    /// Whether to use bias in linear layers.
    pub bias: bool,
    pub d_sequence_vocab: usize,
    pub d_structure_vocab: usize,
}

impl ESM3Config {
    /// Config for `esm3-sm-open-v1` (1.4B parameters).
    pub fn sm_open() -> Self {
        let n_layers = 48usize;
        Self {
            d_model: 1536,
            n_heads: 24,
            n_layers,
            n_layers_geom: 1,
            v_head_transformer: 128,
            expansion_ratio: 8.0 / 3.0,
            scale_residue: true,
            mask_and_zero_frameless: false,
            qk_layernorm: true,
            bias: false,
            d_sequence_vocab: 64,
            d_structure_vocab: 4096,
        }
    }

    /// Residual scaling factor: sqrt(n_layers / 36) when scale_residue is true.
    pub fn residue_scaling_factor(&self) -> f64 {
        if self.scale_residue {
            (self.n_layers as f64 / 36.0).sqrt()
        } else {
            1.0
        }
    }
}

/// ESM3 multimodal generative model (stub).
pub struct ESM3 {
    pub config: ESM3Config,
}
