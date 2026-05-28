//! LM encoder adapter: projects ESMC-6B hidden states → single representation.
//!
//! Weight layout:
//! ```text
//! lm_encoder.blocks.{0..3}.attn.*   — MultiHeadAttention
//! lm_encoder.blocks.{0..3}.ffn.*    — SwiGLU FFN
//! lm_encoder.norm.weight            — final LayerNorm
//! lm_encoder.proj.weight            — linear 2560 → 384
//! ```

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// 4-layer transformer adapter that maps ESMC-6B hidden states
/// (`[B, L, 2560]`) to the ESMFold2 single representation (`[B, L, 384]`).
pub struct LMEncoder {
    // TODO: blocks: Vec<TransformerBlock>,
    // TODO: norm: LayerNorm,
    // TODO: proj: Linear,
    d_in: usize,
    d_out: usize,
    device: candle_core::Device,
}

impl LMEncoder {
    /// Load the LM encoder from a `VarBuilder` rooted at `lm_encoder.*`.
    ///
    /// # Arguments
    /// * `vb`       — builder rooted at `lm_encoder`
    /// * `d_in`     — ESMC hidden dim (2560)
    /// * `d_out`    — target single-repr dim (384)
    /// * `n_layers` — number of transformer blocks (4)
    pub fn load(vb: VarBuilder, d_in: usize, d_out: usize, n_layers: usize) -> Result<Self> {
        // TODO: load transformer blocks from vb.pp(format!("blocks.{}", i))
        // TODO: load LayerNorm from vb.pp("norm")
        // TODO: load proj from vb.pp("proj")
        let _ = n_layers; // suppress unused warning until blocks are implemented
        let device = vb.device().clone();
        Ok(Self {
            d_in,
            d_out,
            device,
        })
    }

    /// Projects ESMC-6B hidden states to the single representation.
    ///
    /// - Input:  `[B, L, d_in=2560]`
    /// - Output: `[B, L, d_out=384]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        // TODO: run transformer blocks + final LayerNorm + projection
        Tensor::zeros((b, l, self.d_out), x.dtype(), &self.device)
    }
}
