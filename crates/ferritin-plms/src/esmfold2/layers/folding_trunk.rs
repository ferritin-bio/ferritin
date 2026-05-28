//! Folding trunk: 24-layer Evoformer/Pairformer-style transformer.
//!
//! Weight layout:
//! ```text
//! folding_trunk.blocks.{0..23}.*   — Pairformer layers
//! folding_trunk.norm.*             — final single-repr LayerNorm
//! ```

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// 24-layer Pairformer trunk that jointly refines single and pair representations.
///
/// - Single repr: `[B, N, d_single=384]`
/// - Pair repr:   `[B, N, N, d_pair=256]`
pub struct FoldingTrunk {
    // TODO: blocks: Vec<PairformerBlock>,
    // TODO: norm: LayerNorm,
    n_layers: usize,
    d_single: usize,
    d_pair: usize,
    device: candle_core::Device,
}

impl FoldingTrunk {
    /// Load the folding trunk from a `VarBuilder` rooted at `folding_trunk.*`.
    ///
    /// # Arguments
    /// * `vb`       — builder rooted at `folding_trunk`
    /// * `n_layers` — number of Pairformer blocks (24)
    /// * `d_single` — single representation dimension (384)
    /// * `d_pair`   — pair representation dimension (256)
    pub fn load(vb: VarBuilder, n_layers: usize, d_single: usize, d_pair: usize) -> Result<Self> {
        // TODO: load Pairformer blocks from vb.pp(format!("blocks.{}", i))
        // TODO: load LayerNorm from vb.pp("norm")
        let device = vb.device().clone();
        Ok(Self {
            n_layers,
            d_single,
            d_pair,
            device,
        })
    }

    /// Runs all Pairformer layers.
    ///
    /// # Returns
    /// `(single_repr [B, N, d_single], pair_repr [B, N, N, d_pair])`
    pub fn forward(&self, single: &Tensor, pair: &Tensor) -> Result<(Tensor, Tensor)> {
        // TODO: run n_layers Pairformer blocks + final LayerNorm on single
        Ok((single.clone(), pair.clone()))
    }
}
