//! Folding trunk: 24-layer Pairformer transformer.
//!
//! Weight layout:
//! ```text
//! folding_trunk.pair_init.*            — pair representation initializer
//! folding_trunk.blocks.{0..23}.*       — Pairformer layers (TODO)
//! folding_trunk.norm.*                 — final single-repr LayerNorm (TODO)
//! ```

use super::pair_init::PairInit;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// 24-layer Pairformer trunk that jointly refines single and pair representations.
///
/// - Single repr: `[B, N, d_single=384]`
/// - Pair repr:   `[B, N, N, d_pair=256]`
pub struct FoldingTrunk {
    pair_init: PairInit,
    // TODO: blocks: Vec<PairformerBlock>,
    // TODO: norm: LayerNorm,
    n_layers: usize,
}

impl FoldingTrunk {
    /// Load the folding trunk from a `VarBuilder` rooted at `folding_trunk.*`.
    pub fn load(vb: VarBuilder, n_layers: usize, d_single: usize, d_pair: usize) -> Result<Self> {
        let pair_init = PairInit::load(
            vb.pp("pair_init"),
            d_single,
            d_pair,
            32,  // n_relpos_bins — from ESMFold2Config::n_relative_residx_bins
            32,  // d_outer — inner dim for outer-product projection
        )?;
        // TODO: load Pairformer blocks from vb.pp(format!("blocks.{}", i))
        // TODO: load LayerNorm from vb.pp("norm")
        Ok(Self { pair_init, n_layers })
    }

    /// Initialises the pair representation from sequence and chain metadata.
    ///
    /// Call this once before `forward` to build the starting pair tensor.
    ///
    /// # Arguments
    /// * `single`          — `[B, N, d_single]`
    /// * `residue_indices` — `[B, N]` integer residue positions
    /// * `chain_ids`       — `[B, N]` integer chain identifiers
    ///
    /// # Returns
    /// `[B, N, N, d_pair]`
    pub fn init_pair(
        &self,
        single: &Tensor,
        residue_indices: &Tensor,
        chain_ids: &Tensor,
    ) -> Result<Tensor> {
        self.pair_init.forward(single, residue_indices, chain_ids)
    }

    /// Runs all Pairformer layers.
    ///
    /// # Arguments
    /// * `single` — `[B, N, d_single]`
    /// * `pair`   — `[B, N, N, d_pair]` (from `init_pair`)
    ///
    /// # Returns
    /// `(single [B, N, d_single], pair [B, N, N, d_pair])`
    pub fn forward(&self, single: &Tensor, pair: &Tensor) -> Result<(Tensor, Tensor)> {
        // TODO: run n_layers Pairformer blocks + final LayerNorm on single
        let _ = self.n_layers;
        Ok((single.clone(), pair.clone()))
    }
}
