//! Folding trunk: 24-layer Pairformer transformer.
//!
//! Weight layout:
//! ```text
//! folding_trunk.pair_init.*            — pair representation initializer
//! folding_trunk.blocks.{0..23}.*       — Pairformer layers
//! folding_trunk.norm.*                 — final single-repr LayerNorm (TODO)
//! ```

use super::pair_init::PairInit;
use super::pairformer::PairformerBlock;
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// 24-layer Pairformer trunk that jointly refines single and pair representations.
///
/// - Single repr: `[B, N, d_single=384]`
/// - Pair repr:   `[B, N, N, d_pair=256]`
pub struct FoldingTrunk {
    pair_init: PairInit,
    blocks: Vec<PairformerBlock>,
    // TODO: norm: LayerNorm — final single-repr LayerNorm
}

impl FoldingTrunk {
    /// Load the folding trunk from a `VarBuilder` rooted at `folding_trunk.*`.
    ///
    /// `n_heads` comes from `ESMFold2Config::trunk_n_heads` (8 for ESMFold2-Fast).
    pub fn load(
        vb: VarBuilder,
        n_layers: usize,
        d_single: usize,
        d_pair: usize,
        n_heads: usize,
    ) -> Result<Self> {
        let pair_init = PairInit::load(
            vb.pp("pair_init"),
            d_single,
            d_pair,
            32, // n_relpos_bins — from ESMFold2Config::n_relative_residx_bins
            32, // d_outer — inner dim for outer-product projection
        )?;
        let blocks = (0..n_layers)
            .map(|i| PairformerBlock::load(vb.pp(format!("blocks.{i}")), d_pair, n_heads))
            .collect::<Result<Vec<_>>>()?;
        // TODO: load LayerNorm from vb.pp("norm")
        Ok(Self { pair_init, blocks })
    }

    /// Initialise the pair representation from sequence and chain metadata.
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

    /// Run all 24 Pairformer layers.
    ///
    /// # Arguments
    /// * `single` — `[B, N, d_single]`  (passed through; single update is TODO)
    /// * `pair`   — `[B, N, N, d_pair]` (from `init_pair`)
    ///
    /// # Returns
    /// `(single [B, N, d_single], pair [B, N, N, d_pair])`
    pub fn forward(&self, single: &Tensor, pair: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut z = pair.clone();
        for block in &self.blocks {
            z = block.forward(&z)?;
        }
        // TODO: apply final LayerNorm on single
        Ok((single.clone(), z))
    }
}
