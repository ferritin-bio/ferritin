//! Confidence head: predicts pLDDT, pAE, pDE, and distogram.
//!
//! Weight layout:
//! ```text
//! confidence_head.trunk.blocks.{0..3}.*   — 4-layer Pairformer
//! confidence_head.plddt_head.*            — linear → 50 bins
//! confidence_head.pae_head.*              — linear → 64 bins
//! confidence_head.pde_head.*              — linear → 64 bins
//! confidence_head.distogram_head.*        — linear → 39 bins
//! ```

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Output of the confidence head.
pub struct ConfidenceOutput {
    /// pLDDT logits `[B, N_tok, 50]`.
    ///
    /// Apply softmax then take a weighted dot with bin centres to get a
    /// per-token scalar confidence in [0, 1].
    pub plddt_logits: Tensor,

    /// pAE logits `[B, N_tok, N_tok, 64]`. `None` when not requested.
    pub pae_logits: Option<Tensor>,
}

/// 4-layer Pairformer confidence head.
pub struct ConfidenceHead {
    // TODO: trunk:          PairformerTrunk (4 layers)
    // TODO: plddt_head:     Linear (d_single → 50)
    // TODO: pae_head:       Linear (d_pair   → 64)
    // TODO: pde_head:       Linear (d_pair   → 64)
    // TODO: distogram_head: Linear (d_pair   → 39)
    num_plddt_bins: usize, // 50
    num_pae_bins: usize,   // 64
    device: candle_core::Device,
}

impl ConfidenceHead {
    /// Load the confidence head from a `VarBuilder` rooted at `confidence_head.*`.
    ///
    /// # Arguments
    /// * `vb`             — builder rooted at `confidence_head`
    /// * `num_plddt_bins` — number of pLDDT bins (50)
    /// * `num_pae_bins`   — number of pAE bins (64)
    pub fn load(vb: VarBuilder, num_plddt_bins: usize, num_pae_bins: usize) -> Result<Self> {
        // TODO: load Pairformer trunk from vb.pp("trunk")
        // TODO: load plddt_head      from vb.pp("plddt_head")
        // TODO: load pae_head        from vb.pp("pae_head")
        // TODO: load pde_head        from vb.pp("pde_head")
        // TODO: load distogram_head  from vb.pp("distogram_head")
        let device = vb.device().clone();
        Ok(Self {
            num_plddt_bins,
            num_pae_bins,
            device,
        })
    }

    /// Compute confidence predictions from trunk representations.
    ///
    /// - `single` — `[B, N_tok, d_single]`
    /// - `pair`   — `[B, N_tok, N_tok, d_pair]`
    pub fn forward(&self, single: &Tensor, pair: &Tensor) -> Result<ConfidenceOutput> {
        // TODO: run 4-layer Pairformer trunk, then per-head linear projections
        let _ = pair; // suppress unused warning until trunk is implemented
        let (b, n, _) = single.dims3()?;
        Ok(ConfidenceOutput {
            plddt_logits: Tensor::zeros((b, n, self.num_plddt_bins), single.dtype(), &self.device)?,
            pae_logits: None,
        })
    }
}
