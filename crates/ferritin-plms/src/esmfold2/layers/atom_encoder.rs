//! Atom encoder: per-atom feature encoder for ESMFold2.
//!
//! Weight layout:
//! ```text
//! inputs.atom_encoder.blocks.{0..2}.*   — 3 atom transformer blocks
//! ```

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// 3-block atom-level transformer encoder.
///
/// Maps per-atom features (`[B, N_atom, d_atom=128]`) to per-token atom
/// representations (`[B, N_tok, d_token=768]`) via windowed attention and
/// token-level pooling.
pub struct AtomEncoder {
    // TODO: blocks: Vec<AtomTransformerBlock> (3 blocks, SWA window=128)
    // TODO: proj:   Linear (d_atom → d_token)
    d_atom: usize,
    d_token: usize,
    device: candle_core::Device,
}

impl AtomEncoder {
    /// Load the atom encoder from a `VarBuilder` rooted at `inputs.atom_encoder.*`.
    ///
    /// # Arguments
    /// * `vb`       — builder rooted at `inputs.atom_encoder`
    /// * `d_atom`   — per-atom feature dimension (128)
    /// * `d_token`  — per-token output dimension (768)
    /// * `n_blocks` — number of atom transformer blocks (3)
    pub fn load(vb: VarBuilder, d_atom: usize, d_token: usize, n_blocks: usize) -> Result<Self> {
        // TODO: load n_blocks atom transformer blocks with SWA window=128
        let _ = n_blocks; // suppress unused warning until blocks are implemented
        let device = vb.device().clone();
        Ok(Self {
            d_atom,
            d_token,
            device,
        })
    }

    /// Encodes per-atom features to per-token atom representations.
    ///
    /// - Input:  `[B, N_atom, d_atom=128]`
    /// - Output: `[B, N_tok, d_token=768]`
    ///
    /// The number of output tokens depends on the residue-level grouping of atoms.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO: run atom transformer blocks + token-level mean-pooling + projection
        let _ = self.d_atom; // used only in weight loading
        let (b, _n_atom, _) = x.dims3()?;
        // Placeholder: single token per batch (real impl emits N_tok tokens)
        Tensor::zeros((b, 1_usize, self.d_token), x.dtype(), &self.device)
    }
}
