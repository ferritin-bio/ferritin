//! ESM3 TransformerStack: stacked UnifiedTransformerBlocks with geometric attention.

use crate::esm3::layers::blocks::UnifiedTransformerBlock;
use crate::esm3::models::esm3::ESM3Config;
use crate::esm3::utils::affine3d::Affine3D;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

pub struct TransformerStack {
    blocks: Vec<UnifiedTransformerBlock>,
    norm: nn::LayerNorm,
}

impl TransformerStack {
    pub fn load(vb: VarBuilder, config: &ESM3Config) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            blocks.push(UnifiedTransformerBlock::load(
                vb.pp(format!("blocks.{}", i)),
                config,
                i,
            )?);
        }

        // Final LayerNorm: weight only, no bias.
        let norm_weight = vb.pp("norm").get((config.d_model,), "weight")?;
        let norm = nn::LayerNorm::new_no_bias(norm_weight, 1e-5);

        Ok(Self { blocks, norm })
    }

    /// Forward pass through the full ESM3 transformer stack.
    ///
    /// - `x`:           `(B, L, d_model)` input embeddings.
    /// - `sequence_id`: optional `(B, L)` int — per-protein bin-packing IDs.
    /// - `affine`:      optional per-residue local frames for geometric attention.
    /// - `affine_mask`: optional `(B, L)` u8 — 1 where frame is valid.
    /// - `chain_id`:    optional `(B, L)` int — chain identity per residue.
    ///
    /// Returns `(post_norm, pre_norm)` matching the Python API.
    pub fn forward(
        &self,
        x: &Tensor,
        sequence_id: Option<&Tensor>,
        affine: Option<&Affine3D>,
        affine_mask: Option<&Tensor>,
        chain_id: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let mut x = x.clone();

        for block in &self.blocks {
            x = block.forward(&x, sequence_id, affine, affine_mask, chain_id)?;
        }

        let post_norm = self.norm.forward(&x)?;
        Ok((post_norm, x))
    }
}
