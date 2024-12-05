use crate::esm::layers::blocks::UnifiedTransformerBlock;
use crate::esm::models::esmc::ESMCConfig;
use crate::esm::utils::structure::affine3d::Affine3D;
use candle_core::{Module, Result, Tensor, D};
use candle_nn::{self as nn, LayerNorm, LayerNormConfig};

pub struct TransformerStack {
    /*
    A stack of transformer blocks used in the ESM-3 model. Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (i64): The dimensionality of the input and output feature vectors.
        n_heads (i64): The number of attention heads.
        v_heads (Option<i64>): The number of voting heads.
        n_layers (i64): The number of transformer blocks in the stack.
        n_layers_geom (i64, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    */
    blocks: Vec<UnifiedTransformerBlock>,
    norm: LayerNorm,
}

impl TransformerStack {
    pub fn load(vb: nn::VarBuilder, config: ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            n_heads,
            n_layers,
            ffn_type,
            v_head_transformer,
            use_plain_attn,
            n_layers_geom,
            scale_residue,
            residue_scaling_factor,
            mask_and_zero_frameless,
            bias,
            qk_layernorm,
            expansion_ratio,
            tokenizer,
        } = config;
        let mut blocks = Vec::with_capacity(n_layers as usize);
        for i in 0..n_layers {
            blocks.push(UnifiedTransformerBlock::load(vb.pp("layer"), config), i);
        }

        let ln_conf = LayerNormConfig::from(1e-5);
        let norm = nn::layer_norm(d_model, ln_conf, vb.pp("layer_norm"))?;

        Ok(Self { blocks, norm })
    }
    // pub fn new(
    //     d_model: i64,
    //     n_heads: i64,
    //     v_heads: Option<i64>,
    //     n_layers: i64,
    //     n_layers_geom: i64,
    //     scale_residue: bool,
    //     mask_and_zero_frameless: bool,
    //     bias: bool,
    //     qk_layernorm: bool,
    //     ffn_type: &str,
    //     expansion_ratio: f64,
    // ) -> Result<Self> {
    //     let mut blocks = Vec::with_capacity(n_layers as usize);
    //     for i in 0..n_layers {
    //         blocks.push(UnifiedTransformerBlock::new(
    //             d_model,
    //             n_heads,
    //             v_heads,
    //             i < n_layers_geom,
    //             if scale_residue {
    //                 (n_layers as f64 / 36.0).sqrt()
    //             } else {
    //                 1.0
    //             },
    //             expansion_ratio,
    //             mask_and_zero_frameless,
    //             bias,
    //             qk_layernorm,
    //             ffn_type,
    //         )?);
    //     }

    //     let norm = nn::LayerNorm::new(d_model, 1e-5, false)?;

    //     Ok(Self { blocks, norm })
    // }

    // pub fn forward(
    //     &self,
    //     x: &Tensor,
    //     sequence_id: Option<&Tensor>,
    //     affine: Option<&Affine3D>,
    //     affine_mask: Option<&Tensor>,
    //     chain_id: Option<&Tensor>,
    // ) -> Result<(Tensor, Tensor)> {
    //     let mut x = x.clone();

    //     let chain_id = if chain_id.is_none() {
    //         let batch_dims = x.shape().split_last().unwrap().1;
    //         Tensor::ones(batch_dims, (x.device(), DType::I64))?
    //     } else {
    //         chain_id.unwrap().clone()
    //     };

    //     for block in self.blocks.iter() {
    //         x = block.forward(&x, sequence_id, affine, affine_mask, &chain_id)?;
    //     }

    //     let normalized = self.norm.forward(&x)?;
    //     Ok((normalized, x))
    // }
}
