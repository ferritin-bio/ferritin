use super::attention::MultiHeadAttention;
use super::geom_attention::GeometricReasoningOriginalImpl;
use crate::esm::models::esmc::{ESMCConfig, Ffn_Type};
use crate::esm::utils::structure::affine3d::Affine3D;
use candle_core::{Module, Result, Tensor, D};
use candle_nn::ops::silu;
use candle_nn::{self as nn, VarBuilder};

pub struct SwiGLU {
    layer_norm: nn::LayerNorm,
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl SwiGLU {
    fn swiglu_correction_fn(expansion_ratio: f64, d_model: usize) -> usize {
        // set hidden dimension to nearest multiple of 256 after expansion ratio
        ((expansion_ratio * d_model as f64 + 255.0) / 256.0).floor() as usize * 256
    }

    pub fn load(vb: VarBuilder, config: ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            expansion_ratio,
            bias,
            ..
        } = config;

        let hidden_dim = Self::swiglu_correction_fn(expansion_ratio, d_model);

        Ok(Self {
            layer_norm: nn::layer_norm(d_model, 1e-5, vb.pp("layer_norm"))?,
            linear1: nn::linear(d_model, hidden_dim * 2, vb.pp("linear1"))?,
            linear2: nn::linear(hidden_dim, d_model, b.pp("linear2"))?,
        })
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let chunks = x.chunk(2, D::Minus1)?;
        let x1 = &chunks[0];
        let x2 = &chunks[1];
        let x = x1.silu()? * x2;
        self.linear2.forward(&x?)
    }
}

pub struct UnifiedTransformerBlock {
    use_plain_attn: bool,
    attn: Option<MultiHeadAttention>,
    use_geom_attn: bool,
    geom_attn: Option<GeometricReasoningOriginalImpl>,
    ffn: nn::Sequential,
    scaling_factor: f64,
}

impl UnifiedTransformerBlock {
    /// Creates a new UnifiedTransformerBlock.
    ///
    /// # Parameters
    /// - d_model: The dimensionality of the input and output features
    /// - n_heads: The number of attention heads
    /// - use_geom_attn: Whether to use geometric attention
    /// - use_plain_attn: Whether to use plain attention
    /// - v_heads: Number of heads for geometric attention
    // pub fn new(
    //     d_model: i64,
    //     n_heads: i64,
    //     use_geom_attn: bool,
    //     use_plain_attn: bool,
    //     v_heads: Option<i64>,
    //     bias: bool,
    //     expansion_ratio: f64,
    //     residue_scaling_factor: f64,
    //     mask_and_zero_frameless: bool,
    //     qk_layernorm: bool,
    //     ffn_type: &str,
    // ) -> Result<Self> {
    //     let attn = if use_plain_attn {
    //         Some(MultiHeadAttention::new(
    //             d_model,
    //             n_heads,
    //             bias,
    //             qk_layernorm,
    //         )?)
    //     } else {
    //         None
    //     };

    //     let geom_attn = if use_geom_attn {
    //         match v_heads {
    //             Some(vh) => Some(GeometricReasoningOriginalImpl::new(
    //                 d_model,
    //                 vh,
    //                 bias,
    //                 mask_and_zero_frameless,
    //             )?),
    //             None => {
    //                 return Err(candle_core::Error::Msg(
    //                     "v_heads must be specified when use_geom_attn is True".into(),
    //                 ))
    //             }
    //         }
    //     } else {
    //         None
    //     };

    //     let ffn = match ffn_type {
    //         "swiglu" => swiglu_ln_ffn(d_model, expansion_ratio, bias)?,
    //         "gelu" => gelu_ln_ffn(d_model, expansion_ratio, bias)?,
    //         _ => {
    //             return Err(candle_core::Error::Msg(format!(
    //                 "Unknown ffn_type: {}",
    //                 ffn_type
    //             )))
    //         }
    //     };

    //     Ok(Self {
    //         use_plain_attn,
    //         attn,
    //         use_geom_attn,
    //         geom_attn,
    //         ffn,
    //         scaling_factor: residue_scaling_factor,
    //     })
    // }
    pub fn load(vb: VarBuilder, config: ESMCConfig, layer: usize) -> Self {
        // d_model: i64,
        // n_heads: i64,
        // use_geom_attn: bool,
        // use_plain_attn: bool,
        // v_heads: Option<i64>,
        // bias: bool,
        // expansion_ratio: f64,
        // residue_scaling_factor: f64,
        // mask_and_zero_frameless: bool,
        // qk_layernorm: bool,
        // ffn_type: &str,
        let ESMCConfig {
            d_model,
            n_heads,
            n_layers,
            v_head_transformer,
            ffn_type,
            tokenizer,
            use_plain_attn,
            n_layers_geom,
            scale_residue,
            residue_scaling_factor,
            mask_and_zero_frameless,
            bias,
            qk_layernorm,
            expansion_ratio,
        } = config;

        let use_geom_attn: bool = layer < n_layers_geom;

        let attn = match use_plain_attn {
            false => None,
            true => Some(GeometricReasoningOriginalImpl::load(vb, config)),
        };

        let geom_attn = match use_geom_attn {
            false => None,
            true => Some(GeometricReasoningOriginalImpl::load(vb, config)),
        };

        let ffn = match ffn_type {
            Ffn_Type::GLU => unimplemented!(),
            Ffn_Type::SWIGLU => Swiglu::load(vb, config),
        };

        Self {
            use_plain_attn,
            attn,
            use_geom_attn,
            geom_attn,
            ffn,
            scaling_factor: residue_scaling_factor,
        }
    }
}

// impl Module for UnifiedTransformerBlock {
//     fn forward(&self, x: &Tensor) -> Result<Tensor> {
//         let mut x = x.clone();

//         if self.use_plain_attn {
//             if let Some(attn) = &self.attn {
//                 let r1 = attn.forward(&x)?;
//                 x = &x + &(&r1 / self.scaling_factor)?;
//             }
//         }

//         if self.use_geom_attn {
//             if let Some(geom_attn) = &self.geom_attn {
//                 let r2 = geom_attn.forward(&x)?;
//                 x = &x + &(&r2 / self.scaling_factor)?;
//             }
//         }

//         let r3 = self.ffn.forward(&x)?;
//         let r3 = &r3 / self.scaling_factor;
//         Ok(&x + &r3)
//     }
// }
