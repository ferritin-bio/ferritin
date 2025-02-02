use super::attention::MultiHeadAttention;
use super::geom_attention::GeometricReasoningOriginalImpl;
use crate::esm::models::esmc::{ESMCConfig, FfnType};
// use crate::esm::utils::structure::affine3d::Affine3D;
use candle_core::{Module, Result, Tensor, D};
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

    pub fn load(vb: VarBuilder, config: &ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            expansion_ratio,
            ..
        } = config;

        let hidden_dim = Self::swiglu_correction_fn(*expansion_ratio, *d_model);

        Ok(Self {
            layer_norm: nn::layer_norm(*d_model, 1e-5, vb.pp("0"))?,
            linear1: nn::linear_no_bias(*d_model, hidden_dim * 2, vb.pp("1"))?,
            linear2: nn::linear_no_bias(hidden_dim, *d_model, vb.pp("3"))?,
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
    ffn: SwiGLU,
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
    pub fn load(vb: VarBuilder, config: &ESMCConfig, layer: usize) -> Result<Self> {
        let ESMCConfig {
            ffn_type,
            v_head_transformer,
            use_plain_attn,
            n_layers_geom,
            residue_scaling_factor,
            ..
        } = config;

        let attn = match use_plain_attn {
            false => None,
            true => Some(MultiHeadAttention::load(vb.pp("attn"), config)?),
        };

        // println!("LAYER; GEOM: {}, {}", layer, n_layers_geom);
        let use_geom_attn: bool = layer < *n_layers_geom;
        // println!("Geom ATTN {}", use_geom_attn);
        // let geom_attn = match use_geom_attn {
        //     false => None,
        //     true => Some(GeometricReasoningOriginalImpl::load(
        //         vb.pp("geometric"),
        //         config,
        //     )?),
        // };

        let geom_attn = None;

        let ffn = match ffn_type {
            FfnType::SWIGLU => SwiGLU::load(vb.pp("ffn"), config)?,
            _ => unimplemented!(), // FfnType::GLU => unimplemented!(),
        };

        Ok(Self {
            use_plain_attn: *use_plain_attn,
            attn,
            use_geom_attn,
            geom_attn,
            ffn,
            scaling_factor: *residue_scaling_factor,
        })
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
