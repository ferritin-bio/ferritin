use super::attention::MultiHeadAttention;
use super::geom_attention::GeometricReasoningOriginalImpl;
use crate::esm::utils::structure::affine3d::Affine3D;
use candle_core::{Module, Result, Tensor};
use candle_nn as nn;
use candle_nn::ops::silu;

fn swiglu_correction_fn(expansion_ratio: f64, d_model: i64) -> i64 {
    // set hidden dimension to nearest multiple of 256 after expansion ratio
    ((expansion_ratio * d_model as f64 + 255.0) / 256.0).floor() as i64 * 256
}

pub struct SwiGLU {}

impl SwiGLU {
    pub fn new() -> Self {
        Self {}
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (x1, x2) = x.chunk(2, -1)?;
        Ok(&silu(&x1)? * &x2)
    }
}

fn swiglu_ln_ffn(d_model: i64, expansion_ratio: f64, bias: bool) -> Result<nn::Sequential> {
    let hidden_dim = swiglu_correction_fn(expansion_ratio, d_model);
    let seq = nn::seq()
        .add(nn::LayerNorm::new(d_model)?)
        .add(nn::linear(d_model, hidden_dim * 2, bias)?)
        .add(SwiGLU::new())
        .add(nn::linear(hidden_dim, d_model, bias)?);
    Ok(seq)
}

fn gelu_ln_ffn(d_model: i64, expansion_ratio: f64, bias: bool) -> Result<nn::Sequential> {
    let hidden_dim = (expansion_ratio * d_model as f64) as i64;
    let seq = nn::seq()
        .add(nn::LayerNorm::new(d_model)?)
        .add(nn::linear(d_model, hidden_dim, bias)?)
        .add_fn(|x| x.gelu())
        .add(nn::linear(hidden_dim, d_model, bias)?);
    Ok(seq)
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
    pub fn new(
        d_model: i64,
        n_heads: i64,
        use_geom_attn: bool,
        use_plain_attn: bool,
        v_heads: Option<i64>,
        bias: bool,
        expansion_ratio: f64,
        residue_scaling_factor: f64,
        mask_and_zero_frameless: bool,
        qk_layernorm: bool,
        ffn_type: &str,
    ) -> Result<Self> {
        let attn = if use_plain_attn {
            Some(MultiHeadAttention::new(
                d_model,
                n_heads,
                bias,
                qk_layernorm,
            )?)
        } else {
            None
        };

        let geom_attn = if use_geom_attn {
            match v_heads {
                Some(vh) => Some(GeometricReasoningOriginalImpl::new(
                    d_model,
                    vh,
                    bias,
                    mask_and_zero_frameless,
                )?),
                None => {
                    return Err(candle_core::Error::Msg(
                        "v_heads must be specified when use_geom_attn is True".into(),
                    ))
                }
            }
        } else {
            None
        };

        let ffn = match ffn_type {
            "swiglu" => swiglu_ln_ffn(d_model, expansion_ratio, bias)?,
            "gelu" => gelu_ln_ffn(d_model, expansion_ratio, bias)?,
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Unknown ffn_type: {}",
                    ffn_type
                )))
            }
        };

        Ok(Self {
            use_plain_attn,
            attn,
            use_geom_attn,
            geom_attn,
            ffn,
            scaling_factor: residue_scaling_factor,
        })
    }
}

impl Module for UnifiedTransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();

        if self.use_plain_attn {
            if let Some(attn) = &self.attn {
                let r1 = attn.forward(&x)?;
                x = &x + &(&r1 / self.scaling_factor)?;
            }
        }

        if self.use_geom_attn {
            if let Some(geom_attn) = &self.geom_attn {
                let r2 = geom_attn.forward(&x)?;
                x = &x + &(&r2 / self.scaling_factor)?;
            }
        }

        let r3 = self.ffn.forward(&x)?;
        let r3 = &r3 / self.scaling_factor;
        Ok(&x + &r3)
    }
}
