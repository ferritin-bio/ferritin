//! ESM3 UnifiedTransformerBlock: plain MHA + geometric attention + SwiGLU FFN.

use crate::esm3::models::esm3::ESM3Config;
use crate::esm3::utils::affine3d::Affine3D;
use crate::esmc::layers::attention::MultiHeadAttention;
use crate::esmc::layers::geom_attention::GeometricReasoningOriginalImpl;
use crate::esmc::models::esmc::{ESMCConfig, ESMTokenizer, FfnType};
use candle_core::{D, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

// ── SwiGLU FFN ───────────────────────────────────────────────────────────────

pub struct SwiGLU {
    layer_norm: nn::LayerNorm,
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl SwiGLU {
    fn hidden_dim(expansion_ratio: f64, d_model: usize) -> usize {
        ((expansion_ratio * d_model as f64 + 255.0) / 256.0).floor() as usize * 256
    }

    pub fn load(vb: VarBuilder, config: &ESM3Config) -> Result<Self> {
        let hidden = Self::hidden_dim(config.expansion_ratio, config.d_model);
        Ok(Self {
            layer_norm: nn::layer_norm(config.d_model, 1e-5, vb.pp("0"))?,
            linear1: nn::linear_no_bias(config.d_model, hidden * 2, vb.pp("1"))?,
            linear2: nn::linear_no_bias(hidden, config.d_model, vb.pp("3"))?,
        })
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let chunks = x.chunk(2, D::Minus1)?;
        self.linear2.forward(&(chunks[0].silu()? * &chunks[1])?)
    }
}

// ── ESM3 UnifiedTransformerBlock ─────────────────────────────────────────────

pub struct UnifiedTransformerBlock {
    attn: MultiHeadAttention,
    geom_attn: Option<GeometricReasoningOriginalImpl>,
    ffn: SwiGLU,
    scaling_factor: f64,
}

impl UnifiedTransformerBlock {
    pub fn load(vb: VarBuilder, config: &ESM3Config, layer_idx: usize) -> Result<Self> {
        // Build an ESMCConfig shim so we can reuse the existing load() implementations.
        let esmc_cfg = esmc_config_from_esm3(config);

        let attn = MultiHeadAttention::load(vb.pp("attn"), &esmc_cfg)?;

        let geom_attn = if layer_idx < config.n_layers_geom {
            // Checkpoints name this block "geom_attn", not "geometric"
            // (ferritin-100.21).
            Some(GeometricReasoningOriginalImpl::load(
                vb.pp("geom_attn"),
                &esmc_cfg,
            )?)
        } else {
            None
        };

        let ffn = SwiGLU::load(vb.pp("ffn"), config)?;

        Ok(Self {
            attn,
            geom_attn,
            ffn,
            scaling_factor: config.residue_scaling_factor(),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        sequence_id: Option<&Tensor>,
        affine: Option<&Affine3D>,
        affine_mask: Option<&Tensor>,
        chain_id: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = x.clone();

        // Standard multi-head attention residual
        let r1 = self.attn.forward(&x, sequence_id)?;
        x = (&x + (r1 / self.scaling_factor)?)?;

        // Geometric attention residual (only in layers where geom_attn is present)
        if let (Some(geom), Some(aff), Some(mask)) = (&self.geom_attn, affine, affine_mask) {
            let r2 = geom.forward(&x, aff, mask, sequence_id, chain_id)?;
            x = (&x + (r2 / self.scaling_factor)?)?;
        }

        // FFN residual
        let r3 = self.ffn.forward(&x)?;
        x = (&x + (r3 / self.scaling_factor)?)?;

        Ok(x)
    }
}

/// Build an `ESMCConfig` shim from `ESM3Config` so ESMC layer loaders can be reused.
fn esmc_config_from_esm3(cfg: &ESM3Config) -> ESMCConfig {
    let n_layers = cfg.n_layers;
    ESMCConfig {
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        n_layers,
        v_head_transformer: Some(cfg.v_head_transformer),
        ffn_type: FfnType::SWIGLU,
        tokenizer: ESMTokenizer::Esm3OpenSmall,
        use_plain_attn: true,
        n_layers_geom: cfg.n_layers_geom,
        scale_residue: cfg.scale_residue,
        residue_scaling_factor: cfg.residue_scaling_factor(),
        mask_and_zero_frameless: cfg.mask_and_zero_frameless,
        bias: cfg.bias,
        qk_layernorm: cfg.qk_layernorm,
        expansion_ratio: cfg.expansion_ratio,
        // Unused by the layer loaders we call
        regression_head_output_dim: 0,
        regression_head_hidden_dim: 0,
        embedding_dim: 0,
    }
}
