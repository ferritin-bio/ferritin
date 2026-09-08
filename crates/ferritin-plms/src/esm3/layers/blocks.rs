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

    /// Load the FFN, taking the bias terms **from the checkpoint** rather than
    /// assuming they are absent (ferritin-100.27).
    ///
    /// The two checkpoints that share this block disagree, and hardcoding
    /// either answer is silently wrong for the other:
    ///
    /// | checkpoint | `ffn.1.bias` / `ffn.3.bias` |
    /// |---|---|
    /// | `esm3_sm_open_v1.pth` (the sequence model) | absent |
    /// | `esm3_structure_encoder_v0.pth` (the VQ-VAE encoder) | **present** |
    ///
    /// This used to be `linear_no_bias` unconditionally, which is correct for
    /// the sequence model and drops two real tensors per block for the
    /// structure encoder. Nothing caught it: the loader only ever asked for the
    /// weights, so "every tensor resolves" stayed true while the biases sat
    /// unread in the file. The cost was ~4% drift in the encoder's latent —
    /// invisible to any self-consistency check, but enough to move 11 of 93
    /// residues onto a different codebook entry.
    ///
    /// Probed rather than made a config flag because the checkpoint is the
    /// authority here, and the same probing idiom already resolves ESM-C's
    /// wrapper prefix in `loader::optional_prefix`.
    pub fn load(vb: VarBuilder, config: &ESM3Config) -> Result<Self> {
        let hidden = Self::hidden_dim(config.expansion_ratio, config.d_model);
        let linear = |d_in, d_out, vb: VarBuilder| {
            if vb.contains_tensor("bias") {
                nn::linear(d_in, d_out, vb)
            } else {
                nn::linear_no_bias(d_in, d_out, vb)
            }
        };
        Ok(Self {
            layer_norm: nn::layer_norm(config.d_model, 1e-5, vb.pp("0"))?,
            linear1: linear(config.d_model, hidden * 2, vb.pp("1"))?,
            linear2: linear(hidden, config.d_model, vb.pp("3"))?,
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
    /// Absent in the structure encoder, whose blocks are geometric-only.
    attn: Option<MultiHeadAttention>,
    geom_attn: Option<GeometricReasoningOriginalImpl>,
    ffn: SwiGLU,
    scaling_factor: f64,
}

impl UnifiedTransformerBlock {
    pub fn load(vb: VarBuilder, config: &ESM3Config, layer_idx: usize) -> Result<Self> {
        // Build an ESMCConfig shim so we can reuse the existing load() implementations.
        let esmc_cfg = esmc_config_from_esm3(config);

        let attn = Some(MultiHeadAttention::load(vb.pp("attn"), &esmc_cfg)?);

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

    /// A geometric-only block, as used by the ESM3 structure encoder.
    ///
    /// The released `esm3_structure_encoder_v0.pth` has no `attn.*` tensors at
    /// all — its two blocks are `geom_attn` plus `ffn`. Loading it through
    /// [`load`][Self::load] fails on `attn.layernorm_qkv.0.weight`, which is
    /// how this difference was found (ferritin-100.22).
    pub fn load_geometric(vb: VarBuilder, config: &ESM3Config) -> Result<Self> {
        let esmc_cfg = esmc_config_from_esm3(config);
        let geom_attn = Some(GeometricReasoningOriginalImpl::load(
            vb.pp("geom_attn"),
            &esmc_cfg,
        )?);
        let ffn = SwiGLU::load(vb.pp("ffn"), config)?;
        Ok(Self {
            attn: None,
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

        // Standard multi-head attention residual, when the block has one.
        if let Some(attn) = &self.attn {
            let r1 = attn.forward(&x, sequence_id)?;
            x = (&x + (r1 / self.scaling_factor)?)?;
        }

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    fn cfg(d_model: usize) -> ESM3Config {
        ESM3Config {
            d_model,
            expansion_ratio: 8.0 / 3.0,
            ..ESM3Config::sm_open()
        }
    }

    /// Build the tensors an FFN needs, optionally including the two Linear
    /// biases the structure encoder ships and the sequence model does not.
    fn ffn_tensors(d_model: usize, with_bias: bool, device: &Device) -> HashMap<String, Tensor> {
        let hidden = SwiGLU::hidden_dim(8.0 / 3.0, d_model);
        let mut ts = HashMap::new();
        let ones = |dims: (usize, usize)| Tensor::ones(dims, DType::F32, device).unwrap();
        ts.insert(
            "0.weight".into(),
            Tensor::ones(d_model, DType::F32, device).unwrap(),
        );
        ts.insert(
            "0.bias".into(),
            Tensor::zeros(d_model, DType::F32, device).unwrap(),
        );
        ts.insert("1.weight".into(), ones((hidden * 2, d_model)));
        ts.insert("3.weight".into(), ones((d_model, hidden)));
        if with_bias {
            ts.insert(
                "1.bias".into(),
                Tensor::ones(hidden * 2, DType::F32, device).unwrap(),
            );
            ts.insert(
                "3.bias".into(),
                Tensor::ones(d_model, DType::F32, device).unwrap(),
            );
        }
        ts
    }

    /// The two checkpoints sharing this block disagree about the FFN biases:
    /// `esm3_sm_open_v1.pth` has none, `esm3_structure_encoder_v0.pth` has both.
    /// Loading must follow the checkpoint, not a hardcoded choice
    /// (ferritin-100.27).
    ///
    /// This is the regression guard for a bug that cost ~4% drift in the
    /// structure encoder's latent and moved 11 of 93 residues onto a different
    /// codebook entry — while every self-consistency check stayed green,
    /// because the loader simply never asked for the tensors it was dropping.
    #[test]
    fn test_ffn_bias_follows_the_checkpoint() -> Result<()> {
        let device = Device::Cpu;
        let d_model = 16usize;

        for with_bias in [false, true] {
            let vb = VarBuilder::from_tensors(
                ffn_tensors(d_model, with_bias, &device),
                DType::F32,
                &device,
            );
            let ffn = SwiGLU::load(vb, &cfg(d_model))?;
            assert_eq!(
                ffn.linear1.bias().is_some(),
                with_bias,
                "linear1 bias presence should follow the checkpoint (with_bias = {with_bias})"
            );
            assert_eq!(
                ffn.linear2.bias().is_some(),
                with_bias,
                "linear2 bias presence should follow the checkpoint (with_bias = {with_bias})"
            );
        }
        Ok(())
    }

    /// A checkpoint carrying biases must actually *use* them: the old
    /// `linear_no_bias` loaded without error and silently ignored them, so
    /// "it loads" was never evidence that it was right.
    #[test]
    fn test_ffn_biases_change_the_output() -> Result<()> {
        let device = Device::Cpu;
        let d_model = 16usize;
        let x = Tensor::ones((1, 4, d_model), DType::F32, &device)?;

        let plain = SwiGLU::load(
            VarBuilder::from_tensors(ffn_tensors(d_model, false, &device), DType::F32, &device),
            &cfg(d_model),
        )?
        .forward(&x)?;
        let biased = SwiGLU::load(
            VarBuilder::from_tensors(ffn_tensors(d_model, true, &device), DType::F32, &device),
            &cfg(d_model),
        )?
        .forward(&x)?;

        let a: Vec<f32> = plain.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = biased.flatten_all()?.to_vec1()?;
        let diff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            diff > 1e-3,
            "the biases should change the output; max diff {diff}"
        );
        Ok(())
    }
}
