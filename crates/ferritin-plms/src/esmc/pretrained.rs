//! ESMC pretrained model loading.
//!
//! Downloads weights from `biohub/ESMC-{300M,600M,6B}` on HuggingFace and
//! wraps the ESMC model for sequence embedding.
//!
//! ## Weight key layout (`biohub/ESMC-*` safetensors)
//!
//! The weights are saved from `ESMCForMaskedLM`, a HuggingFace wrapper that
//! nests the backbone under an `esmc` attribute. All backbone keys are
//! therefore prefixed with `esmc.`:
//!
//! | Python (HF safetensors key)                              | Rust VarBuilder path                          |
//! |----------------------------------------------------------|-----------------------------------------------|
//! | `esmc.embed.weight`                                      | `embed.weight`                                |
//! | `esmc.transformer.blocks.{i}.attn.layernorm_qkv.0.*`   | `transformer.blocks.{i}.attn.layernorm_qkv.0.*` |
//! | `esmc.transformer.blocks.{i}.attn.layernorm_qkv.1.*`   | `transformer.blocks.{i}.attn.layernorm_qkv.1.*` |
//! | `esmc.transformer.blocks.{i}.attn.out_proj.weight`      | `transformer.blocks.{i}.attn.out_proj.weight`   |
//! | `esmc.transformer.blocks.{i}.attn.q_ln.weight`          | `transformer.blocks.{i}.attn.q_ln.weight`       |
//! | `esmc.transformer.blocks.{i}.attn.k_ln.weight`          | `transformer.blocks.{i}.attn.k_ln.weight`       |
//! | `esmc.transformer.blocks.{i}.attn.rotary.*`             | `transformer.blocks.{i}.attn.rotary.*`          |
//! | `esmc.transformer.blocks.{i}.ffn.0.*`                   | `transformer.blocks.{i}.ffn.0.*`                |
//! | `esmc.transformer.blocks.{i}.ffn.1.weight`              | `transformer.blocks.{i}.ffn.1.weight`           |
//! | `esmc.transformer.blocks.{i}.ffn.3.weight`              | `transformer.blocks.{i}.ffn.3.weight`           |
//! | `esmc.transformer.norm.weight`                           | `transformer.norm.weight`                       |
//! | `esmc.sequence_head.0.weight/bias`                       | `sequence_head.0.weight/bias`                   |
//! | `esmc.sequence_head.2.weight`                            | `sequence_head.2.weight`                        |
//! | `esmc.sequence_head.3.weight/bias`                       | `sequence_head.3.weight/bias`                   |
//!
//! The `from_pretrained` loader auto-detects whether the `esmc.` prefix is
//! present and sets the VarBuilder root accordingly, so the same Rust model
//! code works against both the wrapped and unwrapped formats.

use crate::esmc::models::esmc::{ESMC, ESMCConfig, LogitsConfig};
use crate::loader::{LoadOptions, WeightSource, optional_prefix};
use crate::plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout};
use anyhow::Result;
use candle_core::{Device, Tensor};

/// Available ESMC model variants hosted on HuggingFace.
pub enum ESMCModels {
    /// ESMC 300M — 30 layers, d_model=960 (~1.3 GB weights)
    ESMC300M,
    /// ESMC 600M — 36 layers, d_model=1152
    ESMC600M,
    /// ESMC 6B — 80 layers, d_model=2560 (backbone for ESMFold2)
    ESMC6B,
}

impl ESMCModels {
    /// Returns `(weight_source, config)` for this variant.
    pub fn model_info(&self) -> (WeightSource, ESMCConfig) {
        match self {
            Self::ESMC300M => (
                WeightSource::safetensors("biohub/ESMC-300M"),
                ESMCConfig::esmc_300m(),
            ),
            Self::ESMC600M => (
                WeightSource::safetensors("biohub/ESMC-600M"),
                ESMCConfig::esmc_600m(),
            ),
            Self::ESMC6B => (
                WeightSource::safetensors("biohub/ESMC-6B"),
                ESMCConfig::esmc_6b(),
            ),
        }
    }
}

/// Wraps a loaded ESMC model for sequence embedding inference.
pub struct ESMCRunner {
    model: ESMC,
    /// Retained so `PlmRunner::metadata` can report dimensions.
    config: ESMCConfig,
}

impl ESMCRunner {
    /// Download weights from HuggingFace and load the model.
    ///
    /// Handles the `ESMCForMaskedLM` wrapper prefix (`esmc.`) transparently:
    /// probes for `esmc.embed.weight` in the safetensors and sets the
    /// VarBuilder root to `vb.pp("esmc")` when found, otherwise uses the
    /// flat (unwrapped) layout.
    pub fn from_pretrained(model: ESMCModels, device: Device) -> Result<Self> {
        let (source, config) = model.model_info();
        let vb = source.var_builder("model.safetensors", &LoadOptions::new(device))?;
        // Weights saved from ESMCForMaskedLM nest the backbone under "esmc".
        let vb_root = optional_prefix(vb, "esmc", "embed.weight");
        let esmc = ESMC::load(vb_root, config.clone())?;
        Ok(Self {
            model: esmc,
            config,
        })
    }

    /// Tokenize `sequence` and run a forward pass.
    ///
    /// Returns per-residue embeddings with shape `(1, L, d_model)` where
    /// `L` includes the BOS and EOS tokens added by the tokenizer.
    pub fn embed_sequence(&self, sequence: &str) -> Result<Tensor> {
        let tokens = self.model.encode(sequence)?;
        let tokens = tokens.unsqueeze(0)?;
        let output = self.model.forward(&tokens, None, false)?;
        output
            .embeddings
            .ok_or_else(|| anyhow::anyhow!("ESMC forward() returned no embeddings"))
    }
}

impl PlmRunner for ESMCRunner {
    /// Delegate to `embed_sequence`, returning per-residue embeddings `(1, L, d_model)`.
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        self.embed_sequence(sequence)
    }

    fn model_name(&self) -> &str {
        "esmc"
    }

    /// `ESMC::encode` calls `tokenize_sequence(.., true)`, which wraps the
    /// sequence in BOS and EOS.
    fn special_tokens(&self) -> SpecialTokenLayout {
        SpecialTokenLayout::BOS_EOS
    }

    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            d_model: self.config.d_model,
            n_layers: self.config.n_layers,
            vocab_size: self.config.embedding_dim,
            // Rotary positions: no hard architectural cap.
            max_positions: None,
        }
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    /// Sequence logits `(1, L + 2, vocab_size)`.
    fn logits(&self, sequence: &str) -> Result<Tensor> {
        let tokens = self.model.encode(sequence)?.unsqueeze(0)?;
        let out = self.model.logits(
            &tokens,
            LogitsConfig {
                sequence: true,
                return_embeddings: false,
                return_hidden_states: false,
            },
        )?;
        out.sequence_logits
            .ok_or_else(|| anyhow::anyhow!("ESMC logits() returned no sequence logits"))
    }
}
