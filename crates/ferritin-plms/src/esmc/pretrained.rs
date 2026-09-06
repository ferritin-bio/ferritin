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
use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

/// Why [`ESMCModels::ESMC6B`] cannot be loaded yet.
///
/// Its checkpoint uses the same unfused layout as 300M/600M, so this is
/// plumbing rather than an architecture mismatch — but two pieces are missing.
pub const ESMC6B_UNSUPPORTED: &str = "\
ESMCModels::ESMC6B is not yet supported. EvolutionaryScale/esmc-6b-2024-12 splits its 808 \
tensors across six safetensors shards (model-0000{1..6}-of-00006.safetensors with a \
model.safetensors.index.json), and WeightSource::var_builder loads a single file; it also \
names its output head 'lm_head' at the top level rather than 'sequence_head' under the 'esmc' \
prefix this port descends into. Refusing until both are handled, rather than loading part of \
the model (ferritin-100.24). ESMC300M and ESMC600M work.";

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
    /// Returns `(weight_source, filename, config)` for this variant.
    ///
    /// These point at the **EvolutionaryScale** originals, not the
    /// `biohub/ESMC-*` re-exports the loader used to request. Those re-exports
    /// are TransformerEngine-FUSED — `attn.layernorm_qkv.weight` as one fused
    /// tensor with separate `layer_norm_weight`/`layer_norm_bias`,
    /// `ffn.fc1_weight`/`fc2_weight`, an `lm_head` instead of
    /// `sequence_head`, plus 90 empty `_extra_state` tensors — so nothing this
    /// port asks for resolved (ferritin-100.23).
    ///
    /// `ESMC6B` is not yet supported: see [`ESMC6B_UNSUPPORTED`].
    pub fn model_info(&self) -> Result<(WeightSource, &'static str, ESMCConfig)> {
        match self {
            Self::ESMC300M => Ok((
                WeightSource::pth("EvolutionaryScale/esmc-300m-2024-12", None),
                "data/weights/esmc_300m_2024_12_v0.pth",
                ESMCConfig::esmc_300m(),
            )),
            Self::ESMC600M => Ok((
                WeightSource::pth("EvolutionaryScale/esmc-600m-2024-12", None),
                "data/weights/esmc_600m_2024_12_v0.pth",
                ESMCConfig::esmc_600m(),
            )),
            Self::ESMC6B => bail!("{ESMC6B_UNSUPPORTED}"),
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
        Self::from_pretrained_with(model, &LoadOptions::new(device))
    }

    /// Load with an explicit device and dtype.
    ///
    /// `ESMC6B` is ~24 GB at F32 but ~12 GB at BF16 — its on-disk precision —
    /// so reduced precision is the difference between loadable and not on a
    /// 32 GB machine. See `tests/test_plm_dtype_parity.rs` for the numerical
    /// cost (ferritin-100.9).
    pub fn from_pretrained_with(model: ESMCModels, opts: &LoadOptions) -> Result<Self> {
        let (source, filename, config) = model.model_info()?;
        let vb = source.var_builder(filename, opts)?;
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
