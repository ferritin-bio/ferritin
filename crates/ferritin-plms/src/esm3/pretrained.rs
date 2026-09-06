//! ESM3 pretrained weight loading.
//!
//! Downloads weights from `EvolutionaryScale/esm3-sm-open-v1` on HuggingFace (gated access —
//! the user must accept the Cambrian Non-Commercial license and run `huggingface-cli login`).
//!
//! ## Weight layout
//!
//! The `.pth` file is a plain PyTorch state dict saved via `torch.save(model.state_dict(), ...)`.
//! No wrapping key is used; `PthTensors::new(path, None)` loads it directly.
//!
//! | Model attribute (Python) | VarBuilder prefix |
//! |--------------------------|-------------------|
//! | `encoder.*`              | `encoder.*`       |
//! | `transformer.*`          | `transformer.*`   |
//! | `output_heads.*`         | `output_heads.*`  |
//!
//! The structure encoder uses a separate checkpoint (`esm3_structure_encoder_v0.pth`):
//!
//! | Python attribute        | VarBuilder prefix  |
//! |-------------------------|--------------------|
//! | `encoder.*` (blocks)    | `encoder.*`        |
//! | `pre_vq_proj.weight`    | `pre_vq_proj.*`    |
//! | `codebook.embeddings`   | `codebook.*`       |

use crate::esm3::models::esm3::{ESM3, ESM3Config};
use crate::esm3::models::vqvae::{StructureTokenEncoder, VqVaeConfig};
use crate::esm3::tokenization::sequence::tokenize_sequence;
use crate::loader::{LoadOptions, WeightSource};
use crate::plm_runner::{ModelMetadata, PlmRunner, SpecialTokenLayout};
use anyhow::Result;
use candle_core::{Device, Tensor};

// ── ESM3Models enum ───────────────────────────────────────────────────────────

/// Available ESM3 model variants.
pub enum ESM3Models {
    /// esm3-sm-open-v1 — 1.4B parameter open-access model (Cambrian Non-Commercial license).
    SmOpen,
}

/// The `esm3-sm-open-v1` repo, whose `.pth` checkpoints store tensors at the root.
const ESM3_REPO: WeightSource = WeightSource::pth("EvolutionaryScale/esm3-sm-open-v1", None);

impl ESM3Models {
    /// Weight source, `.pth` filename, and config for this variant.
    pub fn model_info(&self) -> (WeightSource, &'static str, ESM3Config) {
        match self {
            Self::SmOpen => (ESM3_REPO, "esm3_sm_open_v1.pth", ESM3Config::sm_open()),
        }
    }
}

// ── ESM3Runner ────────────────────────────────────────────────────────────────

/// Wraps a loaded ESM3 model for sequence embedding and multi-track inference.
///
/// ## HuggingFace access
/// The ESM3 model is gated. Before calling `from_pretrained` the user must:
/// 1. Accept the Cambrian Non-Commercial license at
///    <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1>
/// 2. Run `huggingface-cli login` (or set `HF_TOKEN` env var)
pub struct ESM3Runner {
    model: ESM3,
    device: Device,
}

impl ESM3Runner {
    /// Download the ESM3 weights from HuggingFace and load the model.
    ///
    /// The `.pth` checkpoint is loaded directly via `candle_core::pickle::PthTensors`.
    pub fn from_pretrained(variant: ESM3Models, device: Device) -> Result<Self> {
        Self::from_pretrained_with(variant, &LoadOptions::new(device))
    }

    /// Load with an explicit device and dtype (ferritin-100.9).
    pub fn from_pretrained_with(variant: ESM3Models, opts: &LoadOptions) -> Result<Self> {
        let (source, filename, config) = variant.model_info();
        let vb = source.var_builder(filename, opts)?;
        let model = ESM3::load(vb, config)?;
        Ok(Self {
            model,
            device: opts.device.clone(),
        })
    }

    /// Tokenize `sequence` and return per-residue embeddings `(1, L, d_model)`.
    ///
    /// Only the sequence track is provided; all other tracks are `None`.
    /// Embeddings are the pre-norm transformer hidden states (raw activations).
    pub fn embed_sequence(&self, sequence: &str) -> Result<Tensor> {
        let token_ids = tokenize_sequence(sequence, true);
        let tokens = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?; // (1, L)

        let output = self.model.forward(
            Some(&tokens), // sequence_tokens
            None,          // structure_tokens
            None,          // ss8_tokens
            None,          // sasa_tokens
            None,          // function_tokens
            None,          // residue_annotation_tokens
            None,          // average_plddt
            None,          // per_res_plddt
            None,          // sequence_id
            None,          // structure_coords
            None,          // chain_id
        )?;

        output
            .embeddings
            .ok_or_else(|| anyhow::anyhow!("ESM3 forward() returned no embeddings"))
    }
}

impl PlmRunner for ESM3Runner {
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        self.embed_sequence(sequence)
    }

    fn model_name(&self) -> &str {
        "esm3"
    }

    /// `embed_sequence` calls `tokenize_sequence(.., true)`, which prepends
    /// `SEQUENCE_BOS_TOKEN` and appends `SEQUENCE_EOS_TOKEN`.
    fn special_tokens(&self) -> SpecialTokenLayout {
        SpecialTokenLayout::BOS_EOS
    }

    fn metadata(&self) -> ModelMetadata {
        let config = &self.model.config;
        ModelMetadata {
            d_model: config.d_model,
            n_layers: config.n_layers,
            vocab_size: config.d_sequence_vocab,
            // Rotary positions: no hard architectural cap.
            max_positions: None,
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    /// Sequence-track logits `(1, L + 2, d_sequence_vocab)`.
    ///
    /// Only the sequence track is supplied; the other tracks are `None`.
    fn logits(&self, sequence: &str) -> Result<Tensor> {
        let token_ids = tokenize_sequence(sequence, true);
        let tokens = Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let output = self.model.forward(
            Some(&tokens),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        output
            .sequence_logits
            .ok_or_else(|| anyhow::anyhow!("ESM3 forward() returned no sequence logits"))
    }
}

// ── StructureEncoderRunner ────────────────────────────────────────────────────

/// Wraps the ESM3 structure token encoder for converting backbone coordinates to tokens.
///
/// Uses a separate checkpoint (`esm3_structure_encoder_v0.pth`) from the main model.
pub struct StructureEncoderRunner {
    encoder: StructureTokenEncoder,
    #[allow(dead_code)]
    device: Device,
}

impl StructureEncoderRunner {
    /// Download and load the structure encoder from HuggingFace.
    pub fn from_pretrained(device: Device) -> Result<Self> {
        Self::from_pretrained_with(&LoadOptions::new(device))
    }

    /// Load with an explicit device and dtype (ferritin-100.9).
    pub fn from_pretrained_with(opts: &LoadOptions) -> Result<Self> {
        let vb = ESM3_REPO.var_builder("esm3_structure_encoder_v0.pth", opts)?;
        let encoder = StructureTokenEncoder::load(vb, VqVaeConfig::default())?;
        Ok(Self {
            encoder,
            device: opts.device.clone(),
        })
    }

    /// Encode backbone coordinates to structure tokens.
    ///
    /// - `coords`: `(B, L, 3, 3)` backbone `(N, CA, C)` atom positions.
    ///
    /// Returns `(B, L)` u32 structure token indices.
    pub fn encode(&self, coords: &Tensor) -> Result<Tensor> {
        self.encoder.encode(coords, None, None).map_err(Into::into)
    }
}
