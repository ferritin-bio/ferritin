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

use crate::esm3::models::esm3::{ESM3Config, ESM3};
use crate::esm3::models::vqvae::{StructureTokenEncoder, VqVaeConfig};
use crate::esm3::tokenization::sequence::tokenize_sequence;
use crate::plm_runner::PlmRunner;
use anyhow::{Context, Result};
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::HFClientSync;

const ESM3_DTYPE: DType = DType::F32;

// ── ESM3Models enum ───────────────────────────────────────────────────────────

/// Available ESM3 model variants.
pub enum ESM3Models {
    /// esm3-sm-open-v1 — 1.4B parameter open-access model (Cambrian Non-Commercial license).
    SmOpen,
}

impl ESM3Models {
    /// HuggingFace repo and `.pth` filename for this variant.
    pub fn model_info(&self) -> (&'static str, &'static str, ESM3Config) {
        match self {
            Self::SmOpen => (
                "EvolutionaryScale/esm3-sm-open-v1",
                "esm3_sm_open_v1.pth",
                ESM3Config::sm_open(),
            ),
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
        let (repo_id, filename, config) = variant.model_info();
        let (owner, name) = repo_id.split_once('/').unwrap_or(("", repo_id));

        let client = HFClientSync::new().context("failed to initialise HF client")?;
        let weights_path = client
            .model(owner, name)
            .download_file()
            .filename(filename)
            .send()
            .with_context(|| format!("failed to download {} from {}", filename, repo_id))?;

        let pth = PthTensors::new(&weights_path, None)
            .with_context(|| format!("failed to parse {}", weights_path.display()))?;
        let vb = VarBuilder::from_backend(Box::new(pth), ESM3_DTYPE, device.clone());

        let model = ESM3::load(vb, config)?;
        Ok(Self { model, device })
    }

    /// Tokenize `sequence` and return per-residue embeddings `(1, L, d_model)`.
    ///
    /// Only the sequence track is provided; all other tracks are `None`.
    /// Embeddings are the pre-norm transformer hidden states (raw activations).
    pub fn embed_sequence(&self, sequence: &str) -> Result<Tensor> {
        let token_ids = tokenize_sequence(sequence, true);
        let tokens =
            Tensor::new(token_ids.as_slice(), &self.device)?.unsqueeze(0)?; // (1, L)

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
}

// ── StructureEncoderRunner ────────────────────────────────────────────────────

/// Wraps the ESM3 structure token encoder for converting backbone coordinates to tokens.
///
/// Uses a separate checkpoint (`esm3_structure_encoder_v0.pth`) from the main model.
pub struct StructureEncoderRunner {
    encoder: StructureTokenEncoder,
    device: Device,
}

impl StructureEncoderRunner {
    /// Download and load the structure encoder from HuggingFace.
    pub fn from_pretrained(device: Device) -> Result<Self> {
        let repo_id = "EvolutionaryScale/esm3-sm-open-v1";
        let filename = "esm3_structure_encoder_v0.pth";
        let (owner, name) = repo_id.split_once('/').unwrap_or(("", repo_id));

        let client = HFClientSync::new().context("failed to initialise HF client")?;
        let weights_path = client
            .model(owner, name)
            .download_file()
            .filename(filename)
            .send()
            .with_context(|| format!("failed to download {} from {}", filename, repo_id))?;

        let pth = PthTensors::new(&weights_path, None)
            .with_context(|| format!("failed to parse {}", weights_path.display()))?;
        let vb = VarBuilder::from_backend(Box::new(pth), ESM3_DTYPE, device.clone());

        let encoder = StructureTokenEncoder::load(vb, VqVaeConfig::default())?;
        Ok(Self { encoder, device })
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
