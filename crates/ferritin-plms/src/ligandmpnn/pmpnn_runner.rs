//! ProteinMPNN Runner
//!
//! Loads and runs ProteinMPNN models without requiring callers to import candle directly.
use super::configs::ProteinMPNNConfig;
use super::model::ProteinMPNN;
use super::proteinfeatures::ProteinFeatures;
use crate::types::PseudoProbability;
use anyhow::{Result, anyhow};
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::HFClientSync;
use std::path::Path;

const PMPNN_DTYPE: DType = DType::F32;

pub enum ProteinMPNNModels {
    /// proteinmpnn_v_48_020: k_neighbors=48, dropout=0.2
    V48_020,
}

impl ProteinMPNNModels {
    fn hf_info(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        // (owner, repo, revision, filename)
        match self {
            Self::V48_020 => (
                "zcpbx",
                "ligandmpnn-weights",
                "main",
                "model_params/proteinmpnn_v_48_020.pt",
            ),
        }
    }
}

pub struct ProteinMPNNRunner {
    model: ProteinMPNN,
}

impl ProteinMPNNRunner {
    /// Load a ProteinMPNN model from HuggingFace hub.
    pub fn load_model(modeltype: ProteinMPNNModels, device: Device) -> Result<Self> {
        let (owner, repo, revision, filename) = modeltype.hf_info();
        let client = HFClientSync::new()?;
        let hf_repo = client.model(owner, repo);
        let weights_path = hf_repo
            .download_file()
            .filename(filename)
            .revision(revision)
            .send()
            .map_err(|e| anyhow!("Failed to download ProteinMPNN weights from HF hub: {e}"))?;
        Self::from_path(&weights_path, device)
    }

    /// Load from a local .pt file (e.g. from ferritin-test-data or a cached download).
    pub fn from_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        let path = path.as_ref();
        let pth = PthTensors::new(path, Some("model_state_dict"))
            .map_err(|e| anyhow!("Failed to open {}: {e}", path.display()))?;
        let vb = VarBuilder::from_backend(Box::new(pth), PMPNN_DTYPE, device);
        let config = ProteinMPNNConfig::proteinmpnn();
        let model = ProteinMPNN::load(vb, &config)
            .map_err(|e| anyhow!("Failed to load ProteinMPNN weights: {e}"))?;
        Ok(Self { model })
    }

    /// Run ProteinMPNN and return a (L, 21) log-probability tensor for all positions.
    ///
    /// Useful for numerical parity tests against a Python reference.  Values are
    /// log-softmax of the raw logits from a single structure-conditioned forward pass
    /// (the same computation as `simple_decode`).
    pub fn get_log_probs(&self, features: &ProteinFeatures) -> Result<Tensor> {
        let output = self
            .model
            .simple_decode(features)
            .map_err(|e| anyhow!("ProteinMPNN forward pass failed: {e}"))?;
        // log_probs shape: (1, L, 21) — squeeze the batch dimension
        output
            .get_log_probs()
            .squeeze(0)
            .map_err(|e| anyhow!("Failed to squeeze batch dimension: {e}"))
    }

    /// Run ProteinMPNN and return per-residue pseudo-probabilities for the 21 amino acids.
    pub fn get_pseudo_probabilities(
        &self,
        features: &ProteinFeatures,
    ) -> Result<Vec<PseudoProbability>> {
        let output = self
            .model
            .simple_decode(features)
            .map_err(|e| anyhow!("ProteinMPNN forward pass failed: {e}"))?;
        output
            .get_pseudo_probabilities()
            .map_err(|e| anyhow!("Failed to extract pseudo-probabilities: {e}"))
    }
}
