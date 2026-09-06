//! ProteinMPNN Runner
//!
//! Loads and runs ProteinMPNN models without requiring callers to import candle directly.
use super::configs::ProteinMPNNConfig;
use super::model::ProteinMPNN;
use super::proteinfeatures::ProteinFeatures;
use crate::loader::{Format, LoadOptions, WeightSource, var_builder_from_path};
use crate::types::PseudoProbability;
use anyhow::{Result, anyhow};
use candle_core::{Device, Tensor};
use std::path::Path;

/// ProteinMPNN `.pt` checkpoints nest their tensors under `model_state_dict`.
const PMPNN_FORMAT: Format = Format::Pth {
    root_key: Some("model_state_dict"),
};

pub enum ProteinMPNNModels {
    /// proteinmpnn_v_48_020: k_neighbors=48, dropout=0.2
    V48_020,
}

impl ProteinMPNNModels {
    /// Weight source and file for this variant.
    pub fn model_info(&self) -> (WeightSource, &'static str) {
        match self {
            Self::V48_020 => (
                WeightSource::pth("zcpbx/ligandmpnn-weights", Some("model_state_dict"))
                    .at_revision("main"),
                "model_params/proteinmpnn_v_48_020.pt",
            ),
        }
    }

    /// Renamed to [`model_info`][Self::model_info] (ferritin-100.8).
    #[deprecated(since = "0.4.0", note = "renamed to `model_info`")]
    pub fn hf_info(&self) -> (WeightSource, &'static str) {
        self.model_info()
    }
}

pub struct ProteinMPNNRunner {
    model: ProteinMPNN,
}

impl ProteinMPNNRunner {
    /// Load a ProteinMPNN model from HuggingFace hub.
    pub fn from_pretrained(modeltype: ProteinMPNNModels, device: Device) -> Result<Self> {
        Self::from_pretrained_with(modeltype, &LoadOptions::new(device))
    }

    /// Renamed to [`from_pretrained`][Self::from_pretrained] (ferritin-100.8).
    #[deprecated(since = "0.4.0", note = "renamed to `from_pretrained`")]
    pub fn load_model(modeltype: ProteinMPNNModels, device: Device) -> Result<Self> {
        Self::from_pretrained(modeltype, device)
    }

    /// Renamed to [`from_pretrained_with`][Self::from_pretrained_with] (ferritin-100.8).
    #[deprecated(since = "0.4.0", note = "renamed to `from_pretrained_with`")]
    pub fn load_model_with(modeltype: ProteinMPNNModels, opts: &LoadOptions) -> Result<Self> {
        Self::from_pretrained_with(modeltype, opts)
    }

    /// Load with an explicit device and dtype (ferritin-100.9).
    pub fn from_pretrained_with(modeltype: ProteinMPNNModels, opts: &LoadOptions) -> Result<Self> {
        let (source, filename) = modeltype.model_info();
        let weights_path = source.fetch(filename)?;
        Self::from_path_with(&weights_path, opts)
    }

    /// Load from a local .pt file (e.g. from ferritin-test-data or a cached download).
    pub fn from_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::from_path_with(path, &LoadOptions::new(device))
    }

    /// Load from a local file with an explicit device and dtype.
    pub fn from_path_with(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self> {
        let path = path.as_ref();
        let vb = var_builder_from_path(path, PMPNN_FORMAT, opts)?;
        let config = ProteinMPNNConfig::proteinmpnn();
        let model = ProteinMPNN::load(vb, &config)
            .map_err(|e| anyhow!("Failed to load ProteinMPNN weights: {e}"))?;
        Ok(Self { model })
    }

    /// Consume the runner and yield the loaded model.
    ///
    /// Lets callers that need the bare [`ProteinMPNN`] — such as
    /// `MPNNExecConfig::load_model` — go through this tested loading path
    /// instead of duplicating the download (ferritin-100.10).
    pub fn into_model(self) -> ProteinMPNN {
        self.model
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
