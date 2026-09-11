//! ProteinMPNN Runner
//!
//! Loads and runs ProteinMPNN models without requiring callers to import candle directly.
use super::configs::ProteinMPNNConfig;
use super::model::ProteinMPNN;
use super::proteinfeatures::ProteinFeatures;
use crate::loader::{Format, LoadOptions, WeightSource, var_builder_from_path};
use crate::registry::{self, ModelCard};
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
    /// ligandmpnn_v_32_020_25: k_neighbors=32, 25 ligand context atoms.
    ///
    /// Same `ProteinMPNN` struct, a different featurizer and an extra context
    /// path — see `ModelTypes::LigandMPNN` (ferritin-100.11).
    LigandV32_020_25,
}

impl ProteinMPNNModels {
    /// This variant's registry id.
    pub const fn registry_id(&self) -> &'static str {
        match self {
            Self::V48_020 => "proteinmpnn-v48-020",
            Self::LigandV32_020_25 => "ligandmpnn-v32-020-25",
        }
    }

    /// The architecture config this checkpoint expects.
    ///
    /// `k_neighbors` and `atom_context_num` differ between the two and are
    /// NOT recoverable from the tensor shapes — they live in the `.pt` as
    /// plain Python ints, which candle's pickle reader does not surface — so
    /// picking the wrong one loads cleanly and computes the wrong thing.
    pub fn config(&self) -> ProteinMPNNConfig {
        match self {
            Self::V48_020 => ProteinMPNNConfig::proteinmpnn(),
            Self::LigandV32_020_25 => ProteinMPNNConfig::ligandmpnn(),
        }
    }

    /// This variant's [`ModelCard`].
    pub fn card(&self) -> &'static ModelCard {
        registry::lookup(self.registry_id())
            .expect("every ProteinMPNNModels variant must have a registry entry")
    }

    /// Weight source and file for this variant.
    ///
    /// Delegates to [`REGISTRY`][crate::registry::REGISTRY] (ferritin-goh.1).
    pub fn model_info(&self) -> (WeightSource, &'static str) {
        let card = self.card();
        (card.source, card.file)
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
        Self::from_path_as(&weights_path, &modeltype.config(), opts)
    }

    /// Load from a local .pt file (e.g. from ferritin-test-data or a cached download).
    ///
    /// Assumes a ProteinMPNN checkpoint. For LigandMPNN use
    /// [`from_path_as`][Self::from_path_as] — the two need different configs
    /// and a LigandMPNN checkpoint loaded as ProteinMPNN silently ignores its
    /// nine ligand tensors.
    pub fn from_path(path: impl AsRef<Path>, device: Device) -> Result<Self> {
        Self::from_path_with(path, &LoadOptions::new(device))
    }

    /// Load from a local file with an explicit device and dtype.
    pub fn from_path_with(path: impl AsRef<Path>, opts: &LoadOptions) -> Result<Self> {
        Self::from_path_as(path, &ProteinMPNNConfig::proteinmpnn(), opts)
    }

    /// Load from a local file against an explicit architecture config.
    pub fn from_path_as(
        path: impl AsRef<Path>,
        config: &ProteinMPNNConfig,
        opts: &LoadOptions,
    ) -> Result<Self> {
        let path = path.as_ref();
        let vb = var_builder_from_path(path, PMPNN_FORMAT, opts)?;
        let model = ProteinMPNN::load(vb, config)
            .map_err(|e| anyhow!("Failed to load MPNN weights from {}: {e}", path.display()))?;
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
