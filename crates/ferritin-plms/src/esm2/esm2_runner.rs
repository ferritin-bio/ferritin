//! ESM2 Runner
//!
//! Class for loading and running the ESM2 models
use super::esm2::{ESM2, ESM2Config, ESM2Output};
use anyhow::{Error as E, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json;
use tokenizers::Tokenizer;

const ESM2_DTYPE: DType = DType::F32;

pub enum ESM2Models {
    T6_8M,
    T12_35M,
    T30_150M,
    T33_650M,
    T36_3B,
    T48_15B,
}
impl ESM2Models {
    pub fn get_model_files(model: Self) -> (&'static str, &'static str, ESM2Config) {
        match model {
            Self::T6_8M => ("facebook/esm2_t6_8M_UR50D", "main", ESM2Config::t6_8m()),
            Self::T12_35M => ("facebook/esm2_t12_35M_UR50D", "main", ESM2Config::t12_35m()),
            Self::T30_150M => (
                "facebook/esm2_t30_150M_UR50D",
                "main",
                ESM2Config::t30_150m(),
            ),
            Self::T33_650M => (
                "facebook/esm2_t33_650M_UR50D",
                "main",
                ESM2Config::t33_650m(),
            ),
            Self::T36_3B => ("facebook/esm2_t36_3B_UR50D", "main", ESM2Config::t36_3b()),
            Self::T48_15B => ("facebook/esm2_t48_15B_UR50D", "main", ESM2Config::t48_15b()),
        }
    }
}

pub struct ESM2Runner {
    model: ESM2,
    tokenizer: Tokenizer,
}
impl ESM2Runner {
    /// Load model from HuggingFace hub, downloading config.json, tokenizer files, and weights.
    pub fn load_model(modeltype: ESM2Models, device: Device) -> Result<ESM2Runner> {
        let (model_id, revision, fallback_config) = ESM2Models::get_model_files(modeltype);
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let api = Api::new()?;
        let api = api.repo(repo);
        // Try to load config from HF hub; fall back to hardcoded config if unavailable.
        let config = match api.get("config.json") {
            Ok(config_path) => {
                let config_str = std::fs::read_to_string(config_path)?;
                serde_json::from_str::<ESM2Config>(&config_str).unwrap_or(fallback_config)
            }
            Err(_) => fallback_config,
        };
        let weights_filename = api.get("model.safetensors")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], ESM2_DTYPE, &device)?
        };
        let model = ESM2::load(vb, config)?;
        let tokenizer = ESM2::load_tokenizer()?;
        Ok(ESM2Runner { model, tokenizer })
    }
    pub fn run_forward(&self, prot_sequence: &str) -> Result<ESM2Output> {
        let device = self.model.get_device();
        let tokens = self
            .tokenizer
            .encode(prot_sequence.to_string(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let encoded = self.model.forward(&token_ids)?;
        Ok(encoded)
    }
    pub fn decode_logits(&self, output: ESM2Output) -> Result<String> {
        // Get the predicted token IDs by taking argmax along the vocabulary dimension
        let predicted_token_ids = output.logits.argmax(2)?;
        let predicted_token_ids = if predicted_token_ids.dims().len() > 1 {
            predicted_token_ids.squeeze(0)?
        } else {
            predicted_token_ids
        };
        let token_ids: Vec<u32> = predicted_token_ids.to_vec1::<u32>()?;
        let decoded_sequence = self
            .tokenizer
            .decode(&token_ids, true) // set skip_special_tokens to true
            .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?
            .replace(" ", "");
        Ok(decoded_sequence)
    }
}
