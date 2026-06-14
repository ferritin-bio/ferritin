//! Amplify RUnner
//!
//! Class for loading and running the AMPLIFY models

use super::super::types::{ContactMap, PseudoProbability};
use super::amplify::{AMPLIFY, AmplifyOutput};
use super::config::AMPLIFYConfig;
use crate::plm_runner::PlmRunner;
use anyhow::{Error as E, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_nn::ops;
use hf_hub::HFClientSync;
use tokenizers::Tokenizer;

const AMPLIFY_DTYPE: DType = DType::F32;

pub enum AmplifyModels {
    AMP120M,
    AMP350M,
}
impl AmplifyModels {
    pub fn get_model_files(model: Self) -> (&'static str, &'static str) {
        match model {
            AmplifyModels::AMP120M => ("chandar-lab/AMPLIFY_120M", "main"),
            AmplifyModels::AMP350M => ("chandar-lab/AMPLIFY_350M", "main"),
        }
    }
}

pub struct AmplifyRunner {
    model: AMPLIFY,
    tokenizer: Tokenizer,
}
impl AmplifyRunner {
    pub fn load_model(modeltype: AmplifyModels, device: Device) -> Result<AmplifyRunner> {
        let (model_id, revision) = AmplifyModels::get_model_files(modeltype);
        let (owner, name) = model_id.split_once('/').unwrap_or(("", model_id));
        let client = HFClientSync::new()?;
        let repo = client.model(owner, name);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let config = repo.download_file().filename("config.json").revision(revision).send()?;
            let tokenizer = repo.download_file().filename("tokenizer.json").revision(revision).send()?;
            let weights = repo.download_file().filename("model.safetensors").revision(revision).send()?;
            (config, tokenizer, weights)
        };
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_str = config_str
            .replace("SwiGLU", "swiglu")
            .replace("Swiglu", "swiglu");
        let config: AMPLIFYConfig = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], AMPLIFY_DTYPE, &device)?
        };
        let model = AMPLIFY::load(vb, &config)?;
        Ok(AmplifyRunner { model, tokenizer })
    }
    pub fn run_forward(&self, prot_sequence: &str) -> Result<AmplifyOutput> {
        let device = self.model.get_device();
        let tokens = self
            .tokenizer
            .encode(prot_sequence.to_string(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let encoded = self.model.forward(&token_ids, None, false, true)?;
        Ok(encoded)
    }
    pub fn get_best_prediction(
        &self,
        prot_sequence: &str,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let model_output: AmplifyOutput = self.run_forward(prot_sequence)?;
        let predictions = model_output.logits.argmax(D::Minus1)?;
        let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        let decoded = self.tokenizer.decode(indices.as_slice(), true)?;
        let decoded = decoded.replace(" ", "");
        Ok(decoded)
    }
    pub fn get_pseudo_probabilities(&self, prot_sequence: &str) -> Result<Vec<PseudoProbability>> {
        let model_output: AmplifyOutput = self.run_forward(prot_sequence)?;
        let predictions = model_output.logits;
        let outputs = self.extract_logits(&predictions)?;
        Ok(outputs)
    }
    pub fn get_contact_map(&self, prot_sequence: &str) -> Result<Vec<ContactMap>> {
        let model_output: AmplifyOutput = self.run_forward(prot_sequence)?;
        let contact_map_tensor = model_output.get_contact_map()?;
        let averaged = contact_map_tensor.clone().unwrap().max_keepdim(D::Minus1)?;
        let (position1, position2, val) = averaged.dims3()?;
        let data = averaged.to_vec3::<f32>()?;

        let mut contacts = Vec::new();
        for i in 0..position1 {
            for j in 0..position2 {
                for k in 0..val {
                    contacts.push(ContactMap {
                        position_1: i,
                        amino_acid_1: self
                            .tokenizer
                            .decode(&[i as u32], true)
                            .ok()
                            .and_then(|s| s.chars().next())
                            .unwrap_or('?'),
                        position_2: j,
                        amino_acid_2: self
                            .tokenizer
                            .decode(&[j as u32], true)
                            .ok()
                            .and_then(|s| s.chars().next())
                            .unwrap_or('?'),
                        contact_estimate: data[i][j][k],
                        layer: 1,
                    });
                }
            }
        }
        Ok(contacts)
    }
    // Softmax and simplify
    fn extract_logits(&self, tensor: &Tensor) -> Result<Vec<PseudoProbability>> {
        let tensor = ops::softmax(tensor, D::Minus1)?;
        let data = tensor.to_vec3::<f32>()?;
        let (_, seq_len, vocab_size) = tensor.dims3()?;
        let mut logit_positions = Vec::with_capacity(seq_len * vocab_size);
        for seq_pos in 0..seq_len {
            for vocab_idx in 0..vocab_size {
                let score = data[0][seq_pos][vocab_idx];
                let amino_acid_char = self
                    .tokenizer
                    .decode(&[vocab_idx as u32], false)
                    .map_err(|e| anyhow!("Failed to decode: {}", e))?
                    .chars()
                    .next()
                    .ok_or_else(|| anyhow!("Empty decoded string"))?;
                logit_positions.push(PseudoProbability {
                    position: seq_pos,
                    amino_acid: amino_acid_char,
                    pseudo_prob: score,
                });
            }
        }
        Ok(logit_positions)
    }
}

impl PlmRunner for AmplifyRunner {
    /// Run AMPLIFY and return the last-layer hidden states as per-residue embeddings.
    ///
    /// Shape: `(1, L, hidden_size)` where `L` includes BOS and EOS tokens.
    fn embed(&self, sequence: &str) -> Result<Tensor> {
        let device = self.model.get_device();
        let tokens = self
            .tokenizer
            .encode(sequence.to_string(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let output = self.model.forward(&token_ids, None, true, false)?;
        let mut hidden_states = output
            .hidden_states
            .ok_or_else(|| anyhow!("AMPLIFY forward() returned no hidden states"))?;
        hidden_states
            .pop()
            .ok_or_else(|| anyhow!("AMPLIFY returned empty hidden states list"))
    }

    fn model_name(&self) -> &str {
        "amplify"
    }
}
