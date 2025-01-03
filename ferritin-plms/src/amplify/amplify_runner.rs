//! Amplify RUnner
//!
//! Class for loading and running the AMPLIFY models

use super::super::types::{ContactMap, PseudoProbability};
use super::amplify::AMPLIFY;
use super::config::AMPLIFYConfig;
use super::outputs::ModelOutput;
use anyhow::{anyhow, Error as E, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::ops;
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

pub enum AmplifyModels {
    AMP120M,
    AMP350M,
}
impl AmplifyModels {
    pub fn get_model_files(model: Self) -> Result<(&'static str, &'static str)> {
        let (repo, rev) = match model {
            AmplifyModels::AMP120M => ("chandar-lab/AMPLIFY_120M", "main"),
            AmplifyModels::AMP350M => ("chandar-lab/AMPLIFY_350M", "main"),
        };
        Ok((repo, rev))
    }
}

pub struct AmplifyRunner {
    model: AMPLIFY,
    tokenizer: Tokenizer,
}
impl AmplifyRunner {
    pub fn load_model(modeltype: AmplifyModels, device: Device) -> Result<AmplifyRunner> {
        let (model_id, revision) = AmplifyModels::get_model_files(modeltype)?;
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_str = config_str
            .replace("SwiGLU", "swiglu")
            .replace("Swiglu", "swiglu");
        let config: AMPLIFYConfig = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let amp_dtype = DType::F32;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], amp_dtype, &device)?
        };
        let model = AMPLIFY::load(vb, &config)?;
        Ok(AmplifyRunner { model, tokenizer })
    }
    pub fn run_forward(&self, prot_sequence: &str) -> Result<ModelOutput> {
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
        let model_output = self.run_forward(prot_sequence)?;
        let predictions = model_output.logits.argmax(D::Minus1)?;
        let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        let decoded = self.tokenizer.decode(indices.as_slice(), true)?;
        let decoded = decoded.replace(" ", "");
        Ok(decoded)
    }
    //
    pub fn get_pseudo_probabilities(&self, prot_sequence: &str) -> Result<Vec<PseudoProbability>> {
        let model_output = self.run_forward(prot_sequence)?;
        let predictions = model_output.logits;
        let outputs = self.extract_logits(&predictions)?;
        Ok(outputs)
    }
    pub fn get_contact_map(&self, prot_sequence: &str) -> Result<Vec<ContactMap>> {
        let model_output = self.run_forward(prot_sequence)?;
        let contact_map_tensor = model_output.get_contact_map()?;
        // let (res1, res2, attn) = contact_map_tensor.clone().unwrap().dims3()?;

        // Note: we might want mean or average here.
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
                            .decode(&[i as u32], true)
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
        let shape = tensor.dims();
        let mut logit_positions = Vec::new();
        for seq_pos in 0..shape[1] {
            for vocab_idx in 0..shape[2] {
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
