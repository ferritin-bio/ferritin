//! Amplify RUnner
//!
//! Class for loading and running the AMPLIFY models

use super::super::types::{ContactMap, PseudoProbability};
use super::amplify::AMPLIFY;
use super::config::AMPLIFYConfig;
use super::outputs::ModelOutput;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

pub enum AmplifyModels {
    AMP120M,
    AMP350M,
}
impl AmplifyModels {
    pub fn get_model_files(model: Self) -> Result<(&str, &str)> {
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
    pub fn get_pseudo_probabilities(&self, prot_sequence: &str) -> Result<()> {
        let model_output = self.run_forward(prot_sequence)?;
        let predictions = model_output.logits;
        println!("{:?}", predictions);
        // let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        // let decoded = self.tokenizer.decode(indices.as_slice(), true)?;
        Ok(())
    }
    // pub fn get_pseudo_probabilities(&self, prot_sequence: &str) -> Result<Vec<PseudoProbability>> {
    //     let model_output = self.run_forward(prot_sequence)?;
    //     let predictions = model_output.logits?;
    //     Ok(predictions)
    //     // let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    //     // let decoded = self.tokenizer.decode(indices.as_slice(), true)?;
    // }

    pub fn get_contact_map(&self, prot_sequence: &str) -> Result<Option<Tensor>> {
        let model_output = self.run_forward(prot_sequence)?;
        let contact_map = model_output.get_contact_map()?;
        Ok(contact_map)
    }
}
