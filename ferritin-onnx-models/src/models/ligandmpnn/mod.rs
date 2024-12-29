//! ESM2 Struct. Loads the hf tokenizer
//!
use anyhow::{anyhow, Result};
use candle_hf_hub::api::sync::Api;
use std::path::PathBuf;

pub enum LigandMPNNModels {
    ProteinMPNN,
    LigandMPNN,
}

pub struct LigandMPNN {}

impl LigandMPNN {
    pub fn load_model_path(model: LigandMPNNModels) -> Result<(PathBuf, PathBuf)> {
        let api = Api::new().unwrap();
        let (repo_id, encoder_filename, decoder_filename) = match model {
            LigandMPNNModels::ProteinMPNN => (
                "zcpbx/proteinmpnn-v48-030-onnx".to_string(),
                "protmpnn_encoder.onnx",
                "protmpnn_decoder_step.onnx",
            ),
            LigandMPNNModels::LigandMPNN => (
                "zcpbx/ligandmpnn-v32-030-25-onnx".to_string(),
                "ligand_encoder.onnx",
                "ligand_decoder.onnx",
            ),
        };
        let encoder_path = api.model(repo_id.clone()).get(encoder_filename).unwrap();
        let decoder_path = api.model(repo_id).get(decoder_filename).unwrap();
        Ok((encoder_path, decoder_path))
    }
    // pub fn load_tokenizer() -> Result<Tokenizer> {
    //     let tokenizer_bytes = include_bytes!("tokenizer.json");
    //     Tokenizer::from_bytes(tokenizer_bytes)
    //         .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
}
