//! ESM2 Struct. Loads the hf tokenizer
//!
use anyhow::{anyhow, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub enum ESM2Models {
    ESM2_T6_8M,
    ESM2_T12_35M,
    ESM2_T30_150M,
    // ESM2_T33_650M,
}

pub struct ESM2 {}

impl ESM2 {
    pub fn load_model_path(model: ESM2Models) -> Result<PathBuf> {
        let api = Api::new().unwrap();
        let repo_id = match model {
            ESM2Models::ESM2_T6_8M => "zcpbx/esm2-t6-8m-UR50D-onnx",
            ESM2Models::ESM2_T12_35M => "zcpbx/esm2-t12-35M-UR50D-onnx",
            ESM2Models::ESM2_T30_150M => "zcpbx/esm2-t30-150M-UR50D-onnx",
            // ESM2Models::ESM2_T33_650M => "zcpbx/esm2-t33-650M-UR50D-onnx",
        }
        .to_string();

        let model_path = api.model(repo_id).get("model.onnx").unwrap();
        Ok(model_path)
    }
    pub fn load_tokenizer() -> Result<Tokenizer> {
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer_load() -> Result<()> {
        let tokenizer = ESM2::load_tokenizer()?;
        let text = "MLKLRV";
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("Failed to encode: {}", e))?;
        let tokens = encoding.get_tokens();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens, &["M", "L", "K", "L", "R", "V"]);
        Ok(())
    }
}
