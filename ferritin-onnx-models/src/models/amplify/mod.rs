//! AMPLIFY is an optimized transformer model focused on optimizing the context of sequence models
//! while maintaining computational efficiency.
//!
//! Key features:
//! - Rotary positional embeddings
//! - RMSNorm for improved training stability
//! - SwiGLU activation function
//! - Specialized architecture optimizations
//! - Memory efficient inference
//!
//!
//! AMPLIFY
//!
//! Utilities for working with the Amplify protein language model.
//!
//! - [GH Amplify Code](https://github.com/chandar-lab/AMPLIFY)
//! - [HF - 120M Model ](https://huggingface.co/chandar-lab/AMPLIFY_120M)
//! - [Paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
//!
use anyhow::{anyhow, Result};
use candle_hf_hub::api::sync::Api;
use std::path::PathBuf;
use tokenizers::Tokenizer;

pub enum AMPLIFYModels {
    AMP_120M,
    AMP_350M,
}

/// The AMPLIFY model
///
/// - [GH PythonModel](https://github.com/chandar-lab/AMPLIFY/blob/rc-0.1/src/amplify/model/amplify.py)
/// - [paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1)
/// - [HF](https://huggingface.co/chandar-lab/AMPLIFY_120M)
///
#[derive(Debug)]
pub struct AMPLIFY {}

impl AMPLIFY {
    pub fn load_model_path(model: AMPLIFYModels) -> Result<PathBuf> {
        let api = Api::new().unwrap();
        let repo_id = match model {
            AMPLIFYModels::AMP_120M => "zcpbx/amplify-120M-onnx",
            AMPLIFYModels::AMP_350M => "zcpbx/amplify-350M-onnx",
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
