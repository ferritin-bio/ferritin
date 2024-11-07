use candle_core::{DType, Device, Result, Tensor};
use candle_nn::loss::{CrossEntropyConfig, CrossEntropyLoss};
use std::collections::HashMap;
use std::path::Path;

pub struct LossConfig {
    pub vocab_path: String,
    pub pad_token_id: usize,
    pub mask_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub unk_token_id: usize,
    pub other_special_token_ids: Option<Vec<usize>>,
    pub label_smoothing: f64,
    pub weights: Option<HashMap<String, f64>>,
    pub dtype: DType,
}

pub fn get_loss(device: &Device, config: &LossConfig) -> Result<CrossEntropyLoss> {
    let tokenizer = ProteinTokenizer::new(
        Path::new(&config.vocab_path),
        config.pad_token_id,
        config.mask_token_id,
        config.bos_token_id,
        config.eos_token_id,
        config.unk_token_id,
        config.other_special_token_ids.clone(),
    )?;

    // Handle class weights if provided
    let class_weights = if let Some(weights) = &config.weights {
        if weights.values().any(|&w| w != 1.0) {
            let weights_vec: Vec<f64> = (0..tokenizer.len())
                .map(|i| {
                    let token = tokenizer.id_to_token(i);
                    *weights.get(&token).unwrap_or(&1.0)
                })
                .collect();

            Some(Tensor::new(weights_vec.as_slice(), device)?.to_dtype(config.dtype)?)
        } else {
            None
        }
    } else {
        None
    };

    // Create CrossEntropyLoss configuration
    let ce_config = CrossEntropyConfig {
        reduction: candle_nn::loss::Reduction::Mean,
        ignore_index: Some(-100),
        label_smoothing: config.label_smoothing,
        weights: class_weights,
    };

    CrossEntropyLoss::new(ce_config)
}

// Example usage:
fn create_default_loss(device: &Device) -> Result<CrossEntropyLoss> {
    let config = LossConfig {
        vocab_path: "vocab.txt".to_string(),
        pad_token_id: 0,
        mask_token_id: 1,
        bos_token_id: 2,
        eos_token_id: 3,
        unk_token_id: 4,
        other_special_token_ids: None,
        label_smoothing: 0.0,
        weights: None,
        dtype: DType::F32,
    };

    get_loss(device, &config)
}
