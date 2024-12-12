use crate::esm::layers::regression_head::RegressionHead;
use crate::esm::layers::transformer_stack::TransformerStack;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};
// use crate::esm::pretrained::load_local_model;
// use crate::esm::sdk::api::ESMProtein;
// use crate::esm::sdk::api::ESMProteinTensor;
// use crate::esm::sdk::api::ForwardTrackData;
// use crate::esm::sdk::api::LogitsConfig;
// use crate::esm::sdk::api::LogitsOutput;
use crate::esm::tokenization::sequence_tokenizer::EsmSequenceTokenizer;
use crate::esm::tokenization::TokenizerCollection;
// use crate::esm::utils::decoding::decode_sequence;
// use crate::esm::utils::encoding::tokenize_sequence;
// use crate::esm::utils::sampling::BatchedESMProteinTensor;

pub fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer_bytes = include_bytes!("tokenizer.json");
    Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
}
