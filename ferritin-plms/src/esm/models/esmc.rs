use crate::esm::layers::regression_head::RegressionHead;
use crate::esm::layers::transformer_stack::TransformerStack;
// use crate::esm::pretrained::load_local_model;
// use crate::esm::sdk::api::ESMProtein;
// use crate::esm::sdk::api::ESMProteinTensor;
// use crate::esm::sdk::api::ForwardTrackData;
// use crate::esm::sdk::api::LogitsConfig;
// use crate::esm::sdk::api::LogitsOutput;
use crate::esm::tokenization::sequence_tokenizer::EsmSequenceTokenizer;
use crate::esm::tokenization::TokenizerCollection;
use candle_core::{Result, Tensor};
use candle_nn::{self as nn, VarBuilder};
// use crate::esm::utils::decoding::decode_sequence;
// use crate::esm::utils::encoding::tokenize_sequence;
// use crate::esm::utils::sampling::BatchedESMProteinTensor;

#[derive(Debug)]
struct ESMCOutput {
    sequence_logits: Tensor,
    embeddings: Option<Tensor>,
}

#[derive(Clone, Copy)]
pub enum ESMTokenizer {
    Esm3OpenSmall,
}
impl ESMTokenizer {
    pub fn get_model_tokenizers(&self) -> TokenizerCollection {
        match self {
            ESMTokenizer::Esm3OpenSmall => {
                let esm_tokenizer = EsmSequenceTokenizer::default();
                TokenizerCollection {
                    sequence: esm_tokenizer,
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Ffn_Type {
    SWIGLU,
    GLU,
}

#[derive(Clone)]
pub struct ESMCConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub v_head_transformer: Option<usize>,
    pub ffn_type: Ffn_Type,
    pub tokenizer: ESMTokenizer,
    // oringal above.
    pub use_plain_attn: bool,
    pub n_layers_geom: usize,
    pub scale_residue: bool,
    pub residue_scaling_factor: f64,
    pub mask_and_zero_frameless: bool,
    pub bias: bool,
    pub qk_layernorm: bool,
    pub expansion_ratio: f64,
    // reg
    pub regression_head_output_dim: usize,
    pub regression_head_hidden_dim: usize,
    pub embedding_dim: usize,
}

impl ESMCConfig {
    pub fn esmc_300m() -> Self {
        //
        //    residue_scaling_factor=  if scale_residue {
        //         (n_layers as f64 / 36.0).sqrt()
        //     } else {
        //         1.0
        //     },

        Self {
            d_model: 960,
            n_heads: 15,
            n_layers: 30,
            v_head_transformer: None,
            ffn_type: Ffn_Type::SWIGLU,
            tokenizer: ESMTokenizer::Esm3OpenSmall,
            use_plain_attn: true,
            n_layers_geom: 1,
            scale_residue: true,
            residue_scaling_factor: (30f64 / 36.).sqrt(),
            mask_and_zero_frameless: false,
            bias: false,
            qk_layernorm: true,
            expansion_ratio: 8.0 / 3.0,
            regression_head_output_dim: 64,
            regression_head_hidden_dim: 960, // d_model
            embedding_dim: 64,
        }
    }
}

pub struct ESMC {
    embed: candle_nn::Embedding,
    transformer: TransformerStack,
    sequence_head: RegressionHead,
    tokenizer: EsmSequenceTokenizer,
}

impl ESMC {
    // pub fn new(
    //     d_model: usize,
    //     n_heads: usize,
    //     n_layers: usize,
    //     tokenizer: EsmSequenceTokenizer,
    // ) -> Self {
    //     Self {
    //         embed: nn::embedding(64, d_model, Default::default())?,
    //         transformer: TransformerStack::new(d_model, n_heads, None, n_layers, 0)?,
    //         sequence_head: RegressionHead::new(d_model, 64)?,
    //         tokenizer,
    //     }
    // }

    pub fn load(vb: VarBuilder, config: ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            tokenizer,
            embedding_dim,
            ..
        } = config;

        let tokenizer_collection = tokenizer.get_model_tokenizers();

        Ok(Self {
            embed: nn::embedding(embedding_dim, d_model, vb.pp("embed"))?,
            transformer: TransformerStack::load(vb.pp("transformer"), &config)?,
            sequence_head: RegressionHead::load(vb.pp("sequence_head"), &config)?,
            tokenizer: tokenizer_collection.sequence,
        })
    }

    // pub fn from_pretrained(model_name: impl Into<String>, device: Option<Device>) -> Result<Self> {
    //     let device = device.unwrap_or(Device::cuda_if_available()?);
    //     let model = load_local_model(&model_name.into(), &device)?;
    //     if device.is_cuda() {
    //         model.to_dtype(DType::BF16)?;
    //     }
    //     Ok(model)
    // }

    // pub fn forward(
    //     &self,
    //     sequence_tokens: Option<&Tensor>,
    //     sequence_id: Option<&Tensor>,
    // ) -> Result<ESMCOutput> {
    //     let sequence_id = sequence_id
    //         .unwrap_or({ &(sequence_tokens.unwrap().eq(self.tokenizer.pad_token_id)?)? });

    //     let x = self.embed.forward(sequence_tokens.unwrap())?;
    //     let (x, _) = self.transformer.forward(&x, Some(sequence_id))?;
    //     let sequence_logits = self.sequence_head.forward(&x)?;

    //     Ok(ESMCOutput {
    //         sequence_logits,
    //         embeddings: Some(x),
    //     })
    // }

    // pub fn encode(&self, input: &ESMProtein) -> Result<ESMProteinTensor> {
    //     let sequence_tokens = if let Some(seq) = &input.sequence {
    //         Some(tokenize_sequence(seq, &self.tokenizer, true)?)
    //     } else {
    //         None
    //     };

    //     Ok(ESMProteinTensor::new(sequence_tokens)?.to_device(&self.device())?)
    // }

    // pub fn decode(&self, input: &ESMProteinTensor) -> Result<ESMProtein> {
    //     let sequence = input.sequence.as_ref().ok_or("Missing sequence")?;
    //     let sequence = decode_sequence(&sequence.slice(1..-1)?, &self.tokenizer)?;
    //     Ok(ESMProtein::new(Some(sequence)))
    // }

    // pub fn logits(&self, input: &ESMProteinTensor, config: &LogitsConfig) -> Result<LogitsOutput> {
    //     let input = if !input.is_batched() {
    //         BatchedESMProteinTensor::from_protein_tensor(input)?
    //     } else {
    //         input.clone()
    //     };

    //     candle_core::no_grad(|| {
    //         let output = self.forward(Some(&input.sequence), None)?;

    //         Ok(LogitsOutput {
    //             logits: ForwardTrackData {
    //                 sequence: if config.sequence {
    //                     Some(output.sequence_logits)
    //                 } else {
    //                     None
    //                 },
    //             },
    //             embeddings: if config.return_embeddings {
    //                 output.embeddings
    //             } else {
    //                 None
    //             },
    //         })
    //     })
    // }
}
