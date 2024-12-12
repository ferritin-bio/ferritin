use super::modules::{ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};
use serde::Deserialize;
use tokenizers::Tokenizer;
#[derive(Deserialize)]

pub struct ESM2Config {
    num_attention_heads: i32,
    attention_probs_dropout_prob: f32,
    classifier_dropout: Option<f32>,
    emb_layer_norm_before: bool,
    esmfold_config: Option<String>,
    hidden_act: String,
    hidden_dropout_prob: f32,
    hidden_size: i32,
    initializer_range: f32,
    intermediate_size: i32,
    is_folding_model: bool,
    layer_norm_eps: f32,
    mask_token_id: i32,
    max_position_embeddings: i32,
    model_type: String,
    num_hidden_layers: i32,
    pad_token_id: i32,
    position_embedding_type: String,
    token_dropout: bool,
    torch_dtype: String,
    transformers_version: String,
    use_cache: bool,
    vocab_list: Option<Vec<String>>,
    vocab_size: i32,
}

impl ESM2Config {
    pub fn esm2_t36_3b_ur50() -> Self {
        Self {
            num_attention_heads: 40,
            attention_probs_dropout_prob: 0.0,
            classifier_dropout: None,
            emb_layer_norm_before: false,
            esmfold_config: None,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            hidden_size: 2560,
            initializer_range: 0.02,
            intermediate_size: 10240,
            is_folding_model: false,
            layer_norm_eps: 1e-5,
            mask_token_id: 32,
            max_position_embeddings: 1026,
            model_type: "esm".to_string(),
            num_hidden_layers: 36,
            pad_token_id: 1,
            position_embedding_type: "rotary".to_string(),
            token_dropout: true,
            torch_dtype: "float32".to_string(),
            transformers_version: "4.25.0.dev0".to_string(),
            use_cache: true,
            vocab_list: None,
            vocab_size: 33,
        }
    }
}

//   "hidden_size": 2560,
//   "initializer_range": 0.02,
//   "intermediate_size": 10240,
//   "is_folding_model": false,
//   "layer_norm_eps": 1e-05,
//   "mask_token_id": 32,
//   "max_position_embeddings": 1026,
//   "model_type": "esm",
//   "num_attention_heads": 40,
//   "num_hidden_layers": 36,
//   "pad_token_id": 1,
//   "position_embedding_type": "rotary",
//   "token_dropout": true,
//   "torch_dtype": "float32",
//   "transformers_version": "4.25.0.dev0",
//   "use_cache": true,
//   "vocab_list": null,
//   "vocab_size": 33
// }

/// ESM2 Architecture
pub struct ESM2 {
    // num_layers: i32,
    // embed_dim: i32,
    // attention_heads: i32,
    // alphabet_size: i32,
    // padding_idx: i32,
    // mask_idx: i32,
    // cls_idx: i32,
    // eos_idx: i32,
    // prepend_bos: bool,
    // append_eos: bool,
    // token_dropout: bool,
    // embed_scale: f32,
    // embed_tokens: nn::Embedding,
    // layers: Vec<TransformerLayer>,
    // contact_head: ContactPredictionHead,
    // emb_layer_norm_after: ESM1bLayerNorm,
    // lm_head: RobertaLMHead,
}

impl ESM2 {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Self {
        Self {}
    }
    // pub fn get_device(&self) -> &Device {
    //     self.freqs_cis.device()
    // }
    //
    // pub fn new(
    //     num_layers: i32,
    //     embed_dim: i32,
    //     attention_heads: i32,
    //     // alphabet: &EsmSequenceTokenizer,
    //     alphabet: &Tokenizer,
    //     token_dropout: bool,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     let alphabet_size = alphabet.vocab_size();
    //     let padding_idx = alphabet.padding_idx();
    //     let mask_idx = alphabet.mask_idx();
    //     let cls_idx = alphabet.cls_idx();
    //     let eos_idx = alphabet.eos_idx();
    //     let prepend_bos = alphabet.prepend_bos();
    //     let append_eos = alphabet.append_eos();

    //     let embed_scale = 1.0;
    //     let embed_tokens =
    //         nn::embedding(alphabet_size, embed_dim, padding_idx, vb.pp("embed_tokens"))?;

    //     let mut layers = Vec::with_capacity(num_layers as usize);
    //     for i in 0..num_layers {
    //         layers.push(TransformerLayer::new(
    //             embed_dim,
    //             4 * embed_dim,
    //             attention_heads,
    //             false,
    //             true,
    //             true,
    //             vb.pp(&format!("layers.{}", i)),
    //         )?);
    //     }

    //     let contact_head = ContactPredictionHead::new(
    //         num_layers * attention_heads,
    //         prepend_bos,
    //         append_eos,
    //         eos_idx,
    //         vb.pp("contact_head"),
    //     )?;

    //     let emb_layer_norm_after = ESM1bLayerNorm::new(embed_dim, vb.pp("emb_layer_norm_after"))?;

    //     let lm_head =
    //         RobertaLMHead::new(embed_dim, alphabet_size, &embed_tokens, vb.pp("lm_head"))?;

    //     Ok(Self {
    //         num_layers,
    //         embed_dim,
    //         attention_heads,
    //         alphabet_size,
    //         padding_idx,
    //         mask_idx,
    //         cls_idx,
    //         eos_idx,
    //         prepend_bos,
    //         append_eos,
    //         token_dropout,
    //         embed_scale,
    //         embed_tokens,
    //         layers,
    //         contact_head,
    //         emb_layer_norm_after,
    //         lm_head,
    //     })
    // }

    // fn forward(
    //     &self,
    //     tokens: &Tensor,
    //     repr_layers: &[i32],
    //     need_head_weights: bool,
    //     return_contacts: bool,
    // ) -> Result<BTreeMap<String, Tensor>> {
    //     let need_head_weights = need_head_weights || return_contacts;

    //     let padding_mask = tokens.eq(self.padding_idx)?;

    //     let mut x = self
    //         .embed_tokens
    //         .forward(tokens)?
    //         .mul_scalar(self.embed_scale)?;

    //     if self.token_dropout {
    //         let mask = tokens.eq(self.mask_idx)?.unsqueeze(-1)?;
    //         x = x.masked_fill(&mask, 0.0)?;

    //         let mask_ratio_train = 0.15 * 0.8;
    //         let src_lengths = padding_mask.logical_not()?.sum_keepdim(-1)?;
    //         let mask_ratio_observed = tokens
    //             .eq(self.mask_idx)?
    //             .sum_keepdim(-1)?
    //             .to_dtype(x.dtype())?
    //             .div(&src_lengths)?;
    //         let scale = (1.0 - mask_ratio_train) / (1.0 - mask_ratio_observed)?;
    //         x = x.mul(&scale.unsqueeze(-1)?)?;
    //     }

    //     if !padding_mask.all()? {
    //         let not_padding = padding_mask.logical_not()?.to_dtype(x.dtype())?;
    //         x = x.mul(&not_padding.unsqueeze(-1)?)?;
    //     }

    //     let repr_layers: HashSet<_> = repr_layers.iter().cloned().collect();
    //     let mut hidden_representations = BTreeMap::new();
    //     if repr_layers.contains(&0) {
    //         hidden_representations.insert("0".to_string(), x.clone());
    //     }

    //     let mut attn_weights = Vec::new();
    //     x = x.transpose(0, 1)?;

    //     let padding_mask = if !padding_mask.any()? {
    //         None
    //     } else {
    //         Some(padding_mask)
    //     };

    //     for (layer_idx, layer) in self.layers.iter().enumerate() {
    //         let (new_x, attn) = layer.forward(&x, padding_mask.as_ref(), need_head_weights)?;
    //         x = new_x;

    //         if repr_layers.contains(&(layer_idx as i32 + 1)) {
    //             hidden_representations
    //                 .insert((layer_idx + 1).to_string(), x.transpose(0, 1)?.clone());
    //         }

    //         if need_head_weights {
    //             attn_weights.push(attn.transpose(1, 0)?);
    //         }
    //     }

    //     x = self.emb_layer_norm_after.forward(&x)?;
    //     x = x.transpose(0, 1)?;

    //     if repr_layers.contains(&(self.layers.len() as i32)) {
    //         hidden_representations.insert(self.layers.len().to_string(), x.clone());
    //     }

    //     let logits = self.lm_head.forward(&x)?;

    //     let mut result = BTreeMap::new();
    //     result.insert("logits".to_string(), logits);
    //     result.insert("representations".to_string(), x);

    //     if need_head_weights {
    //         let attentions = Tensor::stack(&attn_weights, 1)?;
    //         if let Some(padding_mask) = padding_mask {
    //             let attention_mask = padding_mask.logical_not()?.to_dtype(attentions.dtype())?;
    //             let attention_mask = attention_mask
    //                 .unsqueeze(1)?
    //                 .mul(&attention_mask.unsqueeze(2)?)?;
    //             result.insert(
    //                 "attentions".to_string(),
    //                 attentions.mul(&attention_mask.unsqueeze(1)?.unsqueeze(1)?)?,
    //             );
    //         } else {
    //             result.insert("attentions".to_string(), attentions);
    //         }

    //         if return_contacts {
    //             let contacts = self.contact_head.forward(tokens, &attentions)?;
    //             result.insert("contacts".to_string(), contacts);
    //         }
    //     }

    //     Ok(result)
    // }

    // pub fn predict_contacts(&self, tokens: &Tensor) -> Result<Tensor> {
    //     let mut result = self.forward(tokens, &[], false, true)?;
    //     Ok(result.remove("contacts").unwrap())
    // }

    pub fn load_tokenizer() -> Result<Tokenizer> {
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
    }
}
