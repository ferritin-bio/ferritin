use super::modules::{ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{self as nn, Linear, VarBuilder};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Deserialize, Clone)]
pub struct ESM2Config {
    pub(crate) num_attention_heads: i32,
    pub(crate) attention_probs_dropout_prob: f32,
    pub(crate) classifier_dropout: Option<f32>,
    pub(crate) emb_layer_norm_before: bool,
    pub(crate) esmfold_config: Option<String>,
    pub(crate) hidden_act: String,
    pub(crate) hidden_dropout_prob: f32,
    pub(crate) hidden_size: i32,
    pub(crate) initializer_range: f32,
    pub(crate) intermediate_size: i32,
    pub(crate) is_folding_model: bool,
    pub(crate) layer_norm_eps: f32,
    pub(crate) mask_token_id: i32,
    pub(crate) max_position_embeddings: i32,
    pub(crate) model_type: String,
    pub(crate) num_hidden_layers: i32,
    pub(crate) pad_token_id: i32,
    pub(crate) position_embedding_type: String,
    pub(crate) token_dropout: bool,
    pub(crate) torch_dtype: String,
    pub(crate) transformers_version: String,
    pub(crate) use_cache: bool,
    pub(crate) vocab_list: Option<Vec<String>>,
    pub(crate) vocab_size: i32,
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
    pub fn esm2_t6_8M_ur50() -> Self {
        Self {
            num_attention_heads: 20,
            attention_probs_dropout_prob: 0.0,
            classifier_dropout: None,
            emb_layer_norm_before: false,
            esmfold_config: None,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            hidden_size: 320,
            initializer_range: 0.02,
            intermediate_size: 1280,
            is_folding_model: false,
            layer_norm_eps: 1e-5,
            mask_token_id: 32,
            max_position_embeddings: 1026,
            model_type: "esm".to_string(),
            num_hidden_layers: 6,
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

pub struct ModelOutput {}

/// ESM2 Architecture
pub struct ESM2 {
    embed_tokens: Option<nn::Embedding>,
    layers: Vec<TransformerLayer>,
    contact_head: ContactPredictionHead,
    emb_layer_norm_after: ESM1bLayerNorm,
    lm_head: RobertaLMHead,
    config: ESM2Config,
}

impl ESM2 {
    // note: in thisload function we do NOT handle the embedding code which gets invoked only when the model is invokes with tokens
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let ESM2Config {
            intermediate_size,
            num_hidden_layers,
            vocab_size,
            ..
        } = config;

        // num_layers: int = 33,
        // embed_dim: int = 1280,
        // attention_heads: int = 20,
        //
        // self.embed_scale = 1
        // self.embed_tokens = nn.Embedding(
        //     self.alphabet_size,
        //     self.embed_dim,
        //     padding_idx=self.padding_idx,
        // )

        let e_tensor = Tensor::zeros(
            (*vocab_size as usize, *intermediate_size as usize),
            vb.dtype(),
            vb.device(),
        )?;
        let embed_tokens = Linear::new(e_tensor, None);

        let mut layers = Vec::with_capacity(*num_hidden_layers as usize);
        for i in 0..*num_hidden_layers {
            let transformer_layer =
                TransformerLayer::load(vb.pp(format!("esm.encoder.layer.{}", i)), config)?;
            layers.push(transformer_layer);
        }
        let contact_head = ContactPredictionHead::load(vb.pp("esm.contact_head"), config)?;
        let emb_layer_norm_after =
            ESM1bLayerNorm::load(vb.pp("esm.encoder.emb_layer_norm_after"), config)?;
        let lm_head = RobertaLMHead::load(vb.pp("lm_head"), config)?;

        Ok(Self {
            embed_tokens: None,
            layers,
            contact_head,
            emb_layer_norm_after,
            lm_head,
            config: config.clone(),
        })
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

    fn forward(
        &self,
        tokens: &Tensor,
        repr_layers: &[i32],
        need_head_weights: bool,
        return_contacts: bool,
    ) -> Result<ModelOutput> {
        // let need_head_weights = need_head_weights || return_contacts;
        let padding_idx = 1_u32; // see tokenizer.json
        let padding_mask = tokens.eq(padding_idx)?;

        let mut x = self
            .embed_tokens
            .forward(tokens)?
            .mul_scalar(self.embed_scale)?;

        // if self.token_dropout {
        //     let mask = tokens.eq(self.mask_idx)?.unsqueeze(-1)?;
        //     x = x.masked_fill(&mask, 0.0)?;

        //     let mask_ratio_train = 0.15 * 0.8;
        //     let src_lengths = padding_mask.logical_not()?.sum_keepdim(-1)?;
        //     let mask_ratio_observed = tokens
        //         .eq(self.mask_idx)?
        //         .sum_keepdim(-1)?
        //         .to_dtype(x.dtype())?
        //         .div(&src_lengths)?;
        //     let scale = (1.0 - mask_ratio_train) / (1.0 - mask_ratio_observed)?;
        //     x = x.mul(&scale.unsqueeze(-1)?)?;
        // }

        // if !padding_mask.all()? {
        //     let not_padding = padding_mask.logical_not()?.to_dtype(x.dtype())?;
        //     x = x.mul(&not_padding.unsqueeze(-1)?)?;
        // }

        // let repr_layers: HashSet<_> = repr_layers.iter().cloned().collect();
        // let mut hidden_representations = BTreeMap::new();
        // if repr_layers.contains(&0) {
        //     hidden_representations.insert("0".to_string(), x.clone());
        // }

        // let mut attn_weights = Vec::new();
        // x = x.transpose(0, 1)?;

        // let padding_mask = if !padding_mask.any()? {
        //     None
        // } else {
        //     Some(padding_mask)
        // };

        // for (layer_idx, layer) in self.layers.iter().enumerate() {
        //     let (new_x, attn) = layer.forward(&x, padding_mask.as_ref(), need_head_weights)?;
        //     x = new_x;

        //     if repr_layers.contains(&(layer_idx as i32 + 1)) {
        //         hidden_representations
        //             .insert((layer_idx + 1).to_string(), x.transpose(0, 1)?.clone());
        //     }

        //     if need_head_weights {
        //         attn_weights.push(attn.transpose(1, 0)?);
        //     }
        // }

        // x = self.emb_layer_norm_after.forward(&x)?;
        // x = x.transpose(0, 1)?;

        // if repr_layers.contains(&(self.layers.len() as i32)) {
        //     hidden_representations.insert(self.layers.len().to_string(), x.clone());
        // }

        // let logits = self.lm_head.forward(&x)?;

        // let mut result = BTreeMap::new();
        // result.insert("logits".to_string(), logits);
        // result.insert("representations".to_string(), x);

        // if need_head_weights {
        //     let attentions = Tensor::stack(&attn_weights, 1)?;
        //     if let Some(padding_mask) = padding_mask {
        //         let attention_mask = padding_mask.logical_not()?.to_dtype(attentions.dtype())?;
        //         let attention_mask = attention_mask
        //             .unsqueeze(1)?
        //             .mul(&attention_mask.unsqueeze(2)?)?;
        //         result.insert(
        //             "attentions".to_string(),
        //             attentions.mul(&attention_mask.unsqueeze(1)?.unsqueeze(1)?)?,
        //         );
        //     } else {
        //         result.insert("attentions".to_string(), attentions);
        //     }

        //     if return_contacts {
        //         let contacts = self.contact_head.forward(tokens, &attentions)?;
        //         result.insert("contacts".to_string(), contacts);
        //     }
        // }

        // Ok(result)
        Ok(ModelOutput {})
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer_load() -> Result<()> {
        let tokenizer = ESM2::load_tokenizer()?;
        let text = "MLKLRV";
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to encode: {}", e)))?;
        let tokens = encoding.get_tokens();
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens, &["M", "L", "K", "L", "R", "V"]);
        Ok(())
    }
}
