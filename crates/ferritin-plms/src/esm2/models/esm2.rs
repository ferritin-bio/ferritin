use super::modules::{ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer};
use candle_core::{Result, Tensor};
use candle_nn::{self as nn, VarBuilder};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use tokenizers::Tokenizer;

#[derive(Deserialize, Clone)]
pub struct ESM2Config {
    pub num_attention_heads: i32,
    pub attention_probs_dropout_prob: f32,
    pub classifier_dropout: Option<f32>,
    pub emb_layer_norm_before: bool,
    pub esmfold_config: Option<String>,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub hidden_size: i32,
    pub initializer_range: f32,
    pub intermediate_size: i32,
    pub is_folding_model: bool,
    pub layer_norm_eps: f32,
    pub mask_token_id: i32,
    pub max_position_embeddings: i32,
    pub model_type: String,
    pub num_hidden_layers: i32,
    pub pad_token_id: i32,
    pub position_embedding_type: String,
    pub token_dropout: bool,
    pub torch_dtype: String,
    pub transformers_version: String,
    pub use_cache: bool,
    pub vocab_list: Option<Vec<String>>,
    pub vocab_size: i32,
}

impl ESM2Config {
    fn base_config() -> Self {
        Self {
            attention_probs_dropout_prob: 0.0,
            classifier_dropout: None,
            emb_layer_norm_before: false,
            esmfold_config: None,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            initializer_range: 0.02,
            is_folding_model: false,
            layer_norm_eps: 1e-5,
            mask_token_id: 32,
            max_position_embeddings: 1026,
            model_type: "esm".to_string(),
            pad_token_id: 1,
            position_embedding_type: "rotary".to_string(),
            token_dropout: true,
            torch_dtype: "float32".to_string(),
            transformers_version: "4.25.0.dev0".to_string(),
            use_cache: true,
            vocab_list: None,
            vocab_size: 33,
            num_attention_heads: 0, // placeholder
            hidden_size: 0,
            intermediate_size: 0,
            num_hidden_layers: 0,
            // pub prepend_bos: bool,
            // pub append_eos: bool,
            // pub cls_idx: i64,
            // pub eos_idx: i64,
        }
    }

    pub fn esm2_t36_3b_ur50() -> Self {
        Self {
            num_attention_heads: 40,
            hidden_size: 2560,
            intermediate_size: 10240,
            num_hidden_layers: 36,
            ..Self::base_config()
        }
    }

    pub fn esm2_t6_8M_ur50() -> Self {
        Self {
            num_attention_heads: 20,
            hidden_size: 320,
            intermediate_size: 1280,
            num_hidden_layers: 6,
            ..Self::base_config()
        }
    }
}

/// ESM2 Architecture
pub struct ESM2 {
    // Token embedding layer
    embed_tokens: Option<nn::Embedding>,

    // Transformer layers
    layers: Vec<TransformerLayer>,

    // Head components
    contact_head: ContactPredictionHead,
    emb_layer_norm_after: ESM1bLayerNorm,
    lm_head: RobertaLMHead,

    // Model configuration
    config: ESM2Config,
}

impl ESM2 {
    fn padding_idx(&self) -> i64 {
        self.config.pad_token_id as i64
    }
    fn mask_idx(&self) -> i64 {
        self.config.mask_token_id as i64
    }
    fn token_dropout(&self) -> bool {
        self.config.token_dropout
    }
    // note: in thisload function we do NOT handle the embedding code
    // which gets invoked only when the model is invoked with tokens
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
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

    fn forward(
        &self,
        tokens: &Tensor,
        repr_layers: &[i32],
        need_head_weights: bool,
        return_contacts: bool,
    ) -> Result<BTreeMap<String, Tensor>> {
        let need_head_weights = need_head_weights || return_contacts;
        let padding_mask = tokens.eq(self.padding_idx())?;
        let mut x = self
            .embed_tokens?
            .forward(tokens)?
            .mul_scalar(self.embed_scale)?;
        if self.token_dropout() {
            // let mask = tokens.eq(self.mask_idx)?.unsqueeze(-1)?;
            let mask = tokens.eq(self.mask_idx())?.unsqueeze(-1)?;
            x = x.masked_fill(&mask, 0.0)?;
            let mask_ratio_train = 0.15 * 0.8;
            let src_lengths = padding_mask.logical_not()?.sum_keepdim(-1)?;
            let mask_ratio_observed = tokens
                .eq(self.mask_idx())?
                .sum_keepdim(-1)?
                .to_dtype(x.dtype())?
                .div(&src_lengths)?;
            let scale = (1.0 - mask_ratio_train) / (1.0 - mask_ratio_observed);
            x = x.mul(&scale.unsqueeze(-1)?)?;
        }
        if padding_mask.any()? {
            x = x.masked_fill(&padding_mask.unsqueeze(-1)?, 0.0)?;
        }
        let repr_layers: HashSet<_> = repr_layers.iter().cloned().collect();
        let mut hidden_representations = BTreeMap::new();
        if repr_layers.contains(&0) {
            hidden_representations.insert("0".to_string(), x.clone());
        }

        let mut attn_weights = Vec::new();
        x = x.transpose(0, 1)?;

        let padding_mask = if padding_mask.any()? {
            Some(padding_mask)
        } else {
            None
        };
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (new_x, attn) = layer.forward(&x, padding_mask.as_ref(), need_head_weights)?;
            x = new_x;

            let repr_index = layer_idx + 1;
            if repr_layers.contains(&(repr_index as i32)) {
                hidden_representations
                    .insert(format!("{}", repr_index), x.transpose(0, 1)?.clone());
            }
            if need_head_weights {
                attn_weights.push(attn.transpose(1, 0)?);
            }
        }
        x = self.emb_layer_norm_after.forward(&x)?;
        x = x.transpose(0, 1)?;
        if repr_layers.contains(&(self.layers.len() as i32)) {
            hidden_representations.insert(self.layers.len().to_string(), x.clone());
        }
        let logits = self.lm_head.forward(&x)?;
        let mut result = BTreeMap::new();
        // Generate a dummy tensor for the results
        let dummy_tensor = Tensor::zeros(&[1, 1], DType::F32, &self.device)?;
        result.insert("logits".to_string(), dummy_tensor.clone());
        result.insert("representations".to_string(), dummy_tensor);
        if need_head_weights && return_contacts {
            result.insert(
                "contacts".to_string(),
                Tensor::zeros(&[1, 1], DType::F32, &self.device)?,
            );
        }
        Ok(result)
    }

    /// Predict protein contacts from input tokens
    pub fn predict_contacts(&self, tokens: &Tensor) -> Result<Tensor> {
        let mut result = self.forward(tokens, &[], false, true)?;
        result.remove("contacts").ok_or_else(|| {
            candle_core::Error::Msg("Contacts not found in model output".to_string())
        })
    }
    /// Initialize embedding tokens that are set to None during model loading
    pub fn init_embed_tokens(&mut self, vb: VarBuilder) -> Result<()> {
        if self.embed_tokens.is_none() {
            let embedding = nn::embedding(
                self.config.vocab_size as usize,
                self.config.hidden_size as usize,
                vb.pp("esm.embeddings.word_embeddings"),
            )?;
            self.embed_tokens = Some(embedding);
        }
        Ok(())
    }
    /// Load the tokenizer for encoding sequences
    pub fn load_tokenizer() -> Result<Tokenizer> {
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
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
