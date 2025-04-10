// ESM2 Model Architecture
// ======================

//              ┌─────────────────┐
//              │  Input Tokens   │
//              └────────┬────────┘
//                       │
//              ┌────────▼────────┐
//              │  Token Embedding│
//              └────────┬────────┘
//                       │
//           ┌───────────▼───────────┐
//           │   Token Dropout       │ (conditional)
//           └───────────┬───────────┘
//                       │
// ┌─────────────────────▼─────────────────────┐
// │            Transformer Layers              │
// │  ┌─────────────────────────────────────┐  │
// │  │ Layer 1                             │  │
// │  │  ├─ Multi-head Self-Attention       │  │
// │  │  │   (with rotary embeddings)       │  │
// │  │  └─ Feed Forward Network            │  │
// │  └─────────────────────────────────────┘  │
// │                     ⋮                      │
// │  ┌─────────────────────────────────────┐  │
// │  │ Layer N (33 by default)             │  │
// │  │  ├─ Multi-head Self-Attention       │  │
// │  │  │   (with rotary embeddings)       │  │
// │  │  └─ Feed Forward Network            │  │
// │  └─────────────────────────────────────┘  │
// └─────────────────────┬─────────────────────┘
//                       │
//              ┌────────▼────────┐
//              │ Layer Norm After│
//              └───┬───────────┬─┘
//                  │           │
//         ┌────────▼─────┐    │    ┌─────────────────┐
//         │   LM Head    │    └────▶ Contact Head    │
//         │ (predictions)│         │ (if requested)  │
//         └──────────────┘         └─────────────────┘

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, VarBuilder, linear};
use serde::Deserialize;
use tokenizers::Tokenizer;

// for embeddings
const MAX_SEQ_LEN: usize = 5000;

#[derive(Deserialize, Clone)]
pub struct ESM2Config {
    pub num_attention_heads: usize,
    pub attention_probs_dropout_prob: f32,
    pub classifier_dropout: Option<f32>,
    pub emb_layer_norm_before: bool,
    pub esmfold_config: Option<String>,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub hidden_size: usize,
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
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let l = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, l / 2)?;
    let x2 = x.narrow(D::Minus1, l / 2, l - l / 2)?;
    let x21 = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    Ok(x21)
}

#[derive(Debug, Clone)]
struct FalconRotaryEmbedding {
    inv_freq: Tensor,
    cache: Option<(usize, Tensor, Tensor)>,
}

impl FalconRotaryEmbedding {
    fn load(device: &Device, cfg: &ESM2Config) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let inv_freq: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv_freq.as_slice(), device)?,
            cache: None,
        })
    }

    fn cos_sin(
        &mut self,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        match &self.cache {
            Some((s, cos, sin)) if *s == seq_len => {
                return Ok((cos.clone(), sin.clone()));
            }
            _ => {}
        }
        let t = Tensor::arange(0, seq_len as u32, device)?.to_dtype(dtype)?;
        let inv_freq = self.inv_freq.to_dtype(dtype)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        self.cache = Some((seq_len, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }

    fn forward(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        past_kv_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_batch, seq_len, _head_dim) = query.dims3()?;
        let (cos, sin) = self.cos_sin(MAX_SEQ_LEN, query.device(), query.dtype())?;
        let cos = cos.narrow(0, past_kv_len, seq_len)?;
        let sin = sin.narrow(0, past_kv_len, seq_len)?;
        let qs = (query.broadcast_mul(&cos)? + &rotate_half(query)?.broadcast_mul(&sin)?)?;
        let ks = (key.broadcast_mul(&cos)? + &rotate_half(key)?.broadcast_mul(&sin)?)?;
        Ok((qs, ks))
    }
}

pub struct ESM2Embeddings {
    token_embeddings: Embedding,
    embed_scale: f64,
}

pub struct ESM2LMHead {
    dense: Linear,
    layer_norm: ESM1bLayerNorm,
    decoder: Linear, // Weight tied to embeddings
}
impl ESM2LMHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        todo!()
    }
    fn forward() {
        todo!()
    }
}


pub struct ESM2ContactHead {
    contact_scale: Tensor,
    feedforward: Linear,
    prepend_bos: bool,
    append_eos: bool,
    eos_idx: usize,
}
impl ESM2ContactHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        todo!()
    }
    fn forward() {
        todo!()
    }
}


// Attention module with rotary embeddings
pub struct ESM2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    rotary_emb: Option<FalconRotaryEmbedding>,
}
impl ESM2Attention {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        todo!()
    }
    fn forward() {
        todo!()
    }
}



// Feed-forward network
pub struct ESM2FeedForward {
    fc1: Linear,
    fc2: Linear,
    layer_norm: ESM1bLayerNorm,
}
impl ESM2FeedForward {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        todo!()
    }
    fn forward() {
        todo!()
    }
}

// ESM1b style layer norm
pub struct ESM1bLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}
impl ESM1bLayerNorm {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        todo!()
    }
    fn forward() {
        todo!()
    }
}

// Full transformer layer
pub struct ESM2Layer {
    self_attn: ESM2Attention,
    self_attn_layer_norm: ESM1bLayerNorm,
    feed_forward: ESM2FeedForward,
    final_layer_norm: ESM1bLayerNorm,
}

impl ESM2Layer {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let embed_dim = 100;
        let ffn_embed_dim = 100;
        let layer_norm = ESM1bLayerNorm::load(vb.pp("Layer_Norm"), config)?;
        let multi_head = ESM2Attention::load(vb.pp("attention"), config)?;
        let ff = ESM2FeedForward {
            fc1: linear(embed_dim, ffn_embed_dim, vb.pp("fc1"))?,
            fc2: linear(ffn_embed_dim, embed_dim, vb.pp("fc2"))?,
            layer_norm: ESM1bLayerNorm::load(vb.pp("Layer_Norm"), config)?,
        }
        let final_layer_norm = ESM1bLayerNorm::load(vb.pp("LayerNorm"), config)?;

        Ok(Self {
            self_attn: multi_head,
            self_attn_layer_norm: layer_norm,
            feed_forward: ff,
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        self_attn_padding_mask: Option<&Tensor>,
        need_head_weights: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Implementation would include:
        // 1. Self-attention with rotary embeddings
        // 2. Add & norm
        // 3. Feed-forward
        // 4. Add & norm
        // 5. Return output tensor and optional attention weights

        // Would follow candle_transformers patterns for transformer layers
        todo!()
    }
}

// Main model struct
pub struct ESM2 {
    config: ESM2Config,
    // embeddings: ESM2Embeddings,
    layers: Vec<ESM2Layer>,
    layer_norm_after: ESM1bLayerNorm,
    lm_head: ESM2LMHead,
    contact_head: ESM2ContactHead,
}

impl ESM2 {
    pub fn new(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        // Create embeddings, layers, and heads
        todo!()
    }
    // note: in this load function we do NOT handle the embedding code
    // which gets invoked only when the model is invoked with tokens
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|i| ESM2Layer::load(vb.pp(format!("esm.encoder.layer.{}", i)), config))
            .collect::<Result<Vec<_>>>()?;

        let contact_head = ESM2ContactHead::load(vb.pp("esm.contact_head"), config)?;
        let layer_norm_after =
            ESM1bLayerNorm::load(vb.pp("esm.encoder.emb_layer_norm_after"), config)?;
        let lm_head = ESM2LMHead::load(vb.pp("lm_head"), config)?;

        Ok(Self {
            config: config.clone(),
            contact_head,
            // embeddings: None,
            layer_norm_after,
            layers,
            lm_head,

        })
    }
    // // Helper methods for predictions
    // pub fn predict_contacts(&self, tokens: &Tensor) -> Result<Tensor> {
    //     let output = self.forward(tokens, &[], true, true)?;
    //     Ok(output.contacts.unwrap())
    // }
    pub fn forward(
        &self,
        tokens: &Tensor,
        repr_layers: &[usize],
        need_head_weights: bool,
        return_contacts: bool,
    ) -> Result<ESM2Output> {
        // Implementation would:
        // 1. Create padding mask
        // 2. Apply token embeddings
        // 3. Apply token dropout if enabled
        // 4. Process through transformer layers
        // 5. Collect representations from specified layers
        // 6. Apply final layer norm
        // 7. Generate outputs through LM head
        // 8. Calculate contacts if requested
        // ...
        todo!()
    }
    /// Load the tokenizer for encoding sequences
    pub fn load_tokenizer() -> Result<Tokenizer> {
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
    }
}

// Output struct
pub struct ESM2Output {
    pub logits: Tensor,
    pub representations: std::collections::HashMap<usize, Tensor>,
    pub attentions: Option<Tensor>,
    pub contacts: Option<Tensor>,
}
