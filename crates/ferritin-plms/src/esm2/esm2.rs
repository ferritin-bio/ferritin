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
use candle_nn::{Embedding, Linear, VarBuilder, linear, ops};
use serde::Deserialize;
use tokenizers::Tokenizer;

// for embeddings
const MAX_SEQ_LEN: usize = 10000;

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
        }
    }

    pub fn t6_8m() -> Self {
        Self {
            num_attention_heads: 20,
            hidden_size: 320,
            intermediate_size: 1280,
            num_hidden_layers: 6,
            ..Self::base_config()
        }
    }

    pub fn t12_35m() -> Self {
        Self {
            num_attention_heads: 20,
            hidden_size: 480,
            intermediate_size: 1920,
            num_hidden_layers: 12,
            ..Self::base_config()
        }
    }

    pub fn t30_150m() -> Self {
        Self {
            num_attention_heads: 20,
            hidden_size: 640,
            intermediate_size: 2560,
            num_hidden_layers: 30,
            ..Self::base_config()
        }
    }

    pub fn t33_650m() -> Self {
        Self {
            num_attention_heads: 20,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 33,
            ..Self::base_config()
        }
    }

    pub fn t36_3b() -> Self {
        Self {
            num_attention_heads: 40,
            hidden_size: 2560,
            intermediate_size: 10240,
            num_hidden_layers: 36,
            ..Self::base_config()
        }
    }

    pub fn t48_15b() -> Self {
        Self {
            num_attention_heads: 40,
            hidden_size: 5120,
            intermediate_size: 20480,
            num_hidden_layers: 48,
            ..Self::base_config()
        }
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub(crate) fn inv_freq_size(&self) -> usize {
        self.head_dim() / 2
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
struct RotaryEmbedding {
    inv_freq: Tensor,
    max_seq_len: usize,
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let inv_freq = vb.get(config.inv_freq_size(), "inv_freq")?;
        let (cos_cache, sin_cache) =
            Self::precompute_freqs_cis(&inv_freq, MAX_SEQ_LEN, vb.device(), vb.dtype())?;
        Ok(RotaryEmbedding {
            max_seq_len: MAX_SEQ_LEN,
            inv_freq,
            cos_cache,
            sin_cache,
        })
    }
    // Pre-compute rotary embeddings during initialization
    fn precompute_freqs_cis(
        inv_freq: &Tensor,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor)> {
        let t = Tensor::arange(0, max_seq_len as u32, device)?.to_dtype(dtype)?;
        let inv_freq = inv_freq.to_dtype(dtype)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos_cache = emb.cos()?;
        let sin_cache = emb.sin()?;
        Ok((cos_cache, sin_cache))
    }
    fn forward(&self, query: &Tensor, key: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_batch, seq_len, _head_dim) = query.dims3()?;

        // Get the appropriate positional embeddings for this sequence length
        let cos = self.cos_cache.narrow(0, 0, seq_len)?;
        let sin = self.sin_cache.narrow(0, 0, seq_len)?;

        // Reshape to match the dimensions of query and key tensors
        let cos = cos.unsqueeze(0)?; // Shape becomes (1, seq_len, head_dim)
        let sin = sin.unsqueeze(0)?; // Shape becomes (1, seq_len, head_dim)

        // Apply rotary embeddings to query
        let query_rot = rotate_half(query)?;
        let query_rotated = query
            .broadcast_mul(&cos)?
            .add(&query_rot.broadcast_mul(&sin)?)?;

        // Apply rotary embeddings to key
        let key_rot = rotate_half(key)?;
        let key_rotated = key
            .broadcast_mul(&cos)?
            .add(&key_rot.broadcast_mul(&sin)?)?;

        Ok((query_rotated, key_rotated))
    }
}

pub struct ESM2Embeddings {
    pub(crate) word_embeddings: Embedding,
    embedding_scale: f64,
}
impl ESM2Embeddings {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let vocab_size = config.vocab_size as usize;
        let hidden_size = config.hidden_size;
        let word_embeddings = vb.get((vocab_size, hidden_size), "word_embeddings.weight")?;
        Ok(Self {
            word_embeddings: Embedding::new(word_embeddings, hidden_size),
            embedding_scale: (hidden_size as f64).sqrt(),
        })
    }
    pub fn embed_tokens(&self, x: &Tensor) -> Result<Tensor> {
        self.word_embeddings.forward(x)
    }
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens(input_ids)? * self.embedding_scale
    }
}

use candle_nn::LayerNorm;

pub struct ESM2LMHead {
    dense: Linear,
    layer_norm: LayerNorm,
    decoder: Linear,
}
impl ESM2LMHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config, embedding: &Embedding) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let bias = vb.get(config.vocab_size as usize, "bias")?;

        // Use candle_nn::LayerNorm
        let layer_norm = candle_nn::layer_norm(hidden_size, config.layer_norm_eps as f64, vb.pp("layer_norm"))?;

        let decoder = Linear::new(embedding.embeddings().clone(), Some(bias));
        Ok(ESM2LMHead {
            dense,
            layer_norm,
            decoder,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Apply dense layer and GELU activation
        let hidden = xs.apply(&self.dense)?.gelu()?;

        // Apply standard layer norm
        let normalized = hidden.apply(&self.layer_norm)?;

        // Apply decoder (projection to vocabulary)
        normalized.apply(&self.decoder)
    }
}

pub struct ESM2ContactHead {
    // contact_scale: Tensor,
    feedforward: Linear,
    prepend_bos: bool,
    append_eos: bool,
    eos_idx: usize,
}
impl ESM2ContactHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let in_features = config.num_hidden_layers as usize * config.num_attention_heads;
        Ok(ESM2ContactHead {
            // contact_scale: Tensor,
            feedforward: linear(in_features, 1, vb.pp("regression"))?,
            prepend_bos: false,
            append_eos: false,
            eos_idx: 32,
        })
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
    rotary_emb: RotaryEmbedding,
}
impl ESM2Attention {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let embed_dim = hidden_size;
        let num_heads = num_attention_heads;
        let head_dim = embed_dim / num_heads;
        let kdim = hidden_size;
        let vdim = hidden_size;
        let q_proj = linear(embed_dim, embed_dim, vb.pp("self.query"))?;
        let k_proj = linear(kdim, embed_dim, vb.pp("self.key"))?;
        let v_proj = linear(vdim, embed_dim, vb.pp("self.value"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("output.dense"))?;
        let rotary_emb = RotaryEmbedding::load(vb.pp("self.rotary_embeddings"), config)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            rotary_emb,
        })
    }
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (seq_len, batch_size, embed_dim) = query.dims3()?;

        // Project inputs to queries, keys, values
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Reshape for multi-head attention: (seq_len, batch, heads*dim) -> (batch*heads, seq_len, dim)
        let q = q
            .reshape((seq_len, batch_size * self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((seq_len, batch_size * self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((seq_len, batch_size * self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;

        // Apply rotary position embeddings
        let (q, k) = self.rotary_emb.forward(&q, &k)?;

        // Calculate attention scores with proper scaling
        // todo: review for parity with ESM2-py
        let scale = (self.head_dim as f64).powf(-0.5);
        let attention_scores = (q.matmul(&k.transpose(1, 2)?)? * scale)?;

        // Apply softmax to get attention weights
        let attention_weights = ops::softmax_last_dim(&attention_scores)?;

        // Apply attention weights to values
        let attn_output = attention_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((seq_len, batch_size, embed_dim))?;
        let output = self.out_proj.forward(&attn_output)?;
        Ok((output, None))
    }
}


// Full transformer layer
pub struct ESM2Layer {
    self_attn: ESM2Attention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl ESM2Layer {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let ffn_embed_dim = config.intermediate_size as usize;
        let multi_head = ESM2Attention::load(vb.pp("attention"), config)?;
        let fc1 = linear(embed_dim, ffn_embed_dim, vb.pp("intermediate.dense"))?;
        let fc2 = linear(ffn_embed_dim, embed_dim, vb.pp("output.dense"))?;
        let self_attn_layer_norm = candle_nn::layer_norm(embed_dim, config.layer_norm_eps as f64, vb.pp("attention.LayerNorm"))?;
        let final_layer_norm = candle_nn::layer_norm(embed_dim, config.layer_norm_eps as f64, vb.pp("LayerNorm"))?;

        Ok(Self {
            self_attn: multi_head,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        // Input: [seq_len, batch_size, embed_dim]
        // Apply layer norm and then attention
        let norm_x = xs.apply(&self.self_attn_layer_norm)?;
        let (attn_out, attn) = self.self_attn.forward(&norm_x, &norm_x, &norm_x)?;

        // Residual connection
        let x = (attn_out + xs)?;

        // Apply layer norm and then feed-forward network
        let norm_x2 = x.apply(&self.final_layer_norm)?;
        let ffn_out = norm_x2.apply(&self.fc1)?.gelu()?.apply(&self.fc2)?;

        // Another residual connection
        Ok(((ffn_out + x)?, attn))
    }
}

// Main model struct
pub struct ESM2 {
    embeddings: ESM2Embeddings,
    layers: Vec<ESM2Layer>,
    layer_norm_after: LayerNorm,
    lm_head: ESM2LMHead,
    contact_head: ESM2ContactHead,
}

impl ESM2 {
    pub fn load(vb: VarBuilder, config: ESM2Config) -> Result<Self> {
        let embeddings = ESM2Embeddings::load(vb.pp("esm.embeddings"), &config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|i| ESM2Layer::load(vb.pp(format!("esm.encoder.layer.{}", i)), &config))
            .collect::<Result<Vec<_>>>()?;
        let contact_head = ESM2ContactHead::load(vb.pp("esm.contact_head"), &config)?;

        // Use candle_nn::LayerNorm for final layer norm
        let layer_norm_after = candle_nn::layer_norm(config.hidden_size, config.layer_norm_eps as f64, vb.pp("esm.encoder.emb_layer_norm_after"))?;

        let lm_head = ESM2LMHead::load(
            vb.pp("lm_head"),
            &config,
            &embeddings.word_embeddings.clone(),
        )?;

        Ok(Self {
            embeddings,
            contact_head,
            layer_norm_after,
            layers,
            lm_head,
        })
    }
    /// Load the tokenizer for encoding sequences
    pub fn load_tokenizer() -> Result<Tokenizer> {
        let tokenizer_bytes = include_bytes!("tokenizer.json");
        Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {}", e)))
    }
    pub fn forward(&self, x: &Tensor) -> Result<ESM2Output> {
        // x = self.embed_scale * self.embed_tokens(tokens)
        let mut xs = self.embeddings.forward(x)?;

        // Transpose to sequence-first format for transformer processing
        xs = xs.transpose(0, 1)?; // (B, T, E) -> (T, B, E)

        // Process through transformer layers
        for (_layer_idx, layer) in self.layers.iter().enumerate() {
            let (new_xs, _attn) = layer.forward(&xs)?;
            xs = new_xs;
        }

        // Apply final layer normalization
        xs = self.layer_norm_after.forward(&xs)?;

        // Transpose back to batch-first format for output
        xs = xs.transpose(0, 1)?; // (T, B, E) -> (B, T, E)

        // Apply language model head to get logits
        let logits = self.lm_head.forward(&xs)?;

        Ok(ESM2Output { logits })
    }
    pub(crate) fn get_device(&self) -> &Device {
        self.embeddings.word_embeddings.embeddings().device()
    }
}

// Output struct
// todo: potentially expand
pub struct ESM2Output {
    pub logits: Tensor,
    // pub representations: std::collections::HashMap<usize, Tensor>,
    // pub attentions: Option<Tensor>,
    // pub contacts: Option<Tensor>,
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
