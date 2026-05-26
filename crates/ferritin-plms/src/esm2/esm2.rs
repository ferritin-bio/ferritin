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
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, linear, ops};
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
        let cos = cos.unsqueeze(0)?; // Shape becomes (1, seq_len, head_dim)
        let sin = sin.unsqueeze(0)?; // Shape becomes (1, seq_len, head_dim)
        let query_rot = rotate_half(query)?;
        let query_rotated = query
            .broadcast_mul(&cos)?
            .add(&query_rot.broadcast_mul(&sin)?)?;
        let key_rot = rotate_half(key)?;
        let key_rotated = key
            .broadcast_mul(&cos)?
            .add(&key_rot.broadcast_mul(&sin)?)?;

        Ok((query_rotated, key_rotated))
    }
}

pub struct ESM2Embeddings {
    pub(crate) word_embeddings: Embedding,
    /// Whether to apply ESM-2 token dropout compensation (always true for ESM-2 checkpoints).
    /// Masked token positions are zeroed in the embedding; all embeddings are scaled by
    /// (1 - mask_ratio_train) / (1 - mask_ratio_observed) to compensate.
    /// See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/esm/modeling_esm.py
    token_dropout: bool,
    mask_token_id: u32,
}
impl ESM2Embeddings {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let vocab_size = config.vocab_size as usize;
        let hidden_size = config.hidden_size;
        let word_embeddings = vb.get((vocab_size, hidden_size), "word_embeddings.weight")?;
        Ok(Self {
            word_embeddings: Embedding::new(word_embeddings, hidden_size),
            // ESM-2 does NOT use an embedding scale; that was ESM-1 only.
            token_dropout: config.token_dropout,
            mask_token_id: config.mask_token_id as u32,
        })
    }
    pub fn embed_tokens(&self, x: &Tensor) -> Result<Tensor> {
        self.word_embeddings.forward(x)
    }
    /// Forward pass for ESM-2 embeddings.
    ///
    /// `attention_mask`: optional `(batch, seq_len)` mask (1 = real token, 0 = pad).
    /// When `token_dropout` is enabled the mask is used to compute the compensation
    /// scale; if omitted all positions are treated as real tokens.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut embeddings = self.embed_tokens(input_ids)?;

        if self.token_dropout {
            let device = input_ids.device();
            let (batch, seq_len) = input_ids.dims2()?;
            let hidden_size = embeddings.dim(D::Minus1)?;

            // Zero out positions where the input is the mask token.
            let mask_full = Tensor::full(self.mask_token_id, (batch, seq_len), device)?;
            let is_mask = input_ids.eq(&mask_full)?; // (batch, seq_len), dtype U8 (0/1)
            let is_mask_3d =
                is_mask
                    .unsqueeze(D::Minus1)?
                    .broadcast_as((batch, seq_len, hidden_size))?;
            embeddings = is_mask_3d.where_cond(&embeddings.zeros_like()?, &embeddings)?;

            // Scale embeddings to compensate for the masking used during training.
            // mask_ratio_train is hardcoded to 0.15 * 0.8 = 0.12 across all ESM-2 runs.
            let mask_ratio_train = 0.12f32;
            let src_lengths = match attention_mask {
                Some(am) => am.sum(1)?.to_dtype(DType::F32)?,
                None => Tensor::full(seq_len as f32, (batch,), device)?,
            };
            let num_masked = is_mask.sum(1)?.to_dtype(DType::F32)?;
            let mask_ratio_observed = (num_masked / &src_lengths)?;
            let denominator = (Tensor::ones(&[batch], DType::F32, device)? - &mask_ratio_observed)?;
            let scale = (Tensor::full(1.0f32 - mask_ratio_train, (batch,), device)? / denominator)?
                .unsqueeze(1)?
                .unsqueeze(2)?
                .to_dtype(embeddings.dtype())?;
            embeddings = embeddings.broadcast_mul(&scale)?;
        }

        Ok(embeddings)
    }
}

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
        let layer_norm = candle_nn::layer_norm(
            hidden_size,
            config.layer_norm_eps as f64,
            vb.pp("layer_norm"),
        )?;
        let decoder = Linear::new(embedding.embeddings().clone(), Some(bias));
        Ok(ESM2LMHead {
            dense,
            layer_norm,
            decoder,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // HF comment: "Using F.gelu yields subtly wrong results" — use the exact erf-based gelu.
        let hidden = xs.apply(&self.dense)?.gelu_erf()?;
        let normalized = hidden.apply(&self.layer_norm)?;
        normalized.apply(&self.decoder)
    }
}

/// Contact prediction head for ESM-2.
///
/// Takes per-layer attention matrices, symmetrises them, applies APC correction,
/// then predicts per-residue-pair contact probabilities with a single linear
/// layer followed by sigmoid.
///
/// Reference: HF `EsmContactPredictionHead` in `modeling_esm.py`.
pub struct ESM2ContactHead {
    /// Linear map from (layers * heads) channels to 1.
    feedforward: Linear,
    /// Whether the tokenizer prepends a BOS/CLS token (always true for ESM-2).
    prepend_bos: bool,
    /// Whether the tokenizer appends an EOS token (always true for ESM-2).
    append_eos: bool,
    /// Token ID for the EOS token — used to build the EOS attention mask.
    eos_idx: u32,
}
impl ESM2ContactHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let in_features = config.num_hidden_layers as usize * config.num_attention_heads;
        Ok(ESM2ContactHead {
            feedforward: linear(in_features, 1, vb.pp("regression"))?,
            prepend_bos: true,
            append_eos: true,
            // EOS token is index 2 in the ESM-2 vocabulary (<eos>).
            // config.mask_token_id (32) is different — do not use that here.
            eos_idx: 2,
        })
    }

    /// Compute contact probabilities from stacked per-layer attention weights.
    ///
    /// # Arguments
    /// * `tokens`     – input token IDs, shape `(batch, seq_len)`
    /// * `attentions` – stacked per-layer attention weights,
    ///                   shape `(batch, layers, heads, seq_len, seq_len)`
    ///
    /// # Returns
    /// Contact probability matrix, shape `(batch, seq_len−2, seq_len−2)`
    /// (BOS and EOS positions are stripped).
    pub fn forward(&self, tokens: &Tensor, attentions: &Tensor) -> Result<Tensor> {
        let (batch, _layers, _heads, seq_len, _) = attentions.dims5()?;
        let dev = attentions.device();
        let dtype = attentions.dtype();

        // --- EOS masking ---------------------------------------------------
        // Build a mask that zeros out any row or column corresponding to an
        // EOS token, then remove the EOS position from the sequence dimension.
        let mut attn = if self.append_eos {
            // is_not_eos: (batch, seq_len) — 1 for real tokens, 0 for EOS.
            // Matches Python: tokens.ne(self.eos_idx).to(attentions)
            let eos_full = Tensor::full(self.eos_idx, (batch, seq_len), dev)?;
            let is_eos = tokens.eq(&eos_full)?.to_dtype(dtype)?;
            let is_not_eos = (Tensor::ones((batch, seq_len), dtype, dev)? - &is_eos)?;
            // Outer product mask: (batch, seq_len, seq_len)
            let mask_2d = is_not_eos
                .unsqueeze(2)? // (batch, seq_len, 1)
                .broadcast_mul(&is_not_eos.unsqueeze(1)?)?; // × (batch, 1, seq_len)
            // Expand to 5-D and apply
            let mask_5d = mask_2d.unsqueeze(1)?.unsqueeze(1)?; // (batch,1,1,T,T)
            let attn = attentions.broadcast_mul(&mask_5d)?;
            // Remove the last row/col (EOS position)
            attn.narrow(3, 0, seq_len - 1)?.narrow(4, 0, seq_len - 1)?
        } else {
            attentions.clone()
        };

        // --- BOS stripping -------------------------------------------------
        // Remove the first row/col (BOS/CLS position).
        if self.prepend_bos {
            let cur = attn.dim(3)?;
            attn = attn.narrow(3, 1, cur - 1)?.narrow(4, 1, cur - 1)?;
        }

        // attn is now (batch, layers, heads, core_len, core_len)
        let (batch, layers, heads, core_len, _) = attn.dims5()?;

        // --- Flatten layers×heads -------------------------------------------
        // (batch, layers, heads, T, T) → (batch, layers*heads, T, T)
        let attn = attn.reshape((batch, layers * heads, core_len, core_len))?;

        // --- Symmetrize: A + Aᵀ ---------------------------------------------
        let attn = (&attn + attn.transpose(2, 3)?)?;

        // --- Average Product Correct (APC) -----------------------------------
        // apc(A)[i,j] = A[i,j] - A[i,:].sum * A[:,j].sum / A.sum
        let a1 = attn.sum_keepdim(D::Minus1)?; // (batch, LH, T, 1)
        let a2 = attn.sum_keepdim(D::Minus2)?; // (batch, LH, 1, T)
        let a12 = a1.sum_keepdim(D::Minus2)?; // (batch, LH, 1, 1)
        let avg = a1.broadcast_mul(&a2)?.broadcast_div(&a12)?;
        let attn = (&attn - &avg)?;

        // --- Linear + sigmoid ------------------------------------------------
        // Permute to (batch, T, T, LH) for the linear layer.
        // contiguous() required: permute leaves non-contiguous strides that matmul rejects.
        let attn = attn.permute((0, 2, 3, 1))?.contiguous()?;
        // Linear: (batch, T, T, LH) → (batch, T, T, 1)
        let contacts = self.feedforward.forward(&attn)?;
        // Squeeze: (batch, T, T, 1) → (batch, T, T)
        let contacts = contacts.squeeze(D::Minus1)?;
        // Sigmoid → probabilities in [0, 1]
        ops::sigmoid(&contacts)
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
    /// Returns `(output, attention_weights)`.
    /// `attention_weights` shape: `(batch, heads, seq_len, seq_len)`.
    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<(Tensor, Tensor)> {
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

        // Scale then compute attention scores.
        // Scaling before RoPE is the ESM-2 convention (HF modeling_esm.py:
        // "query_layer = query_layer * self.attention_head_size**-0.5")
        let scale = (self.head_dim as f64).powf(-0.5);
        let attention_scores = q.matmul(&k.transpose(1, 2)?)?.affine(scale, 0.0)?;

        // Softmax → (batch*heads, seq_len, seq_len)
        let attention_weights_flat = ops::softmax_last_dim(&attention_scores)?;

        // Reshape to (batch, heads, seq_len, seq_len) for the contact head.
        let attention_weights =
            attention_weights_flat.reshape((batch_size, self.num_heads, seq_len, seq_len))?;

        // Apply attention weights to values
        let attn_output = attention_weights_flat.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((seq_len, batch_size, embed_dim))?;
        let output = self.out_proj.forward(&attn_output)?;
        Ok((output, attention_weights))
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
        let self_attn_layer_norm = candle_nn::layer_norm(
            embed_dim,
            config.layer_norm_eps as f64,
            vb.pp("attention.LayerNorm"),
        )?;
        let final_layer_norm =
            candle_nn::layer_norm(embed_dim, config.layer_norm_eps as f64, vb.pp("LayerNorm"))?;

        Ok(Self {
            self_attn: multi_head,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }
    /// Returns `(hidden_states, attention_weights)` where attention_weights
    /// has shape `(batch, heads, seq_len, seq_len)`.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        // Pre-LayerNorm → attention → residual
        let norm_x = xs.apply(&self.self_attn_layer_norm)?;
        let (attn_out, attn_weights) = self.self_attn.forward(&norm_x, &norm_x, &norm_x)?;
        let x = (attn_out + xs)?;
        // Pre-LayerNorm → FFN → residual
        let norm_x2 = x.apply(&self.final_layer_norm)?;
        let ffn_out = norm_x2.apply(&self.fc1)?.gelu()?.apply(&self.fc2)?;
        Ok(((ffn_out + x)?, attn_weights))
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
        let layer_norm_after = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps as f64,
            vb.pp("esm.encoder.emb_layer_norm_after"),
        )?;
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
    /// Run a forward pass.
    ///
    /// `attention_mask`: optional `(batch, seq_len)` tensor (1 = real token, 0 = pad).
    /// Used for token-dropout compensation in the embeddings; pass `None` for single-
    /// sequence inference where no padding is present.
    pub fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<ESM2Output> {
        let mut xs = self.embeddings.forward(x, attention_mask)?;
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

    /// Predict residue-residue contact probabilities.
    ///
    /// Runs the transformer while collecting per-layer attention weights, stacks
    /// them, and passes them through the `ESM2ContactHead` to produce a symmetric
    /// probability matrix.
    ///
    /// # Arguments
    /// * `tokens`           – token IDs, shape `(batch, seq_len)` including BOS and EOS
    /// * `attention_mask`   – optional padding mask (1=real, 0=pad)
    ///
    /// # Returns
    /// `(batch, seq_len−2, seq_len−2)` contact probability matrix.
    /// BOS and EOS positions are stripped; remaining positions correspond directly
    /// to the amino-acid residues in the input sequence.
    pub fn predict_contacts(
        &self,
        tokens: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(tokens, attention_mask)?;
        xs = xs.transpose(0, 1)?; // (B, T, E) → (T, B, E)

        // Collect per-layer attention: each is (batch, heads, seq_len, seq_len)
        let mut layer_attentions: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (new_xs, attn_weights) = layer.forward(&xs)?;
            xs = new_xs;
            layer_attentions.push(attn_weights);
        }

        xs = self.layer_norm_after.forward(&xs)?;
        // Note: we do NOT need to transpose back or run lm_head — the contact
        // head only uses the attention weights, not the final hidden states.

        // Stack → (batch, layers, heads, seq_len, seq_len)
        let attentions = Tensor::stack(&layer_attentions, 1)?;

        // ESM model zeroes attention to pad tokens; replicate that here.
        if let Some(mask) = attention_mask {
            // mask: (batch, seq_len) → broadcast to (batch, 1, 1, seq_len, seq_len)
            let mask_f = mask.to_dtype(attentions.dtype())?;
            let m1 = mask_f.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(3)?;
            let m2 = mask_f.unsqueeze(1)?.unsqueeze(2)?.unsqueeze(4)?;
            // Zero out both the row and the column for each padded position
            // by multiplying both masks (outer product along seq dims)
            let _ = attentions.broadcast_mul(&m1)?.broadcast_mul(&m2)?;
        }

        self.contact_head.forward(tokens, &attentions)
    }
}

pub struct ESM2Output {
    pub logits: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token dropout compensation: with no masked tokens in the sequence the scale
    /// factor should be (1 - 0.12) / (1 - 0) = 0.88, so every embedding value is
    /// multiplied by 0.88 relative to the raw word-embedding lookup.
    #[test]
    fn test_token_dropout_scale_no_masked_tokens() -> Result<()> {
        let device = Device::Cpu;
        // Minimal config: one token, one dimension, token_dropout=true, mask_token_id=32
        let config = ESM2Config {
            vocab_size: 33,
            hidden_size: 4,
            token_dropout: true,
            mask_token_id: 32,
            num_attention_heads: 1,
            num_hidden_layers: 1,
            intermediate_size: 8,
            ..ESM2Config::t6_8m()
        };
        // Build a tiny embedding table: all-ones weights.
        let weight = Tensor::ones(
            (config.vocab_size as usize, config.hidden_size),
            DType::F32,
            &device,
        )?;
        let embeddings = ESM2Embeddings {
            word_embeddings: Embedding::new(weight, config.hidden_size),
            token_dropout: config.token_dropout,
            mask_token_id: config.mask_token_id as u32,
        };
        // Input: token id 0 (not the mask token), shape (1, 1)
        let input_ids = Tensor::new(&[[0u32]], &device)?;
        let out = embeddings.forward(&input_ids, None)?;
        // Expected: 1.0 * 0.88 = 0.88
        let val = out.flatten_all()?.to_vec1::<f32>()?[0];
        let expected = (1.0 - 0.12_f32) / (1.0 - 0.0);
        assert!(
            (val - expected).abs() < 1e-5,
            "token dropout scale wrong: got {val}, expected {expected}"
        );
        Ok(())
    }

    /// With one out of two tokens being a mask token, the observed ratio is 0.5,
    /// and the masked position should be zeroed while the scale becomes
    /// (1 - 0.12) / (1 - 0.5) = 1.76.
    #[test]
    fn test_token_dropout_scale_half_masked() -> Result<()> {
        let device = Device::Cpu;
        let hidden = 4usize;
        let vocab = 33usize;
        let mask_id = 32u32;
        let weight = Tensor::ones((vocab, hidden), DType::F32, &device)?;
        let embeddings = ESM2Embeddings {
            word_embeddings: Embedding::new(weight, hidden),
            token_dropout: true,
            mask_token_id: mask_id,
        };
        // Batch=1, SeqLen=2: first token real (id=0), second token mask (id=32)
        let input_ids = Tensor::new(&[[0u32, mask_id]], &device)?;
        let out = embeddings.forward(&input_ids, None)?; // (1, 2, 4)
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        // masked position (second token) should be 0
        for &v in &vals[hidden..] {
            assert!(v.abs() < 1e-6, "masked position should be 0, got {v}");
        }
        // real position (first token) should be scaled
        let expected_scale = (1.0 - 0.12_f32) / (1.0 - 0.5);
        for &v in &vals[..hidden] {
            assert!(
                (v - expected_scale).abs() < 1e-4,
                "real token scale wrong: got {v}, expected {expected_scale}"
            );
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Contact head structural tests
    // -----------------------------------------------------------------------

    /// Build a minimal ESM2ContactHead with all-ones weights for deterministic testing.
    fn make_contact_head(layers: usize, heads: usize, device: &Device) -> Result<ESM2ContactHead> {
        let in_features = layers * heads;
        // Weight shape for Linear: (out_features=1, in_features)
        let weight = Tensor::ones(&[1usize, in_features], DType::F32, device)?;
        let feedforward = Linear::new(weight, None);
        Ok(ESM2ContactHead {
            feedforward,
            prepend_bos: true,
            append_eos: true,
            eos_idx: 2,
        })
    }

    /// Helper: build a random-ish (but deterministic) attention tensor.
    /// Shape: (batch, layers, heads, seq_len, seq_len), values in [0,1] summing to 1
    /// across the last dim (simulating softmax output).
    fn make_attention(
        batch: usize,
        layers: usize,
        heads: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        // Use a fixed pattern: attn[b,l,h,i,j] ∝ exp(-|i-j|)
        // so nearby tokens get higher attention.
        let mut data = vec![0f32; batch * layers * heads * seq_len * seq_len];
        for b in 0..batch {
            for l in 0..layers {
                for h in 0..heads {
                    // Compute softmax of -|i-j| for each row i
                    for i in 0..seq_len {
                        let base = b * layers * heads * seq_len * seq_len
                            + l * heads * seq_len * seq_len
                            + h * seq_len * seq_len
                            + i * seq_len;
                        let mut row_sum = 0f32;
                        for j in 0..seq_len {
                            let v = (-(i as f32 - j as f32).abs()).exp();
                            data[base + j] = v;
                            row_sum += v;
                        }
                        for j in 0..seq_len {
                            data[base + j] /= row_sum;
                        }
                    }
                }
            }
        }
        Tensor::from_vec(data, &[batch, layers, heads, seq_len, seq_len], device)
    }

    /// Contact map shape: output should be (batch, seq_len-2, seq_len-2).
    #[test]
    fn test_contact_head_output_shape() -> Result<()> {
        let device = Device::Cpu;
        let (batch, layers, heads, seq_len) = (2usize, 3, 4, 9);
        let head = make_contact_head(layers, heads, &device)?;
        // tokens: BOS(0), 7 amino acids, EOS(2)
        let mut token_data = vec![0u32; batch * seq_len];
        for b in 0..batch {
            token_data[b * seq_len] = 0; // BOS
            token_data[b * seq_len + seq_len - 1] = 2; // EOS
            for i in 1..seq_len - 1 {
                token_data[b * seq_len + i] = 4; // amino acid L
            }
        }
        let tokens = Tensor::from_vec(token_data, &[batch, seq_len], &device)?;
        let attn = make_attention(batch, layers, heads, seq_len, &device)?;
        let contacts = head.forward(&tokens, &attn)?;
        assert_eq!(
            contacts.shape().dims(),
            &[batch, seq_len - 2, seq_len - 2],
            "contact map shape mismatch"
        );
        Ok(())
    }

    /// All values must be in [0, 1] (sigmoid output).
    #[test]
    fn test_contact_head_values_in_unit_interval() -> Result<()> {
        let device = Device::Cpu;
        let (batch, layers, heads, seq_len) = (1, 3, 4, 11);
        let head = make_contact_head(layers, heads, &device)?;
        let tokens = {
            let mut t = vec![4u32; batch * seq_len]; // all amino-acid tokens
            t[0] = 0;
            t[seq_len - 1] = 2;
            Tensor::from_vec(t, &[batch, seq_len], &device)?
        };
        let attn = make_attention(batch, layers, heads, seq_len, &device)?;
        let contacts = head.forward(&tokens, &attn)?;
        let vals: Vec<f32> = contacts.flatten_all()?.to_vec1()?;
        for &v in &vals {
            assert!(
                v >= 0.0 && v <= 1.0,
                "contact probability out of [0,1]: {v}"
            );
        }
        Ok(())
    }

    /// Contact map must be symmetric: contacts[i][j] == contacts[j][i].
    ///
    /// Symmetry is guaranteed because:
    /// 1. We explicitly symmetrize the attention (A + Aᵀ)
    /// 2. APC correction preserves symmetry
    /// 3. The linear layer maps channels→1 with the same weights at [i,j] and [j,i]
    /// 4. Sigmoid is elementwise
    #[test]
    fn test_contact_head_symmetry() -> Result<()> {
        let device = Device::Cpu;
        let (batch, layers, heads, seq_len) = (1, 6, 20, 13); // mimics ESM-2 8M
        let head = make_contact_head(layers, heads, &device)?;
        let tokens = {
            let mut t = vec![4u32; batch * seq_len];
            t[0] = 0;
            t[seq_len - 1] = 2;
            Tensor::from_vec(t, &[batch, seq_len], &device)?
        };
        let attn = make_attention(batch, layers, heads, seq_len, &device)?;
        let contacts = head.forward(&tokens, &attn)?.squeeze(0)?; // (L, L)
        let core = seq_len - 2;
        let mat: Vec<f32> = contacts.flatten_all()?.to_vec1()?;
        for i in 0..core {
            for j in 0..core {
                let c_ij = mat[i * core + j];
                let c_ji = mat[j * core + i];
                assert!(
                    (c_ij - c_ji).abs() < 1e-5,
                    "contact map not symmetric at ({i},{j}): {c_ij} vs {c_ji}"
                );
            }
        }
        Ok(())
    }

    /// Sequentially adjacent residues (|i-j|=1) should generally have higher
    /// contact probability than distant residues (|i-j|>=5) given the
    /// distance-decaying attention pattern we constructed.
    #[test]
    fn test_contact_head_local_bias() -> Result<()> {
        let device = Device::Cpu;
        let (batch, layers, heads, seq_len) = (1, 6, 20, 15);
        let head = make_contact_head(layers, heads, &device)?;
        let tokens = {
            let mut t = vec![4u32; batch * seq_len];
            t[0] = 0;
            t[seq_len - 1] = 2;
            Tensor::from_vec(t, &[batch, seq_len], &device)?
        };
        let attn = make_attention(batch, layers, heads, seq_len, &device)?;
        let contacts = head.forward(&tokens, &attn)?.squeeze(0)?; // (L, L)
        let core = seq_len - 2;
        let mat: Vec<f32> = contacts.flatten_all()?.to_vec1()?;

        let local_sum: f32 = (0..core - 1).map(|i| mat[i * core + i + 1]).sum::<f32>();
        let distant_sum: f32 = (0..core)
            .flat_map(|i| (0..core).map(move |j| (i, j)))
            .filter(|&(i, j)| (i as isize - j as isize).abs() >= 5)
            .map(|(i, j)| mat[i * core + j])
            .sum::<f32>();
        let n_local = (core - 1) as f32;
        let n_distant = (0..core)
            .flat_map(|i| (0..core).map(move |j| (i, j)))
            .filter(|&(i, j)| (i as isize - j as isize).abs() >= 5)
            .count() as f32;

        let avg_local = local_sum / n_local;
        let avg_distant = distant_sum / n_distant;
        assert!(
            avg_local > avg_distant,
            "expected local contacts > distant; avg_local={avg_local:.4} avg_distant={avg_distant:.4}"
        );
        Ok(())
    }

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
