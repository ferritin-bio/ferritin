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
use candle_nn::{
    Embedding, LayerNorm, LayerNormConfig, Linear, VarBuilder, layer_norm, linear, ops,
};
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
struct FalconRotaryEmbedding {
    inv_freq: Tensor,
    cache: Option<(usize, Tensor, Tensor)>,
}

impl FalconRotaryEmbedding {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        Ok(FalconRotaryEmbedding {
            inv_freq: vb.get(config.inv_freq_size(), "inv_freq")?,
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
    layer_norm: LayerNorm,
    decoder: Linear, // Weight tied to embeddings
}
impl ESM2LMHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let hidden_size = config.hidden_size as usize;
        let vocab_size = config.vocab_size as usize;
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        let ln_config = LayerNormConfig {
            eps: config.layer_norm_eps as f64,
            remove_mean: true,
            affine: true,
        };
        let layer_norm = layer_norm(hidden_size, ln_config, vb.pp("layer_norm"))?;
        let decoder_bias = vb.get(config.vocab_size as usize, "bias")?;
        let decoder = Linear::new(
            Tensor::zeros(&[vocab_size, hidden_size], DType::F32, vb.device())?,
            Some(decoder_bias),
        );
        Ok(ESM2LMHead {
            dense,
            layer_norm,
            decoder,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.dense)?
            .gelu()?
            .apply(&self.layer_norm)?
            .apply(&self.decoder)
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
        // let contact_scale
        let in_features = (config.num_hidden_layers as usize * config.num_attention_heads) as usize;
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
    rotary_emb: Option<FalconRotaryEmbedding>,
    // scale val?
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
        let rotary_emb = Some(FalconRotaryEmbedding::load(
            vb.pp("self.rotary_embeddings"),
            config,
        )?);
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
        let batch_size = query.dim(0)?;
        let seq_len = query.dim(1)?;
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;
        // Reshape for multi-head attention
        // [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;
        let v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?;

        // Transpose to [batch_size, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // // Apply rotary embeddings if available
        // let (q, k) = if let Some(rotary_emb) = &self.rotary_emb {
        //     rotary_emb.apply_rotary_emb(&q, &k)?
        // } else {
        //     (q, k)
        // };

        // Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
        let scale = (self.head_dim as f64).powf(-0.5);
        let attention_scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        // Pytorch: attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        // Pytorch: attn_weights = attn_weights_float.type_as(attn_weights)
        let attention_weights = ops::softmax(&attention_scores, 3)?;

        // Pytorch:dropouts
        //         attn_probs = F.dropout(
        //             attn_weights_float.type_as(attn_weights),
        //             p=self.dropout,
        //             training=self.training,
        //         )
        //
        //  Pytorch: attn = torch.bmm(attn_probs, v)

        let context = attention_weights.matmul(&v)?;
        // Transpose and reshape back to [batch_size, seq_len, embed_dim]
        let context = context.transpose(1, 2)?;
        let embed_dim = self.num_heads * self.head_dim;
        let context = context.reshape((batch_size, seq_len, embed_dim))?;

        // Pytorch: Process the weights
        //         if need_weights:
        //             attn_weights = attn_weights_float.view(
        //                 bsz, self.num_heads, tgt_len, src_len
        //             ).type_as(attn).transpose(1, 0)
        //             if not need_head_weights:
        //                 # average attention weights over heads
        //                 attn_weights = attn_weights.mean(dim=0)
        //
        //         return attn
        let output = self.out_proj.forward(&context)?;

        Ok((output, None)) // weights tensor
    }
}

pub struct ESM1LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}
impl ESM1LayerNorm {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        Ok(Self {
            weight: vb.get(config.hidden_size, "weight")?,
            bias: vb.get(config.hidden_size, "bias")?,
            eps: config.layer_norm_eps as f64,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        println!("ESM1LayerNorm Input tensor shape: {:?}", x.dims());
        let x = x.to_dtype(DType::F32)?; // note: needed for OPs but not clear why U32
        let means = x.mean_keepdim(D::Minus1)?.to_dtype(DType::F32)?;
        let x_zeromean = x.sub(&means)?;
        let variances = x_zeromean.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let variances1 = &(variances + self.eps)?.sqrt()?;
        let x_norm = x_zeromean.div(variances1)?;
        println!("Mismatch Below");
        //Error: shape mismatch in mul, lhs: [255, 1], rhs: [480]
        let weighted = x_norm.mul(&self.weight)?;
        weighted.add(&self.bias)
    }
    // fn forward(&self, x: &Tensor) -> Result<Tensor> {
    //     let x = x.to_dtype(DType::F32)?;

    //     // Get the shape of x for debugging
    //     println!("Input tensor shape: {:?}", x.dims());

    //     // Calculate mean across the last dimension
    //     let means = x.mean_keepdim(D::Minus1)?.to_dtype(DType::F32)?;
    //     let x_zeromean = x.sub(&means)?;
    //     let variances = x_zeromean.powf(2.0)?.mean_keepdim(D::Minus1)?;
    //     let variances1 = &(variances + self.eps)?.sqrt()?;
    //     let x_norm = x_zeromean.div(variances1)?;

    //     // In PyTorch, the weight and bias are broadcasted automatically
    //     // We need to reshape them to match the broadcasting semantics in Candle

    //     // Get the dimensions we need to prepend to weight/bias
    //     let input_dims = x.dims();
    //     let mut weight_shape = Vec::with_capacity(input_dims.len());

    //     // Add leading '1's to match the shape of x_norm except the last dimension
    //     for i in 0..input_dims.len()-1 {
    //         weight_shape.push(1);
    //     }
    //     // Add the actual dimension for the last axis
    //     weight_shape.push(self.weight.dim(0)?);

    //     println!("Reshaping weight to: {:?}", weight_shape);

    //     // Reshape weight and bias for proper broadcasting
    //     let weight_reshaped = self.weight.reshape(&weight_shape)?;
    //     let bias_reshaped = self.bias.reshape(&weight_shape)?;

    //     // Now element-wise multiplication should work with broadcasting
    //     let weighted = x_norm.mul(&weight_reshaped)?;
    //     weighted.add(&bias_reshaped)
    // }
}

// Full transformer layer
pub struct ESM2Layer {
    self_attn: ESM2Attention,
    self_attn_layer_norm: ESM1LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: ESM1LayerNorm,
}

impl ESM2Layer {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let ffn_embed_dim = config.intermediate_size as usize;
        let layer_norm = ESM1LayerNorm::load(vb.pp("attention.LayerNorm"), config)?;
        let multi_head = ESM2Attention::load(vb.pp("attention"), config)?;
        let fc1 = linear(embed_dim, ffn_embed_dim, vb.pp("intermediate.dense"))?;
        let fc2 = linear(ffn_embed_dim, embed_dim, vb.pp("output.dense"))?;
        let final_layer_norm = ESM1LayerNorm::load(vb.pp("LayerNorm"), config)?;
        Ok(Self {
            self_attn: multi_head,
            self_attn_layer_norm: layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let residual = xs.clone();
        let x = self.self_attn_layer_norm.forward(xs)?;
        let (x, attn) = self.self_attn.forward(&x, &x, &x)?;
        let x = (x + residual)?;

        // Second block: FFN with residual connection
        let residual = x.clone();
        let x = self
            .final_layer_norm
            .forward(&x)?
            .gelu()?
            .apply(&self.fc1)?
            .apply(&self.fc2)?;
        let x = (x + residual)?;
        Ok((x, attn))
    }
}

// Main model struct
pub struct ESM2 {
    config: ESM2Config,
    layers: Vec<ESM2Layer>,
    layer_norm_after: LayerNorm,
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
        let layer_norm_after = layer_norm(
            config.hidden_size as usize,
            LayerNormConfig {
                eps: 0.001,
                remove_mean: true,
                affine: true,
            },
            vb.pp("esm.encoder.emb_layer_norm_after"),
        )?;
        let lm_head = ESM2LMHead::load(vb.pp("lm_head"), config)?;
        Ok(Self {
            config: config.clone(),
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
}

// Output struct
// todo: potentially expand
pub struct ESM2Output {
    pub logits: Tensor,
    // pub representations: std::collections::HashMap<usize, Tensor>,
    // pub attentions: Option<Tensor>,
    // pub contacts: Option<Tensor>,
}
