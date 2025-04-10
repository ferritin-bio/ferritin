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
use candle_nn::{Embedding, LayerNorm, LayerNormConfig, Linear, VarBuilder, layer_norm, linear};
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
        // Name: lm_head.bias, Shape: [33]
        // Name: lm_head.dense.bias, Shape: [320]
        // Name: lm_head.dense.weight, Shape: [320, 320]
        // Name: lm_head.layer_norm.bias, Shape: [320]
        // Name: lm_head.layer_norm.weight, Shape: [320]
        let dense = linear(
            config.hidden_size as usize,
            config.hidden_size as usize,
            vb.pp("dense"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size as usize,
            LayerNormConfig {
                eps: config.layer_norm_eps as f64,
                remove_mean: true,
                affine: true,
            },
            vb.pp("layer_norm"),
        )?;

        let decoder_bias = vb.get(config.vocab_size as usize, "bias")?;
        let decoder = Linear::new(
            // The weight will be tied to embeddings later
            Tensor::zeros(
                &[config.vocab_size as usize, config.hidden_size as usize],
                DType::F32,
                vb.device(),
            )?,
            Some(decoder_bias),
        );

        Ok(ESM2LMHead {
            dense,
            layer_norm,
            decoder,
        })
    }
    fn forward() {
        todo!()
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
    fn forward() {
        todo!()
    }
}

/// ESM1 style layer norm
/// for esm1blayernorm use Candle::nn::layernorm
///
// // class ESM1LayerNorm(nn.Module):
//     def __init__(self, hidden_size, eps=1e-12, affine=True):
//         """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
//         super().__init__()
//         self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
//         self.eps = eps
//         self.affine = bool(affine)
//         if self.affine:
//             self.weight = nn.Parameter(torch.ones(hidden_size))
//             self.bias = nn.Parameter(torch.zeros(hidden_size))
//         else:
//             self.weight, self.bias = None, None
//
//     def forward(self, x):
//         dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
//         means = x.mean(dims, keepdim=True)
//         x_zeromean = x - means
//         variances = x_zeromean.pow(2).mean(dims, keepdim=True)
//         x = x_zeromean / torch.sqrt(variances + self.eps)
//         if self.affine:
//             x = (self.weight * x) + self.bias
//         return x
//
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
        let means = x.mean_keepdim(D::Minus1)?;
        let x_zeromean = x.sub(&means)?;
        let variances = x_zeromean.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let variances1 = &(variances + self.eps)?.sqrt()?;
        let x_norm = x_zeromean.div(variances1)?;
        let weighted = x_norm.mul(&self.weight)?;
        weighted.add(&self.bias)
    }
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

    // fn forward(
    //     &self,
    //     xs: &Tensor,
    //     self_attn_padding_mask: Option<&Tensor>,
    //     need_head_weights: bool,
    // ) -> Result<(Tensor, Option<Tensor>)> {
    //     let residual = xs.clone();
    //     let x = self.self_attn_layer_norm.forward(xs)?;
    //     let (x, attn) =
    //         self.self_attn
    //             .forward(&x, &x, &x, self_attn_padding_mask, need_head_weights)?;
    //     let x = (x + residual)?;
    //     // Second block: FFN with residual connection
    //     let residual = x.clone();
    //     let x = self.final_layer_norm.forward(&x)?;
    //     let x = self.fc1.forward(&x)?;
    //     let x = x.gelu()?;
    //     let x = self.fc2.forward(&x)?;
    //     let x = x + residual;
    //     Ok((x, attn))
    // }
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
    pub fn forward(&self, x: &Tensor) -> Result<ESM2Output> {
        // def forward(self, tokens, repr_layers=[], need_head_weights=False, return_contacts=False):
        //     if return_contacts:
        //         need_head_weights = True
        //     assert tokens.ndim == 2
        //     padding_mask = tokens.eq(self.padding_idx)  # B, T
        //     x = self.embed_scale * self.embed_tokens(tokens)
        //     if self.token_dropout:
        //         x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
        //         # x: B x T x C
        //         mask_ratio_train = 0.15 * 0.8
        //         src_lengths = (~padding_mask).sum(-1)
        //         mask_ratio_observed = (tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
        //         x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
        //     if padding_mask is not None:
        //         x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        //     repr_layers = set(repr_layers)
        //     hidden_representations = {}
        //     if 0 in repr_layers:
        //         hidden_representations[0] = x
        //     if need_head_weights:
        //         attn_weights = []
        //     # (B, T, E) => (T, B, E)
        //     x = x.transpose(0, 1)
        //     if not padding_mask.any():
        //         padding_mask = None
        //     for layer_idx, layer in enumerate(self.layers):
        //         x, attn = layer(
        //             x,
        //             self_attn_padding_mask=padding_mask,
        //             need_head_weights=need_head_weights,
        //         )
        //         if (layer_idx + 1) in repr_layers:
        //             hidden_representations[layer_idx + 1] = x.transpose(0, 1)
        //         if need_head_weights:
        //             # (H, B, T, T) => (B, H, T, T)
        //             attn_weights.append(attn.transpose(1, 0))
        //     x = self.emb_layer_norm_after(x)
        //     x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        //     # last hidden representation should have layer norm applied
        //     if (layer_idx + 1) in repr_layers:
        //         hidden_representations[layer_idx + 1] = x
        //     x = self.lm_head(x)
        //     result = {"logits": x, "representations": hidden_representations}
        //     if need_head_weights:
        //         # attentions: B x L x H x T x T
        //         attentions = torch.stack(attn_weights, 1)
        //         if padding_mask is not None:
        //             attention_mask = 1 - padding_mask.type_as(attentions)
        //             attention_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        //             attentions = attentions * attention_mask[:, None, None, :, :]
        //         result["attentions"] = attentions
        //         if return_contacts:
        //             contacts = self.contact_head(tokens, attentions)
        //             result["contacts"] = contacts
        //     return result
        // 6. Process through all layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let (new_x, attn) = layer.forward(&x, None, need_head_weights)?;
            x = new_x;
            if need_head_weights {
                attn_weights.push(attn.transpose(1, 0)?);
        }

        // 7. Apply final layer norm
        x = self.layer_norm_after.forward(&x)?;
        // 8. Transpose back to (B, T, E)
        x = x.transpose(0, 1)?;
        // 9. Apply language model head
        let logits = self.lm_head.forward(&x)?;

        // 10. Process attention weights and compute contacts if needed
        let attentions = if need_head_weights {
            Some(Tensor::stack(&attn_weights, 1)?)
        } else {
            None
        };

        let contacts = if need_head_weights {
            // Some(self.contact_head.forward(tokens, &attentions.unwrap())?)
            None // Placeholder
        } else {
            None
        };

        // 11. Return results
        Ok(ESM2Output {
            logits,
            hidden_states: hidden_representations,
            attentions,
            contacts,
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
pub struct ESM2Output {
    pub logits: Tensor,
    pub representations: std::collections::HashMap<usize, Tensor>,
    pub attentions: Option<Tensor>,
    pub contacts: Option<Tensor>,
}
