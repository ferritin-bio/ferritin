use crate::esmc::layers::rotary::RotaryEmbedding;
use crate::esmc::models::esmc::ESMCConfig;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, LayerNormConfig, VarBuilder};
// use scaled_dot_product_attention;

pub struct MultiHeadAttention {
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    layernorm_qkv: nn::Sequential,
    out_proj: nn::Linear,
    q_ln: Box<dyn Module>,
    k_ln: Box<dyn Module>,
    rotary: RotaryEmbedding,
}

impl MultiHeadAttention {
    // pub fn new(d_model: usize, n_heads: usize, bias: bool, qk_layernorm: bool) -> Result<Self> {
    //     let d_head = d_model / n_heads;

    //     let layernorm = nn::LayerNorm::new(d_model)?;
    //     let linear = nn::linear(d_model, d_model * 3, bias)?;
    //     let layernorm_qkv = nn::seq().add(layernorm).add(linear);

    //     let out_proj = nn::linear(d_model, d_model, bias)?;

    //     let (q_ln, k_ln): (Box<dyn Module>, Box<dyn Module>) = if qk_layernorm {
    //         (
    //             Box::new(nn::LayerNorm::new(d_model)?),
    //             Box::new(nn::LayerNorm::new(d_model)?),
    //         )
    //     } else {
    //         (Box::new(nn::Identity), Box::new(nn::Identity))
    //     };

    //     Ok(Self {
    //         d_model,
    //         n_heads,
    //         d_head,
    //         layernorm_qkv,
    //         out_proj,
    //         q_ln,
    //         k_ln,
    //         rotary: RotaryEmbedding::new(d_model / n_heads)?,
    //     })
    // }
    pub fn load(vb: VarBuilder, config: &ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model, n_heads, ..
        } = config;

        let d_head = d_model / n_heads;
        // let ln_conf = LayerNormConfig::from(1e-5);
        let ln_conf = LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: false,
        };
        let layernorm = nn::layer_norm(*d_model, ln_conf, vb.pp("layernorm_qkv.0"))?;
        let linear = nn::linear_no_bias(*d_model, d_model * 3, vb.pp("layernorm_qkv.1"))?;
        let layernorm_qkv = nn::seq().add(layernorm).add(linear);
        let out_proj = nn::linear_no_bias(*d_model, *d_model, vb.pp("out_proj"))?;
        // note: only handling the True case for the moment
        // let  qk_layernorm = true
        let q_ln = Box::new(nn::layer_norm(*d_model, ln_conf, vb.pp("q_ln"))?);
        let k_ln = Box::new(nn::layer_norm(*d_model, ln_conf, vb.pp("k_ln"))?);

        let rotary = RotaryEmbedding::load(vb.pp("rotary"), config)?;

        Ok(Self {
            d_model: *d_model,
            n_heads: *n_heads,
            d_head,
            layernorm_qkv,
            out_proj,
            q_ln,
            k_ln,
            rotary,
        })
    }

    pub fn forward(&self, x: &Tensor, sequence_id: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        // QKV projection: (B, L, d_model) → (B, L, 3*d_model) → split
        let qkv = self.layernorm_qkv.forward(x)?;
        let chunks = qkv.chunk(3, candle_core::D::Minus1)?;
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);

        // Per-head layer norms
        let q = self.q_ln.forward(q)?;
        let k = self.k_ln.forward(k)?;

        // Reshape to (B, n_heads, L, d_head) for rotary + SDPA
        let q = q
            .reshape((b, l, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.n_heads, self.d_head))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.n_heads, self.d_head))?
            .transpose(1, 2)?;

        // Apply rotary positional embeddings
        let (q, k) = self.rotary.forward(&q, &k)?;

        // Scaled dot-product attention
        let scale = (self.d_head as f64).sqrt().recip(); // f64 scalar — candle Mul<f64> overload
        let attn =
            (q.matmul(&k.transpose(candle_core::D::Minus1, candle_core::D::Minus2)?)? * scale)?;

        // Optional key-padding mask from sequence_id (True = real token, False = pad)
        let attn = if let Some(seq_id) = sequence_id {
            // seq_id: (B, L) bool-like; build (B, 1, 1, L) mask so padded keys are masked out
            let mask = seq_id
                .unsqueeze(1)?
                .unsqueeze(1)?
                .broadcast_as(attn.shape())?;
            let neg_inf = (Tensor::ones_like(&attn)? * f64::NEG_INFINITY)?;
            mask.where_cond(&attn, &neg_inf)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax(&attn, candle_core::D::Minus1)?;

        // Weighted sum over values, reshape back to (B, L, d_model)
        let context = attn.matmul(&v)?; // (B, n_heads, L, d_head)
        let context = context.transpose(1, 2)?.reshape((b, l, self.d_model))?;

        self.out_proj.forward(&context)
    }

    // fn apply_rotary(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
    //     let q = q.reshape((-1, self.n_heads, self.d_head))?;
    //     let k = k.reshape((-1, self.n_heads, self.d_head))?;
    //     let (q, k) = self.rotary.forward(&q, &k)?;
    //     let q = q.flatten_from(1)?;
    //     let k = k.flatten_from(1)?;
    //     Ok((q, k))
    // }

    // pub fn forward(&self, x: &Tensor, seq_id: Option<&Tensor>) -> Result<Tensor> {
    //     let qkv = self.layernorm_qkv.forward(x)?;
    //     let chunks = qkv.chunk(3, -1)?;
    //     let (query, key, value) = (&chunks[0], &chunks[1], &chunks[2]);

    //     let query = self.q_ln.forward(query)?;
    //     let key = self.k_ln.forward(key)?;
    //     let (query, key) = self.apply_rotary(&query, &key)?;

    //     let query = query.reshape((query.dims()[0], self.n_heads, -1, self.d_head))?;
    //     let key = key.reshape((key.dims()[0], self.n_heads, -1, self.d_head))?;
    //     let value = value.reshape((value.dims()[0], self.n_heads, -1, self.d_head))?;

    //     let context = if let Some(seq_id) = seq_id {
    //         let mask = seq_id.unsqueeze(-1)?.eq(&seq_id.unsqueeze(-2)?)?;
    //         let mask = mask.unsqueeze(1)?;
    //         scaled_dot_product_attention(&query, &key, &value, Some(&mask))?
    //     } else {
    //         scaled_dot_product_attention(&query, &key, &value, None)?
    //     };

    //     let context = context.flatten_from(2)?;
    //     self.out_proj.forward(&context)
    // }
}
