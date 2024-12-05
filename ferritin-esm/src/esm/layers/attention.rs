use super::rotary::RotaryEmbedding;
use crate::esm::models::esmc::ESMCConfig;
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
    pub fn load(vb: VarBuilder, config: ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            expansion_ratio,
            n_heads,
            bias,
            ..
        } = config;

        let d_head = d_model / n_heads;
        let ln_conf = LayerNormConfig::from(1e-5);
        let layernorm = nn::layer_norm(d_model, ln_conf, vb.pp("layer_norm"))?;
        let linear = nn::linear(d_model, d_model * 3, vb.pp("linear1"))?;
        let layernorm_qkv = nn::seq().add(layernorm).add(linear);
        let out_proj = nn::linear(d_model, d_model, vb.pp("out_proj"))?;

        // note: only handling the True case for the moment
        // let  qk_layernorm = true
        let q_ln = Box::new(nn::layer_norm(d_model, ln_conf, vb.pp("q_ln"))?);
        let k_ln = Box::new(nn::layer_norm(d_model, ln_conf, vb.pp("k_ln"))?);
        let rotary: RotaryEmbedding::load(vb.pp("rotary"), config)?;

        Ok(Self {
            d_model,
            n_heads,
            d_head,
            layernorm_qkv,
            out_proj,
            q_ln,
            k_ln,
            rotary,
        })
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
