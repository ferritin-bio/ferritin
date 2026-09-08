use crate::esmc::layers::rotary::RotaryEmbedding;
use crate::esmc::models::esmc::ESMCConfig;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, LayerNorm, LayerNormConfig, VarBuilder};
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

        // layernorm_qkv.0 has both weight and bias in the checkpoint.
        let layernorm = nn::layer_norm(
            *d_model,
            LayerNormConfig::from(1e-5),
            vb.pp("layernorm_qkv.0"),
        )?;
        let linear = nn::linear_no_bias(*d_model, d_model * 3, vb.pp("layernorm_qkv.1"))?;
        let layernorm_qkv = nn::seq().add(layernorm).add(linear);
        let out_proj = nn::linear_no_bias(*d_model, *d_model, vb.pp("out_proj"))?;

        // q_ln / k_ln have weight but no bias in the checkpoint — use new_no_bias.
        let q_ln: Box<dyn Module> = {
            let w = vb.pp("q_ln").get((*d_model,), "weight")?;
            Box::new(LayerNorm::new_no_bias(w, 1e-5))
        };
        let k_ln: Box<dyn Module> = {
            let w = vb.pp("k_ln").get((*d_model,), "weight")?;
            Box::new(LayerNorm::new_no_bias(w, 1e-5))
        };

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

        // Scaled dot-product attention.
        // k must be made contiguous after transpose — Metal (and some CPU paths) require
        // contiguous tensors for batched matmul.
        let scale = (self.d_head as f64).sqrt().recip();
        let k_t = k
            .transpose(candle_core::D::Minus1, candle_core::D::Minus2)?
            .contiguous()?;
        let attn = (q.contiguous()?.matmul(&k_t)? * scale)?;

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
        let context = attn.matmul(&v.contiguous()?)?; // (B, n_heads, L, d_head)
        let context = context
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, l, self.d_model))?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esmc::models::esmc::ESMCConfig;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    /// A tiny `MultiHeadAttention` with deterministic weights.
    ///
    /// This is the attention ESM-C *and* ESM-3 both use — `esm3::layers::blocks`
    /// loads this same type — so masking correctness here covers both runners'
    /// `embed_batch` (ferritin-100.12).
    fn tiny_attention(device: &Device) -> Result<(MultiHeadAttention, ESMCConfig)> {
        let config = ESMCConfig {
            d_model: 8,
            n_heads: 2,
            n_layers: 1,
            ..ESMCConfig::esmc_300m()
        };
        let d = config.d_model;

        // Tiny LCG: reproducible without pulling in an RNG crate.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rand = |dims: &[usize], device: &Device| -> Result<Tensor> {
            let n: usize = dims.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
                })
                .collect();
            Tensor::from_vec(v, dims, device)
        };

        let mut ts: HashMap<String, Tensor> = HashMap::new();
        ts.insert(
            "layernorm_qkv.0.weight".into(),
            Tensor::ones(d, DType::F32, device)?,
        );
        ts.insert(
            "layernorm_qkv.0.bias".into(),
            Tensor::zeros(d, DType::F32, device)?,
        );
        ts.insert("layernorm_qkv.1.weight".into(), rand(&[d * 3, d], device)?);
        ts.insert("out_proj.weight".into(), rand(&[d, d], device)?);
        ts.insert("q_ln.weight".into(), Tensor::ones(d, DType::F32, device)?);
        ts.insert("k_ln.weight".into(), Tensor::ones(d, DType::F32, device)?);

        let vb = VarBuilder::from_tensors(ts, DType::F32, device);
        let attn = MultiHeadAttention::load(vb, &config)?;
        Ok((attn, config))
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = b.flatten_all()?.to_vec1()?;
        assert_eq!(a.len(), b.len(), "shape mismatch in comparison");
        Ok(a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max))
    }

    /// Deterministic `(1, len, d_model)` activations, then right-padded with a
    /// *different* constant so unmasked padding is guaranteed to perturb.
    fn inputs(len: usize, padded: usize, d: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let real: Vec<f32> = (0..len * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut padded_v = real.clone();
        padded_v.extend(std::iter::repeat_n(0.9f32, (padded - len) * d));
        Ok((
            Tensor::from_vec(real, (1, len, d), device)?,
            Tensor::from_vec(padded_v, (1, padded, d), device)?,
        ))
    }

    fn mask_row(len: usize, padded: usize, device: &Device) -> Result<Tensor> {
        let mut m = vec![1u8; len];
        m.resize(padded, 0);
        Tensor::from_vec(m, (1, padded), device)
    }

    /// `sequence_id` is a `where_cond` predicate — 1 keeps a key, 0 drops it.
    /// With it, right-padding must leave the real positions untouched; without
    /// it, the same padding must change them (so the check is not vacuous).
    #[test]
    fn test_sequence_id_mask_isolates_padded_keys() -> Result<()> {
        let device = Device::Cpu;
        let (attn, config) = tiny_attention(&device)?;
        let (len, padded) = (5usize, 9usize);
        let (x, x_padded) = inputs(len, padded, config.d_model, &device)?;

        let alone = attn.forward(&x, None)?;

        let mask = mask_row(len, padded, &device)?;
        let masked = attn.forward(&x_padded, Some(&mask))?.narrow(1, 0, len)?;
        let diff = max_abs_diff(&alone, &masked)?;
        assert!(
            diff < 1e-5,
            "masked padding changed the real positions: max abs diff {diff}"
        );

        let unmasked = attn.forward(&x_padded, None)?.narrow(1, 0, len)?;
        let leak = max_abs_diff(&alone, &unmasked)?;
        assert!(
            leak > 1e-4,
            "padding without sequence_id should perturb the real positions, but max abs diff was {leak}"
        );
        Ok(())
    }

    /// Two rows of different lengths in one batch: each must match the same
    /// input run alone. This is the ordering check — the mask is broadcast
    /// `(B, 1, 1, L) -> (B, heads, L, L)`, so a row that picked up the other
    /// row's mask would show up here and only here.
    #[test]
    fn test_batched_rows_match_single_row_attention() -> Result<()> {
        let device = Device::Cpu;
        let (attn, config) = tiny_attention(&device)?;
        let d = config.d_model;
        let (long, short) = (7usize, 3usize);

        let (x_long, _) = inputs(long, long, d, &device)?;
        let (x_short, x_short_padded) = inputs(short, long, d, &device)?;

        let batch = Tensor::cat(&[&x_long, &x_short_padded], 0)?;
        let mask = Tensor::cat(
            &[
                &mask_row(long, long, &device)?,
                &mask_row(short, long, &device)?,
            ],
            0,
        )?;
        let batched = attn.forward(&batch, Some(&mask))?;

        for (row, x, len) in [(0usize, &x_long, long), (1usize, &x_short, short)] {
            let alone = attn.forward(x, None)?;
            let from_batch = batched.narrow(0, row, 1)?.narrow(1, 0, len)?;
            let diff = max_abs_diff(&alone, &from_batch)?;
            assert!(
                diff < 1e-5,
                "batch row {row} disagrees with its single-row attention: {diff}"
            );
        }
        Ok(())
    }
}
