//! LM encoder adapter: projects ESMC-6B hidden states → single representation.
//!
//! Four standard pre-norm transformer blocks (self-attention + SwiGLU FFN)
//! running at d_model=2560, followed by a LayerNorm and a linear projection
//! to d_single=384.
//!
//! Weight layout (rooted at `lm_encoder`):
//! ```text
//! blocks.{0..3}.attn.layernorm_qkv.0.*  — pre-attn LayerNorm
//! blocks.{0..3}.attn.layernorm_qkv.1.*  — QKV linear (no bias)
//! blocks.{0..3}.attn.out_proj.*         — output projection (no bias)
//! blocks.{0..3}.attn.q_ln.*             — per-head Q LayerNorm (no bias)
//! blocks.{0..3}.attn.k_ln.*             — per-head K LayerNorm (no bias)
//! blocks.{0..3}.ffn.0.*                 — pre-FFN LayerNorm
//! blocks.{0..3}.ffn.1.*                 — gate+up projection (no bias)
//! blocks.{0..3}.ffn.3.*                 — down projection (no bias)
//! norm.*                                — final LayerNorm
//! proj.*                                — linear d_in → d_out (no bias)
//! ```

use candle_core::{D, Result, Tensor};
use candle_nn::{self as nn, LayerNorm, LayerNormConfig, Module, VarBuilder};

// ── Attention ─────────────────────────────────────────────────────────────

struct LMAttention {
    layernorm_qkv: nn::Sequential,
    out_proj: nn::Linear,
    q_ln: LayerNorm,
    k_ln: LayerNorm,
    n_heads: usize,
    d_head: usize,
}

impl LMAttention {
    fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        let d_head = d_model / n_heads;
        let norm = nn::layer_norm(d_model, LayerNormConfig::from(1e-5), vb.pp("layernorm_qkv.0"))?;
        let qkv = nn::linear_no_bias(d_model, d_model * 3, vb.pp("layernorm_qkv.1"))?;
        let layernorm_qkv = nn::seq().add(norm).add(qkv);
        let out_proj = nn::linear_no_bias(d_model, d_model, vb.pp("out_proj"))?;

        let q_ln = LayerNorm::new_no_bias(vb.pp("q_ln").get((d_model,), "weight")?, 1e-5);
        let k_ln = LayerNorm::new_no_bias(vb.pp("k_ln").get((d_model,), "weight")?, 1e-5);

        Ok(Self { layernorm_qkv, out_proj, q_ln, k_ln, n_heads, d_head })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // Pre-norm + QKV projection: [B, L, 3*d_model]
        let qkv = self.layernorm_qkv.forward(x)?;
        let chunks = qkv.chunk(3, D::Minus1)?;
        let (q, k, v) = (&chunks[0], &chunks[1], &chunks[2]);

        // Per-head LayerNorms on full d_model before reshape
        let q = self.q_ln.forward(q)?;
        let k = self.k_ln.forward(k)?;

        // [B, L, d_model] → [B, n_heads, L, d_head] → [B*n_heads, L, d_head]
        let reshape_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape((b, l, self.n_heads, self.d_head))?
                .transpose(1, 2)?
                .reshape((b * self.n_heads, l, self.d_head))
        };
        let q = reshape_heads(&q)?;
        let k = reshape_heads(&k)?;
        let v = reshape_heads(v)?;

        // Scaled dot-product attention
        let scale = (self.d_head as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let weights = nn::ops::softmax(&scores, D::Minus1)?;
        let out = weights.matmul(&v)?; // [B*n_heads, L, d_head]

        // [B*n_heads, L, d_head] → [B, L, d_model]
        let out = out
            .reshape((b, self.n_heads, l, self.d_head))?
            .transpose(1, 2)?
            .reshape((b, l, self.n_heads * self.d_head))?;

        self.out_proj.forward(&out)
    }
}

// ── SwiGLU FFN ────────────────────────────────────────────────────────────

struct LMFfn {
    norm: LayerNorm,
    gate_up_proj: nn::Linear,
    down_proj: nn::Linear,
}

impl LMFfn {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        // SwiGLU hidden dim: nearest multiple of 256 of (d_model × 8/3)
        let hidden = ((8.0 / 3.0 * d_model as f64 + 255.0) / 256.0).floor() as usize * 256;
        Ok(Self {
            norm: nn::layer_norm(d_model, LayerNormConfig::from(1e-5), vb.pp("0"))?,
            gate_up_proj: nn::linear_no_bias(d_model, hidden * 2, vb.pp("1"))?,
            down_proj: nn::linear_no_bias(hidden, d_model, vb.pp("3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let gate_up = self.gate_up_proj.forward(&x)?;
        let chunks = gate_up.chunk(2, D::Minus1)?;
        let hidden = (chunks[0].silu()? * &chunks[1])?;
        self.down_proj.forward(&hidden)
    }
}

// ── Transformer block ──────────────────────────────────────────────────────

struct LMBlock {
    attn: LMAttention,
    ffn: LMFfn,
}

impl LMBlock {
    fn load(vb: VarBuilder, d_model: usize, n_heads: usize) -> Result<Self> {
        Ok(Self {
            attn: LMAttention::load(vb.pp("attn"), d_model, n_heads)?,
            ffn: LMFfn::load(vb.pp("ffn"), d_model)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + &self.attn.forward(x)?)?;
        (&x + &self.ffn.forward(&x)?)
    }
}

// ── LMEncoder ─────────────────────────────────────────────────────────────

/// 4-layer transformer adapter that maps ESMC-6B hidden states
/// (`[B, L, d_in=2560]`) to the ESMFold2 single representation (`[B, L, d_out=384]`).
pub struct LMEncoder {
    blocks: Vec<LMBlock>,
    norm: LayerNorm,
    proj: nn::Linear,
}

impl LMEncoder {
    /// Load the LM encoder from a `VarBuilder` rooted at `lm_encoder.*`.
    ///
    /// `n_heads` is derived as `d_in / 64` (standard head size; 2560/64 = 40 for ESMC-6B).
    pub fn load(vb: VarBuilder, d_in: usize, d_out: usize, n_layers: usize) -> Result<Self> {
        let n_heads = d_in / 64; // 40 for d_in=2560
        let blocks = (0..n_layers)
            .map(|i| LMBlock::load(vb.pp(format!("blocks.{i}")), d_in, n_heads))
            .collect::<Result<Vec<_>>>()?;
        let norm = nn::layer_norm(d_in, LayerNormConfig::from(1e-5), vb.pp("norm"))?;
        let proj = nn::linear_no_bias(d_in, d_out, vb.pp("proj"))?;
        Ok(Self { blocks, norm, proj })
    }

    /// Project ESMC-6B hidden states to the single representation.
    ///
    /// - Input:  `[B, L, d_in=2560]`
    /// - Output: `[B, L, d_out=384]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        self.proj.forward(&self.norm.forward(&x)?)
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_lm_encoder_output_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let encoder = LMEncoder::load(vb, 2560, 384, 4).unwrap();
        let x = Tensor::zeros(&[1, 16, 2560], DType::F32, &device).unwrap();
        let out = encoder.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 16, 384]);
    }

    #[test]
    fn test_lm_encoder_batch_invariant_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let encoder = LMEncoder::load(vb, 2560, 384, 4).unwrap();
        let x = Tensor::zeros(&[2, 8, 2560], DType::F32, &device).unwrap();
        let out = encoder.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 8, 384]);
    }

    #[test]
    fn test_lm_ffn_hidden_dim() {
        // For d_model=2560, SwiGLU hidden = nearest-256 of (2560 * 8/3) = 6912
        let expected = ((8.0_f64 / 3.0 * 2560.0 + 255.0) / 256.0).floor() as usize * 256;
        assert_eq!(expected, 6912);
    }
}
