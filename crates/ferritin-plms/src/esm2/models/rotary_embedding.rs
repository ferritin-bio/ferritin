/// Rotary Embedding file based on the python version here
///
/// See also:
///     - Candle R-emb: https://github.com/huggingface/candle/blob/d339b01726cc33d40ca2df1bf1cfa55379616e4e/candle-transformers/src/models/whisper/model.rs#L26
///     - YI: https://github.com/huggingface/candle/blob/d339b01726cc33d40ca2df1bf1cfa55379616e4e/candle-transformers/src/models/yi.rs#L70
///     - Mistral: https://github.com/huggingface/candle/blob/d339b01726cc33d40ca2df1bf1cfa55379616e4e/candle-transformers/src/models/quantized_mistral.rs#L53
///     - Falcon (very similar): https://github.com/huggingface/candle/blob/d339b01726cc33d40ca2df1bf1cfa55379616e4e/candle-transformers/src/models/falcon.rs#L130
///
// # Copyright (c) Meta Platforms, Inc. and affiliates.
// #
// # This source code is licensed under the MIT license found in the
// # LICENSE file in the root directory of this source tree.
//
// from typing import Tuple
// import torch
// def rotate_half(x):
//     x1, x2 = x.chunk(2, dim=-1)
//     return torch.cat((-x2, x1), dim=-1)
//
// def apply_rotary_pos_emb(x, cos, sin):
//     cos = cos[:, : x.shape[-2], :]
//     sin = sin[:, : x.shape[-2], :]
//     return (x * cos) + (rotate_half(x) * sin)
//
// class RotaryEmbedding(torch.nn.Module):
//     """
//     The rotary position embeddings from RoFormer_ (Su et. al).
//     A crucial insight from the method is that the query and keys are
//     transformed by rotation matrices which depend on the relative positions.
//     Other implementations are available in the Rotary Transformer repo_ and in
//     GPT-NeoX_, GPT-NeoX was an inspiration
//     .. _RoFormer: https://arxiv.org/abs/2104.09864
//     .. _repo: https://github.com/ZhuiyiTechnology/roformer
//     .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
//     .. warning: Please note that this embedding is not registered on purpose, as it is transformative
//         (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
//     """
//     def __init__(self, dim: int, *_, **__):
//         super().__init__()
//         # Generate and save the inverse frequency buffer (non trainable)
//         inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
//         self.register_buffer("inv_freq", inv_freq)
//         self._seq_len_cached = None
//         self._cos_cached = None
//         self._sin_cached = None
//     def _update_cos_sin_tables(self, x, seq_dimension=1):
//         seq_len = x.shape[seq_dimension]
//         # Reset the tables if the sequence length has changed,
//         # or if we're on a new device (possibly due to tracing for instance)
//         if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
//             self._seq_len_cached = seq_len
//             t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
//             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
//             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
//             self._cos_cached = emb.cos()[None, :, :]
//             self._sin_cached = emb.sin()[None, :, :]
//         return self._cos_cached, self._sin_cached
//     def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
//         self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
//         return (
//             apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
//             apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
//         )
use super::esm2::ESM2Config;
use candle_core::{D, Device, Result, Tensor};
use candle_nn::VarBuilder;

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    // In PyTorch: x1, x2 = x.chunk(2, dim=-1)
    let (x1, x2) = x.chunk(2, D::Minus(1))?;

    // In PyTorch: return torch.cat((-x2, x1), dim=-1)
    let neg_x2 = x2.neg()?;
    Tensor::cat(&[&neg_x2, x1], D::Minus(1))
}

fn apply_rotary_pos_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // In PyTorch: cos = cos[:, : x.shape[-2], :]
    // In Candle, narrow(dim, start, len) is equivalent to PyTorch's slicing
    let cos = cos.narrow(1, 0, x.dim(D::Minus(2))?)?;
    let sin = sin.narrow(1, 0, x.dim(D::Minus(2))?)?;

    // In PyTorch: (x * cos) + (rotate_half(x) * sin)
    let x_cos = x.mul(&cos)?;
    let x_rot = rotate_half(x)?;
    let x_sin = x_rot.mul(&sin)?;
    x_cos.add(&x_sin)
}

#[derive(Debug)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
    seq_len_cached: Option<usize>,
    cos_cached: Option<Tensor>,
    sin_cached: Option<Tensor>,
}

impl RotaryEmbedding {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        // Get head_dim from config to match PyTorch initialization
        // For ESM2, the rotary embedding dim should be the attention head dimension
        let head_dim = config.hidden_size as usize / config.num_attention_heads as usize;
        let dim = head_dim; // Use head dimension directly as in PyTorch

        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();

        let inv_freq = Tensor::new(inv_freq, vb.device())?;

        Ok(Self {
            inv_freq,
            seq_len_cached: None,
            cos_cached: None,
            sin_cached: None,
        })
    }
    pub fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq = Tensor::new(inv_freq, device)?;

        Ok(Self {
            inv_freq,
            seq_len_cached: None,
            cos_cached: None,
            sin_cached: None,
        })
    }

    fn update_cos_sin_tables(
        &mut self,
        x: &Tensor,
        seq_dimension: i64,
    ) -> Result<(&Tensor, &Tensor)> {
        let seq_len = x.dim(seq_dimension as usize)?;

        // Reset tables if sequence length changed or we're on a different device
        // This matches the PyTorch logic
        // Check if we need to regenerate the tables
        // 1. If sequence length has changed
        // 2. If there's no cached values yet
        // 3. If the device has changed (comparing device pointers)
        let need_regeneration = self.seq_len_cached != Some(seq_len)
            || self.cos_cached.is_none()
            || self.cos_cached.as_ref().map_or(false, |t| {
                // More efficient device comparison using pointer address
                !std::ptr::eq(t.device(), x.device())
            });

        if need_regeneration {
            self.seq_len_cached = Some(seq_len);

            // Create sequence tensor from 0 to seq_len-1
            let t = Tensor::arange(0u32, seq_len as u32, x.device())?
                .to_dtype(self.inv_freq.dtype())?;

            // Calculate frequencies: einsum("i,j->ij", t, inv_freq) in PyTorch
            // In Candle, we can do this with outer product: t_i * inv_freq_j
            let freqs = t.unsqueeze(1)?.matmul(&self.inv_freq.unsqueeze(0)?)?;

            // Concatenate frequencies with themselves (same as in PyTorch)
            let emb = Tensor::cat(&[&freqs, &freqs], D::Minus(1))?;

            // Cache the cos and sin values
            self.cos_cached = Some(emb.cos()?.unsqueeze(0)?); // Add batch dimension
            self.sin_cached = Some(emb.sin()?.unsqueeze(0)?);
        }

        Ok((
            self.cos_cached.as_ref().unwrap(),
            self.sin_cached.as_ref().unwrap(),
        ))
    }

    pub fn forward(&mut self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        // In PyTorch: self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)
        // We use -2 as the sequence dimension just like in the PyTorch code
        let (cos_cached, sin_cached) = self.update_cos_sin_tables(k, D::Minus(2))?;

        // In PyTorch: return apply_rotary_pos_emb for both q and k
        Ok((
            apply_rotary_pos_emb(q, cos_cached, sin_cached)?,
            apply_rotary_pos_emb(k, cos_cached, sin_cached)?,
        ))
    }
}
