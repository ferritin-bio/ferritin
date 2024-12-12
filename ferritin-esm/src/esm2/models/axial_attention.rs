// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Dropout, Linear};
use std::f64;

pub struct RowSelfAttention {
    num_heads: usize,
    dropout: f64,
    head_dim: usize,
    scaling: f64,
    max_tokens_per_msa: usize,
    attn_shape: String,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    dropout_module: Dropout,
}

impl RowSelfAttention {
    // pub fn new(
    //     embed_dim: usize,
    //     num_heads: usize,
    //     dropout: f64,
    //     max_tokens_per_msa: usize,
    // ) -> Result<Self> {
    //     let head_dim = embed_dim / num_heads;
    //     let scaling = 1.0 / f64::sqrt(head_dim as f64);

    //     Ok(Self {
    //         num_heads,
    //         dropout,
    //         head_dim,
    //         scaling,
    //         max_tokens_per_msa,
    //         attn_shape: "hnij".to_string(),
    //         k_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         v_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         q_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         out_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         dropout_module: Dropout::new(dropout),
    //     })
    // }

    fn align_scaling(&self, q: &Tensor) -> Result<f64> {
        let num_rows = q.dim(0)?;
        Ok(self.scaling / f64::sqrt(num_rows as f64))
    }

    // fn batched_forward(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    // ) -> Result<(Tensor, Tensor)> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, batch_size, embed_dim) = shape;
    //     let max_rows = std::cmp::max(1, self.max_tokens_per_msa / num_cols);
    //     let scaling = self.align_scaling(x)?;

    //     let mut attns = Tensor::zeros((self.num_heads, batch_size, num_rows, num_rows), x.dtype())?;

    //     for start in (0..num_rows).step_by(max_rows) {
    //         let end = std::cmp::min(start + max_rows, num_rows);
    //         let slice = x.narrow(0, start, end - start)?;

    //         let attn_weights = self.compute_attention_weights(
    //             &slice,
    //             scaling,
    //             self_attn_mask,
    //             self_attn_padding_mask
    //                 .map(|mask| mask.narrow(1, start, end - start))
    //                 .as_ref(),
    //         )?;
    //         attns += attn_weights;
    //     }

    //     let attn_probs = attns.softmax(-1)?;
    //     let attn_probs = self.dropout_module.forward(&attn_probs)?;

    //     let mut outputs = Vec::new();
    //     for start in (0..num_rows).step_by(max_rows) {
    //         let end = std::cmp::min(start + max_rows, num_rows);
    //         let slice = x.narrow(0, start, end - start)?;

    //         let output = self.compute_attention_update(&slice, &attn_probs)?;
    //         outputs.push(output);
    //     }

    //     let output = Tensor::cat(&outputs, 0)?;
    //     Ok((output, attn_probs))
    // }

    // fn compute_attention_weights(
    //     &self,
    //     x: &Tensor,
    //     scaling: f64,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    // ) -> Result<Tensor> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, batch_size, embed_dim) = shape;

    //     let q = self.q_proj.forward(x)?.reshape((
    //         num_rows,
    //         num_cols,
    //         batch_size,
    //         self.num_heads,
    //         self.head_dim,
    //     ))?;
    //     let k = self.k_proj.forward(x)?.reshape((
    //         num_rows,
    //         num_cols,
    //         batch_size,
    //         self.num_heads,
    //         self.head_dim,
    //     ))?;

    //     let q = q.mul_scalar(scaling)?;

    //     if let Some(mask) = self_attn_padding_mask {
    //         let mask = mask
    //             .permute((1, 2, 0))?
    //             .unsqueeze(-1)?
    //             .unsqueeze(-1)?
    //             .to_device(q.device())?;
    //         let q = q.mul(&(1.0 - mask))?;
    //     }

    //     let attn_weights = q.einsum("rinhd,rjnhd->hnij", &[&k])?;

    //     if self_attn_mask.is_some() {
    //         unimplemented!("self_attn_mask not supported");
    //     }

    //     if let Some(mask) = self_attn_padding_mask {
    //         let mask = mask.select(1, 0)?.unsqueeze(0)?.unsqueeze(2)?;
    //         let attn_weights = attn_weights.masked_fill(&mask, -10000.0)?;
    //         Ok(attn_weights)
    //     } else {
    //         Ok(attn_weights)
    //     }
    // }

    // fn compute_attention_update(&self, x: &Tensor, attn_probs: &Tensor) -> Result<Tensor> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, batch_size, embed_dim) = shape;

    //     let v = self.v_proj.forward(x)?.reshape((
    //         num_rows,
    //         num_cols,
    //         batch_size,
    //         self.num_heads,
    //         self.head_dim,
    //     ))?;
    //     let context = attn_probs.einsum(&format!("{},rjnhd->rinhd", self.attn_shape), &[&v])?;
    //     let context = context.reshape((num_rows, num_cols, batch_size, embed_dim))?;
    //     self.out_proj.forward(&context)
    // }

    // pub fn forward(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    // ) -> Result<(Tensor, Tensor)> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, _, _) = shape;

    //     if (num_rows * num_cols > self.max_tokens_per_msa) && !x.requires_grad() {
    //         self.batched_forward(x, self_attn_mask, self_attn_padding_mask)
    //     } else {
    //         let scaling = self.align_scaling(x)?;
    //         let attn_weights =
    //             self.compute_attention_weights(x, scaling, self_attn_mask, self_attn_padding_mask)?;
    //         let attn_probs = attn_weights.softmax(-1)?;
    //         let attn_probs = self.dropout_module.forward(&attn_probs)?;
    //         let output = self.compute_attention_update(x, &attn_probs)?;
    //         Ok((output, attn_probs))
    //     }
    // }
}

pub struct ColumnSelfAttention {
    num_heads: usize,
    dropout: f64,
    head_dim: usize,
    scaling: f64,
    max_tokens_per_msa: usize,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    dropout_module: Dropout,
}

impl ColumnSelfAttention {
    // pub fn new(
    //     embed_dim: usize,
    //     num_heads: usize,
    //     dropout: f64,
    //     max_tokens_per_msa: usize,
    // ) -> Result<Self> {
    //     let head_dim = embed_dim / num_heads;
    //     let scaling = 1.0 / f64::sqrt(head_dim as f64);

    //     Ok(Self {
    //         num_heads,
    //         dropout,
    //         head_dim,
    //         scaling,
    //         max_tokens_per_msa,
    //         k_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         v_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         q_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         out_proj: Linear::new(embed_dim, embed_dim, Default::default())?,
    //         dropout_module: Dropout::new(dropout),
    //     })
    // }

    // fn batched_forward(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    // ) -> Result<(Tensor, Tensor)> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, batch_size, _) = shape;
    //     let max_cols = std::cmp::max(1, self.max_tokens_per_msa / num_rows);

    //     let mut outputs = Vec::new();
    //     let mut attns = Vec::new();

    //     for start in (0..num_cols).step_by(max_cols) {
    //         let end = std::cmp::min(start + max_cols, num_cols);
    //         let slice = x.narrow(1, start, end - start)?;

    //         let (output, attn) = self.forward(
    //             &slice,
    //             self_attn_mask,
    //             self_attn_padding_mask
    //                 .map(|mask| mask.narrow(2, start, end - start))
    //                 .as_ref(),
    //         )?;
    //         outputs.push(output);
    //         attns.push(attn);
    //     }

    //     let output = Tensor::cat(&outputs, 1)?;
    //     let attns = Tensor::cat(&attns, 1)?;
    //     Ok((output, attns))
    // }

    // fn compute_attention_update(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    // ) -> Result<(Tensor, Tensor)> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, batch_size, embed_dim) = shape;

    //     if num_rows == 1 {
    //         let attn_probs = Tensor::ones(
    //             (self.num_heads, num_cols, batch_size, num_rows, num_rows),
    //             x.dtype(),
    //             x.device(),
    //         )?;
    //         let v = self.v_proj.forward(x)?;
    //         let output = self.out_proj.forward(&v)?;
    //         Ok((output, attn_probs))
    //     } else {
    //         let q = self.q_proj.forward(x)?.reshape((
    //             num_rows,
    //             num_cols,
    //             batch_size,
    //             self.num_heads,
    //             self.head_dim,
    //         ))?;
    //         let k = self.k_proj.forward(x)?.reshape((
    //             num_rows,
    //             num_cols,
    //             batch_size,
    //             self.num_heads,
    //             self.head_dim,
    //         ))?;
    //         let v = self.v_proj.forward(x)?.reshape((
    //             num_rows,
    //             num_cols,
    //             batch_size,
    //             self.num_heads,
    //             self.head_dim,
    //         ))?;

    //         let q = q.mul_scalar(self.scaling)?;
    //         let attn_weights = q.einsum("icnhd,jcnhd->hcnij", &[&k])?;

    //         if self_attn_mask.is_some() {
    //             unimplemented!("self_attn_mask not supported");
    //         }

    //         let attn_weights = if let Some(mask) = self_attn_padding_mask {
    //             let mask = mask.permute((2, 0, 1))?.unsqueeze(0)?.unsqueeze(3)?;
    //             attn_weights.masked_fill(&mask, -10000.0)?
    //         } else {
    //             attn_weights
    //         };

    //         let attn_probs = attn_weights.softmax(-1)?;
    //         let attn_probs = self.dropout_module.forward(&attn_probs)?;

    //         let context = attn_probs.einsum("hcnij,jcnhd->icnhd", &[&v])?;
    //         let context = context.reshape((num_rows, num_cols, batch_size, embed_dim))?;
    //         let output = self.out_proj.forward(&context)?;
    //         Ok((output, attn_probs))
    //     }
    // }

    // pub fn forward(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    // ) -> Result<(Tensor, Tensor)> {
    //     let shape = x.dims4()?;
    //     let (num_rows, num_cols, _, _) = shape;

    //     if (num_rows * num_cols > self.max_tokens_per_msa) && !x.requires_grad() {
    //         self.batched_forward(x, self_attn_mask, self_attn_padding_mask)
    //     } else {
    //         self.compute_attention_update(x, self_attn_mask, self_attn_padding_mask)
    //     }
    // }
}
