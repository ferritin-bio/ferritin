/// ESM2 Modules
///
// # Copyright (c) Meta Platforms, Inc. and affiliates.
// #
// # This source code is licensed under the MIT license found in the
// # LICENSE file in the root directory of this source tree.
// import math
// from typing import Optional
// import torch
// import torch.nn as nn
// import torch.nn.functional as F
// from .multihead_attention import MultiheadAttention  # noqa
// from .axial_attention import ColumnSelfAttention, RowSelfAttention
//
// def gelu(x):
//     """Implementation of the gelu activation function.
//     For information: OpenAI GPT's gelu is slightly different
//     (and gives slightly different results):
//     0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
//     """
//     return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
//
// def symmetrize(x):
//     "Make layer symmetric in final two dimensions, used for contact prediction."
//     return x + x.transpose(-1, -2)
//
// def apc(x):
//     "Perform average product correct, used for contact prediction."
//     a1 = x.sum(-1, keepdims=True)
//     a2 = x.sum(-2, keepdims=True)
//     a12 = x.sum((-1, -2), keepdims=True)
//     avg = a1 * a2
//     avg.div_(a12)  # in-place to reduce memory
//     normalized = x - avg
//     return normalized
//
// class ESM1LayerNorm(nn.Module):
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
// try:
//     from apex.normalization import FusedLayerNorm as _FusedLayerNorm
//
//     class ESM1bLayerNorm(_FusedLayerNorm):
//         @torch.jit.unused
//         def forward(self, x):
//             if not x.is_cuda:
//                 return super().forward(x)
//             else:
//                 with torch.cuda.device(x.device):
//                     return super().forward(x)
//
// except ImportError:
//     from torch.nn import LayerNorm as ESM1bLayerNorm
//
//
// class TransformerLayer(nn.Module):
//     """Transformer layer block."""
//
//     def __init__(
//         self,
//         embed_dim,
//         ffn_embed_dim,
//         attention_heads,
//         add_bias_kv=True,
//         use_esm1b_layer_norm=False,
//         use_rotary_embeddings: bool = False,
//     ):
//         super().__init__()
//         self.embed_dim = embed_dim
//         self.ffn_embed_dim = ffn_embed_dim
//         self.attention_heads = attention_heads
//         self.use_rotary_embeddings = use_rotary_embeddings
//         self._init_submodules(add_bias_kv, use_esm1b_layer_norm)
//
//     def _init_submodules(self, add_bias_kv, use_esm1b_layer_norm):
//         BertLayerNorm = ESM1bLayerNorm if use_esm1b_layer_norm else ESM1LayerNorm
//
//         self.self_attn = MultiheadAttention(
//             self.embed_dim,
//             self.attention_heads,
//             add_bias_kv=add_bias_kv,
//             add_zero_attn=False,
//             use_rotary_embeddings=self.use_rotary_embeddings,
//         )
//         self.self_attn_layer_norm = BertLayerNorm(self.embed_dim)
//
//         self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
//         self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)
//
//         self.final_layer_norm = BertLayerNorm(self.embed_dim)
//
//     def forward(
//         self, x, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
//     ):
//         residual = x
//         x = self.self_attn_layer_norm(x)
//         x, attn = self.self_attn(
//             query=x,
//             key=x,
//             value=x,
//             key_padding_mask=self_attn_padding_mask,
//             need_weights=True,
//             need_head_weights=need_head_weights,
//             attn_mask=self_attn_mask,
//         )
//         x = residual + x
//
//         residual = x
//         x = self.final_layer_norm(x)
//         x = gelu(self.fc1(x))
//         x = self.fc2(x)
//         x = residual + x
//         return x, attn
//
// class AxialTransformerLayer(nn.Module):
//     """Implements an Axial MSA Transformer block."""
//
//     def __init__(
//         self,
//         embedding_dim: int = 768,
//         ffn_embedding_dim: int = 3072,
//         num_attention_heads: int = 8,
//         dropout: float = 0.1,
//         attention_dropout: float = 0.1,
//         activation_dropout: float = 0.1,
//         max_tokens_per_msa: int = 2**14,
//     ) -> None:
//         super().__init__()
//         # Initialize parameters
//         self.embedding_dim = embedding_dim
//         self.dropout_prob = dropout
//
//         row_self_attention = RowSelfAttention(
//             embedding_dim,
//             num_attention_heads,
//             dropout=dropout,
//             max_tokens_per_msa=max_tokens_per_msa,
//         )
//
//         column_self_attention = ColumnSelfAttention(
//             embedding_dim,
//             num_attention_heads,
//             dropout=dropout,
//             max_tokens_per_msa=max_tokens_per_msa,
//         )
//
//         feed_forward_layer = FeedForwardNetwork(
//             embedding_dim,
//             ffn_embedding_dim,
//             activation_dropout=activation_dropout,
//             max_tokens_per_msa=max_tokens_per_msa,
//         )
//         self.row_self_attention = self.build_residual(row_self_attention)
//         self.column_self_attention = self.build_residual(column_self_attention)
//         self.feed_forward_layer = self.build_residual(feed_forward_layer)
//
//     def build_residual(self, layer: nn.Module):
//         return NormalizedResidualBlock(
//             layer,
//             self.embedding_dim,
//             self.dropout_prob,
//         )
//
//     def forward(
//         self,
//         x: torch.Tensor,
//         self_attn_mask: Optional[torch.Tensor] = None,
//         self_attn_padding_mask: Optional[torch.Tensor] = None,
//         need_head_weights: bool = False,
//     ):
//         """
//         LayerNorm is applied either before or after the self-attention/ffn
//         modules similar to the original Transformer implementation.
//         """
//         x, row_attn = self.row_self_attention(
//             x,
//             self_attn_mask=self_attn_mask,
//             self_attn_padding_mask=self_attn_padding_mask,
//         )
//         x, column_attn = self.column_self_attention(
//             x,
//             self_attn_mask=self_attn_mask,
//             self_attn_padding_mask=self_attn_padding_mask,
//         )
//         x = self.feed_forward_layer(x)
//         if need_head_weights:
//             return x, column_attn, row_attn
//         else:
//             return x
//
// class LearnedPositionalEmbedding(nn.Embedding):
//     """
//     This module learns positional embeddings up to a fixed maximum size.
//     Padding ids are ignored by either offsetting based on padding_idx
//     or by setting padding_idx to None and ensuring that the appropriate
//     position ids are passed to the forward function.
//     """
//
//     def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
//         if padding_idx is not None:
//             num_embeddings_ = num_embeddings + padding_idx + 1
//         else:
//             num_embeddings_ = num_embeddings
//         super().__init__(num_embeddings_, embedding_dim, padding_idx)
//         self.max_positions = num_embeddings

//     def forward(self, input: torch.Tensor):
//         """Input is expected to be of size [bsz x seqlen]."""
//         if input.size(1) > self.max_positions:
//             raise ValueError(
//                 f"Sequence length {input.size(1)} above maximum "
//                 f" sequence length of {self.max_positions}"
//             )
//         mask = input.ne(self.padding_idx).int()
//         positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
//         return F.embedding(
//             positions,
//             self.weight,
//             self.padding_idx,
//             self.max_norm,
//             self.norm_type,
//             self.scale_grad_by_freq,
//             self.sparse,
//         )
//
// class SinusoidalPositionalEmbedding(nn.Module):
//     def __init__(self, embed_dim, padding_idx, learned=False):
//         super().__init__()
//         self.embed_dim = embed_dim
//         self.padding_idx = padding_idx
//         self.register_buffer("_float_tensor", torch.FloatTensor(1))
//         self.weights = None
//
//     def forward(self, x):
//         bsz, seq_len = x.shape
//         max_pos = self.padding_idx + 1 + seq_len
//         if self.weights is None or max_pos > self.weights.size(0):
//             self.weights = self.get_embedding(max_pos)
//         self.weights = self.weights.type_as(self._float_tensor)
//
//         positions = self.make_positions(x)
//         return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
//
//     def make_positions(self, x):
//         mask = x.ne(self.padding_idx)
//         range_buf = torch.arange(x.size(1), device=x.device).expand_as(x) + self.padding_idx + 1
//         positions = range_buf.expand_as(x)
//         return positions * mask.long() + self.padding_idx * (1 - mask.long())
//
//     def get_embedding(self, num_embeddings):
//         half_dim = self.embed_dim // 2
//         emb = math.log(10000) / (half_dim - 1)
//         emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
//         emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
//         emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
//         if self.embed_dim % 2 == 1:
//             # zero pad
//             emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
//         if self.padding_idx is not None:
//             emb[self.padding_idx, :] = 0
//         return emb
//
// class RobertaLMHead(nn.Module):
//     """Head for masked language modeling."""
//
//     def __init__(self, embed_dim, output_dim, weight):
//         super().__init__()
//         self.dense = nn.Linear(embed_dim, embed_dim)
//         self.layer_norm = ESM1bLayerNorm(embed_dim)
//         self.weight = weight
//         self.bias = nn.Parameter(torch.zeros(output_dim))
//
//     def forward(self, features):
//         x = self.dense(features)
//         x = gelu(x)
//         x = self.layer_norm(x)
//         # project back to size of vocabulary with bias
//         x = F.linear(x, self.weight) + self.bias
//         return x
//
// class ContactPredictionHead(nn.Module):
//     """Performs symmetrization, apc, and computes a logistic regression on the output features"""
//
//     def __init__(
//         self,
//         in_features: int,
//         prepend_bos: bool,
//         append_eos: bool,
//         bias=True,
//         eos_idx: Optional[int] = None,
//     ):
//         super().__init__()
//         self.in_features = in_features
//         self.prepend_bos = prepend_bos
//         self.append_eos = append_eos
//         if append_eos and eos_idx is None:
//             raise ValueError("Using an alphabet with eos token, but no eos token was passed in.")
//         self.eos_idx = eos_idx
//         self.regression = nn.Linear(in_features, 1, bias)
//         self.activation = nn.Sigmoid()
//
//     def forward(self, tokens, attentions):
//         # remove eos token attentions
//         if self.append_eos:
//             eos_mask = tokens.ne(self.eos_idx).to(attentions)
//             eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
//             attentions = attentions * eos_mask[:, None, None, :, :]
//             attentions = attentions[..., :-1, :-1]
//         # remove cls token attentions
//         if self.prepend_bos:
//             attentions = attentions[..., 1:, 1:]
//         batch_size, layers, heads, seqlen, _ = attentions.size()
//         attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
//
//         # features: B x C x T x T
//         attentions = attentions.to(
//             self.regression.weight.device
//         )  # attentions always float32, may need to convert to float16
//         attentions = apc(symmetrize(attentions))
//         attentions = attentions.permute(0, 2, 3, 1)
//         return self.activation(self.regression(attentions).squeeze(3))
//
// class NormalizedResidualBlock(nn.Module):
//     def __init__(
//         self,
//         layer: nn.Module,
//         embedding_dim: int,
//         dropout: float = 0.1,
//     ):
//         super().__init__()
//         self.embedding_dim = embedding_dim
//
//         self.layer = layer
//         self.dropout_module = nn.Dropout(
//             dropout,
//         )
//         self.layer_norm = ESM1bLayerNorm(self.embedding_dim)
//
//     def forward(self, x, *args, **kwargs):
//         residual = x
//         x = self.layer_norm(x)
//         outputs = self.layer(x, *args, **kwargs)
//         if isinstance(outputs, tuple):
//             x, *out = outputs
//         else:
//             x = outputs
//             out = None
//
//         x = self.dropout_module(x)
//         x = residual + x
//         if out is not None:
//             return (x,) + tuple(out)
//         else:
//             return x
//
// class FeedForwardNetwork(nn.Module):
//     def __init__(
//         self,
//         embedding_dim: int,
//         ffn_embedding_dim: int,
//         activation_dropout: float = 0.1,
//         max_tokens_per_msa: int = 2**14,
//     ):
//         super().__init__()
//         self.embedding_dim = embedding_dim
//         self.ffn_embedding_dim = ffn_embedding_dim
//         self.max_tokens_per_msa = max_tokens_per_msa
//         self.activation_fn = nn.GELU()
//         self.activation_dropout_module = nn.Dropout(
//             activation_dropout,
//         )
//         self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
//         self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
//
//     def forward(self, x):
//         x = self.activation_fn(self.fc1(x))
//         x = self.activation_dropout_module(x)
//         x = self.fc2(x)
//         return x
//
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use super::multihead_attention::MultiheadAttention;
use crate::ESM2Config;
use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

// fn gelu(x: &Tensor) -> Result<Tensor> {
//     let x_sqrt2 = x.div_scalar(2f64.sqrt())?;
//     let x_half = x.div_scalar(2.)?;
//     let erf = x_sqrt2.erf()?;
//     x_half.mul(&(erf.add_scalar(1.))?)
// }

// fn symmetrize(x: &Tensor) -> Result<Tensor> {
//     let xt = x.transpose(-1, -2)?;
//     x.add(&xt)
// }

// fn apc(x: &Tensor) -> Result<Tensor> {
fn apc(x: &Tensor) -> Result<()> {
    // let a1 = x.sum_keepdim(D::Minus1)?;
    // let a2 = x.sum_keepdim(D::Minus2)?;
    // let a12 = x.sum_keepdim(&[D::Minus1, D::Minus2])?;
    // let avg = a1.matmul(&a2)?;
    // let avg = avg.div(&a12)?;
    // x.sub(&avg)
    Ok(())
}

#[derive(Debug)]
pub struct ESM1LayerNorm {
    // weight: Tensor,
    // bias: Option<Tensor>,
}

impl ESM1LayerNorm {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        Ok(Self {})
    }

    // pub fn new(size: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<Self> {
    //     let weight = if affine {
    //         vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?
    //     } else {
    //         Tensor::ones(size, &Device::Cpu)?
    //     };

    //     let bias = if affine {
    //         Some(vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.))?)
    //     } else {
    //
    //         None
    //     };

    //     Ok(Self { weight, bias, eps })
    // }
}

// impl Module for ESM1LayerNorm {
//     fn forward(&self, x: &Tensor) -> Result<Tensor> {
//         let dims: Vec<_> = (1..x.dims().len()).rev().collect();
//         let mean = x.mean_dim(dims.as_slice(), true)?;
//         let x_centered = x.broadcast_sub(&mean)?;
//         let var = x_centered.sqr()?.mean_dim(dims.as_slice(), true)?;
//         let scale = (&var + self.eps).sqrt()?.recip()?;
//         let normalized = x_centered.mul(&scale)?;

//         let weighted = normalized.mul(&self.weight)?;
//         match &self.bias {
//             Some(bias) => weighted.add(bias),
//             None => Ok(weighted),
//         }
//     }
// }

pub type ESM1bLayerNorm = ESM1LayerNorm;

#[derive(Debug)]
pub struct TransformerLayer {
    self_attn: MultiheadAttention,
    self_attn_layer_norm: ESM1bLayerNorm,
    // fc1: nn::Linear,
    // fc2: nn::Linear,
    final_layer_norm: ESM1bLayerNorm,
}

impl TransformerLayer {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let ESM2Config {
            // embed_dim,
            // ffn_embed_dim,
            // attention_heads,
            // add_bias_kv,
            // use_esm1b_layer_norm,
            // use_rotary_embeddings,
            ..
        } = config;

        // Todo: Fix this!
        let embed_dim = 100;
        let ffn_embed_dim = 100;

        let layer_norm = ESM1LayerNorm::load(vb.pp("Layer_Norm"), config)?;
        let multi_head = MultiheadAttention::load(vb.pp("attention"), config)?;
        // let fc1 = nn::linear(embed_dim, ffn_embed_dim, vb.pp("fc1"))?;
        // let fc2 = nn::linear(ffn_embed_dim, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = ESM1LayerNorm::load(vb.pp("LayerNorm"), config)?;

        Ok(Self {
            self_attn: multi_head,
            self_attn_layer_norm: layer_norm,
            // fc1,
            // fc2,
            final_layer_norm,
        })
    }

    // pub fn new(
    //     embed_dim: usize,
    //     ffn_embed_dim: usize,
    //     attention_heads: usize,
    //     add_bias_kv: bool,
    //     use_esm1b_layer_norm: bool,
    //     use_rotary_embeddings: bool,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     let norm_builder = vb.pp("layer_norm");
    //     let layer_norm = ESM1LayerNorm::new(embed_dim, 1e-12, true, norm_builder)?;

    //     Ok(Self {
    //         self_attn: MultiheadAttention::new(
    //             embed_dim,
    //             attention_heads,
    //             add_bias_kv,
    //             false,
    //             use_rotary_embeddings,
    //             vb.pp("self_attn"),
    //         )?,
    //         self_attn_layer_norm: layer_norm,
    //         fc1: candle_nn::linear(embed_dim, ffn_embed_dim, vb.pp("fc1"))?,
    //         fc2: candle_nn::linear(ffn_embed_dim, embed_dim, vb.pp("fc2"))?,
    //         final_layer_norm: ESM1LayerNorm::new(
    //             embed_dim,
    //             1e-12,
    //             true,
    //             vb.pp("final_layer_norm"),
    //         )?,
    //     })
    // }

    // pub fn forward(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    //     need_head_weights: bool,
    // ) -> Result<(Tensor, Option<Tensor>)> {
    //     let residual = x;
    //     let x = self.self_attn_layer_norm.forward(x)?;
    //     let (x, attn) = self.self_attn.forward_t(
    //         &x,
    //         &x,
    //         &x,
    //         self_attn_padding_mask,
    //         need_head_weights,
    //         self_attn_mask,
    //     )?;
    //     let x = x.add(residual)?;

    //     let residual = &x;
    //     let x = self.final_layer_norm.forward(&x)?;
    //     let x = gelu(&self.fc1.forward(&x)?)?;
    //     let x = self.fc2.forward(&x)?;
    //     let x = x.add(residual)?;

    //     Ok((x, attn))
    // }
}

#[derive(Debug)]
pub struct AxialTransformerLayer {
    // row_self_attention: NormalizedResidualBlock<RowSelfAttention>,
    // column_self_attention: NormalizedResidualBlock<ColumnSelfAttention>,
    // feed_forward_layer: NormalizedResidualBlock<FeedForwardNetwork>,
}

impl AxialTransformerLayer {
    // pub fn new(
    //     embedding_dim: usize,
    //     ffn_embedding_dim: usize,
    //     num_attention_heads: usize,
    //     dropout: f64,
    //     attention_dropout: f64,
    //     activation_dropout: f64,
    //     max_tokens_per_msa: usize,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     let row_attn = RowSelfAttention::new(
    //         embedding_dim,
    //         num_attention_heads,
    //         dropout,
    //         max_tokens_per_msa,
    //         vb.pp("row_self_attn"),
    //     )?;

    //     let col_attn = ColumnSelfAttention::new(
    //         embedding_dim,
    //         num_attention_heads,
    //         dropout,
    //         max_tokens_per_msa,
    //         vb.pp("col_self_attn"),
    //     )?;

    //     let ffn = FeedForwardNetwork::new(
    //         embedding_dim,
    //         ffn_embedding_dim,
    //         activation_dropout,
    //         max_tokens_per_msa,
    //         vb.pp("ffn"),
    //     )?;

    //     Ok(Self {
    //         row_self_attention: NormalizedResidualBlock::new(row_attn, embedding_dim, dropout)?,
    //         column_self_attention: NormalizedResidualBlock::new(col_attn, embedding_dim, dropout)?,
    //         feed_forward_layer: NormalizedResidualBlock::new(ffn, embedding_dim, dropout)?,
    //     })
    // }

    // pub fn forward(
    //     &self,
    //     x: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     self_attn_padding_mask: Option<&Tensor>,
    //     need_head_weights: bool,
    // ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
    //     let (x, row_attn) =
    //         self.row_self_attention
    //             .forward_t(x, self_attn_mask, self_attn_padding_mask)?;

    //     let (x, col_attn) =
    //         self.column_self_attention
    //             .forward_t(&x, self_attn_mask, self_attn_padding_mask)?;

    //     let x = self.feed_forward_layer.forward(&x)?;

    //     if need_head_weights {
    //         Ok((x, Some(col_attn), Some(row_attn)))
    //     } else {
    //         Ok((x, None, None))
    //     }
    // }
}

#[derive(Debug)]
pub struct LearnedPositionalEmbedding {
    max_positions: usize,
    embedding: candle_nn::Embedding,
    padding_idx: usize,
}

impl LearnedPositionalEmbedding {
    // pub fn new(
    //     num_embeddings: usize,
    //     embedding_dim: usize,
    //     padding_idx: usize,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     let num_embeddings = if padding_idx > 0 {
    //         num_embeddings + padding_idx + 1
    //     } else {
    //         num_embeddings
    //     };

    //     Ok(Self {
    //         max_positions: num_embeddings,
    //         embedding: candle_nn::embedding(num_embeddings, embedding_dim, vb)?,
    //         padding_idx,
    //     })
    // }

    // pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    //     let seq_len = x.dims()[1];
    //     if seq_len > self.max_positions {
    //         return Err(candle_core::Error::Msg(format!(
    //             "sequence length {} above maximum sequence length of {}",
    //             seq_len, self.max_positions
    //         )));
    //     }

    //     let mask = x.ne_scalar(self.padding_idx as i64)?;
    //     let cumsum = mask.cumsum(1)?;
    //     let positions = cumsum.mul(&mask)?;
    //     let positions = positions.add_scalar(self.padding_idx as i64)?;

    //     self.embedding.forward(&positions)
    // }
}

#[derive(Debug)]
pub struct SinusoidalPositionalEmbedding {
    embed_dim: usize,
    padding_idx: usize,
    weights: Option<Tensor>,
}

impl SinusoidalPositionalEmbedding {
    // pub fn new(embed_dim: usize, padding_idx: usize) -> Self {
    //     Self {
    //         embed_dim,
    //         padding_idx,
    //         weights: None,
    //     }
    // }

    // pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
    //     let (bsz, seq_len) = (x.dims()[0], x.dims()[1]);
    //     let max_pos = self.padding_idx + 1 + seq_len;

    //     if self.weights.is_none() || max_pos > self.weights.as_ref().unwrap().dims()[0] {
    //         self.weights = Some(self.get_embedding(max_pos)?);
    //     }

    //     let positions = self.make_positions(x)?;
    //     let embeddings = self.weights.as_ref().unwrap();
    //     let out = embeddings.gather(&positions.flatten_all()?, 0)?;
    //     out.reshape((bsz, seq_len, -1))
    // }

    // fn make_positions(&self, x: &Tensor) -> Result<Tensor> {
    //     let mask = x.ne_scalar(self.padding_idx as i64)?;
    //     let range = Tensor::arange(0u32, x.dims()[1] as u32, &x.device())?;
    //     let range = range.add_scalar((self.padding_idx + 1) as i64)?;
    //     let positions = range.expand_as(x)?;
    //     mask.mul(&positions)
    // }

    // fn get_embedding(&self, num_embeddings: usize) -> Result<Tensor> {
    //     let half_dim = self.embed_dim / 2;
    //     let emb = (PI / 10000f64).ln() / (half_dim as f64 - 1.0);
    //     let emb = Tensor::arange(0f32, half_dim as f32, &Device::Cpu)?
    //         .neg()?
    //         .mul_scalar(emb as f32)?
    //         .exp()?;

    //     let pos = Tensor::arange(0f32, num_embeddings as f32, &Device::Cpu)?;
    //     let emb = pos.unsqueeze(1)?.matmul(&emb.unsqueeze(0)?)?;

    //     let sin = emb.sin()?;
    //     let cos = emb.cos()?;
    //     let emb = Tensor::cat(&[&sin, &cos], 1)?;

    //     if self.embed_dim % 2 == 1 {
    //         let zeros = Tensor::zeros((num_embeddings, 1), DType::F32, &Device::Cpu)?;
    //         let emb = Tensor::cat(&[&emb, &zeros], 1)?;
    //     }

    //     if self.padding_idx > 0 {
    //         let zeros = Tensor::zeros(self.embed_dim, DType::F32, &Device::Cpu)?;
    //         emb.index_select(&[self.padding_idx], &zeros, 0)?;
    //     }

    //     Ok(emb)
    // }
}

#[derive(Debug)]
pub struct RobertaLMHead {
    dense: candle_nn::Linear,
    layer_norm: ESM1bLayerNorm,
}

impl RobertaLMHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let ESM2Config { hidden_size, .. } = config;
        let dense = nn::linear(*hidden_size as usize, *hidden_size as usize, vb.pp("dense"))?;
        let layer_norm = ESM1bLayerNorm::load(vb.pp("LayerNorm"), config)?;
        Ok(Self { dense, layer_norm })
    }
    // pub fn new(
    //     embed_dim: usize,
    //     output_dim: usize,
    //     weight: Tensor,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     Ok(Self {
    //         dense: candle_nn::linear(embed_dim, embed_dim, vb.pp("dense"))?,
    //         layer_norm: ESM1bLayerNorm::new(embed_dim, 1e-12, true, vb.pp("layer_norm"))?,
    //         weight,
    //         bias: vb.get_with_hints(output_dim, "bias", candle_nn::Init::Const(0.))?,
    //     })
    // }

    // pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
    //     let x = self.dense.forward(features)?;
    //     let x = gelu(&x)?;
    //     let x = self.layer_norm.forward(&x)?;
    //     let x = x.matmul(&self.weight)?;
    //     x.add(&self.bias)
    // }
}

#[derive(Debug)]
pub struct ContactPredictionHead {
    // in_features: usize,
    // prepend_bos: bool,
    // append_eos: bool,
    // regression: candle_nn::Linear,
    // eos_idx: Option<usize>,
}

impl ContactPredictionHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        Ok(Self {})
    }
    // pub fn new(
    //     in_features: usize,
    //     prepend_bos: bool,
    //     append_eos: bool,
    //     bias: bool,
    //     eos_idx: Option<usize>,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     if append_eos && eos_idx.is_none() {
    //         return Err(candle_core::Error::Msg(
    //             "Using an alphabet with eos token, but no eos token was passed in.".to_string(),
    //         ));
    //     }

    //     Ok(Self {
    //         in_features,
    //         prepend_bos,
    //         append_eos,
    //         eos_idx,
    //         regression: candle_nn::linear(in_features, 1, vb)?,
    //     })
    // }

    // pub fn forward(&self, tokens: &Tensor, attentions: &Tensor) -> Result<Tensor> {
    //     let mut attns = attentions.clone();

    //     if self.append_eos {
    //         let eos_mask = tokens.ne_scalar(self.eos_idx.unwrap() as i64)?;
    //         let eos_mask = eos_mask.unsqueeze(1)?.matmul(&eos_mask.unsqueeze(2)?)?;
    //         attns = attns.broadcast_mul(&eos_mask.unsqueeze(1)?.unsqueeze(2)?)?;
    //         attns = attns.slice((.., .., .., ..-1, ..-1))?;
    //     }

    //     if self.prepend_bos {
    //         attns = attns.slice((.., .., .., 1.., 1..))?;
    //     }

    //     let (batch_size, layers, heads, seqlen, _) = attns.dims5()?;
    //     let attns = attns.reshape((batch_size, layers * heads, seqlen, seqlen))?;

    //     let attns = apc(&symmetrize(&attns)?)?;
    //     let attns = attns.permute((0, 2, 3, 1))?;
    //     let out = self.regression.forward(&attns)?;
    //     out.squeeze(3)?.sigmoid()
    // }
}

#[derive(Debug)]
pub struct NormalizedResidualBlock<T: Module> {
    layer: T,
    dropout: f64,
    layer_norm: ESM1bLayerNorm,
}

impl<T: Module> NormalizedResidualBlock<T> {
    // pub fn new(layer: T, embedding_dim: usize, dropout: f64) -> Result<Self> {
    //     let vb = VarBuilder::zeros();
    //     Ok(Self {
    //         layer,
    //         dropout,
    //         layer_norm: ESM1bLayerNorm::new(embedding_dim, 1e-12, true, vb)?,
    //     })
    // }

    //     pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
    //         let residual = x;
    //         let x = self.layer_norm.forward(x)?;
    //         let x = self.layer.forward(&x)?;
    //         let x = if self.dropout > 0. {
    //             x.dropout(self.dropout)?
    //         } else {
    //             x
    //         };
    //         x.add(residual)
    //     }

    //     pub fn forward_t<A, B>(&self, x: &Tensor, a: A, b: B) -> Result<(Tensor, Tensor)>
    //     where
    //         T: ModuleWithAttention<A, B>,
    //     {
    //         let residual = x;
    //         let x = self.layer_norm.forward(x)?;
    //         let (x, attn) = self.layer.forward_t(&x, a, b)?;
    //         let x = if self.dropout > 0. {
    //             x.dropout(self.dropout)?
    //         } else {
    //             x
    //         };
    //         let x = x.add(residual)?;
    //         Ok((x, attn))
    //     }
}

pub trait ModuleWithAttention<A, B> {
    fn forward_t(&self, x: &Tensor, a: A, b: B) -> Result<(Tensor, Tensor)>;
}

#[derive(Debug)]
pub struct FeedForwardNetwork {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    activation_dropout: f64,
}

impl FeedForwardNetwork {
    // pub fn new(
    //     embedding_dim: usize,
    //     ffn_embedding_dim: usize,
    //     activation_dropout: f64,
    //     _max_tokens_per_msa: usize,
    //     vb: VarBuilder,
    // ) -> Result<Self> {
    //     Ok(Self {
    //         fc1: candle_nn::linear(embedding_dim, ffn_embedding_dim, vb.pp("fc1"))?,
    //         fc2: candle_nn::linear(ffn_embedding_dim, embedding_dim, vb.pp("fc2"))?,
    //         activation_dropout,
    //     })
    // }
}

// impl Module for FeedForwardNetwork {
//     fn forward(&self, x: &Tensor) -> Result<Tensor> {
//         let x = gelu(&self.fc1.forward(x)?)?;
//         let x = if self.activation_dropout > 0. {
//             x.dropout(self.activation_dropout)?
//         } else {
//             x
//         };
//         self.fc2.forward(&x)
//     }
// }
