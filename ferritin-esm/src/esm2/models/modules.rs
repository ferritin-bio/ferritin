// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::axial_attention::{ColumnSelfAttention, RowSelfAttention};
use super::multihead_attention::MultiheadAttention;
use crate::ESM2Config;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{self as nn, LayerNorm, VarBuilder};
use std::f64::consts::PI;

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

// #[derive(Debug)]
// pub struct ESM1LayerNorm {
//     layernorm: LayerNorm,
// }

// impl ESM1LayerNorm {
//     pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
//         let ln_conf = nn::LayerNormConfig {
//             eps: 1e-5,
//             remove_mean: true,
//             affine: true,
//         };
//         let layernorm = nn::layer_norm((100), ln_conf, vb.pp("LayerNorm"))?;
//         Ok(Self { layernorm })
//     }

// pub fn new(size: usize, eps: f64, affine: bool, vb: VarBuilder) -> Result<Self> {
//     let weight = if affine {
//         vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.))?
//     } else {
//         Tensor::ones(size, &Device::Cpu)?
//     };

//     let bias = if affine {
//         Some(vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.))?)
//     } else {
//         None
//     };

//     Ok(Self { weight, bias, eps })
// }
// }

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

// pub type ESM1bLayerNorm = ESM1LayerNorm;

#[derive(Debug)]
pub struct TransformerLayer {
    self_attn: MultiheadAttention,
    self_attn_layer_norm: LayerNorm,
    // fc1: nn::Linear,
    // fc2: nn::Linear,
    final_layer_norm: LayerNorm,
}

impl TransformerLayer {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let ESM2Config {
            hidden_size,
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
        let ln_conf = nn::LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        };
        let layer_norm = nn::layer_norm((*hidden_size as usize), ln_conf, vb.pp("LayerNorm"))?;
        let multi_head = MultiheadAttention::load(vb.pp("attention"), config)?;
        // let fc1 = nn::linear(embed_dim, ffn_embed_dim, vb.pp("fc1"))?;
        // let fc2 = nn::linear(ffn_embed_dim, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm =
            nn::layer_norm((*hidden_size as usize), ln_conf, vb.pp("LayerNorm"))?;

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

    pub fn forward(
        &self,
        x: &Tensor,
        self_attn_mask: Option<&Tensor>,
        self_attn_padding_mask: Option<&Tensor>,
        need_head_weights: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let (x, attn) = self.self_attn.forward_t(
            &x,
            &x,
            &x,
            self_attn_padding_mask,
            need_head_weights,
            self_attn_mask,
        )?;
        // let x = x.add(residual)?;

        // let residual = &x;
        // let x = self.final_layer_norm.forward(&x)?;
        // let x = gelu(&self.fc1.forward(&x)?)?;
        // let x = self.fc2.forward(&x)?;
        // let x = x.add(residual)?;

        Ok((x, attn))
    }
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
    layer_norm: LayerNorm,
}

impl RobertaLMHead {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let ESM2Config { hidden_size, .. } = config;
        let dense = nn::linear(*hidden_size as usize, *hidden_size as usize, vb.pp("dense"))?;
        let ln_conf = nn::LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        };
        let layer_norm = nn::layer_norm((*hidden_size as usize), ln_conf, vb.pp("layer_norm"))?;
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
    layer_norm: LayerNorm,
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
