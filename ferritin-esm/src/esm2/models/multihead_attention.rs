// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{init, linear, ops, VarBuilder};
use std::collections::HashMap;
// use uuid::Uuid;

// pub fn utils_softmax(x: &Tensor, dim: i64, onnx_trace: bool) -> Result<Tensor> {
//     if onnx_trace {
//         x.to_dtype(candle_core::DType::F32)?.softmax(dim)
//     } else {
//         x.softmax(dim)
//     }
// }

pub struct FairseqIncrementalState {
    incremental_state_id: String,
}

impl FairseqIncrementalState {
    pub fn new() -> Self {
        Self {
            incremental_state_id: Uuid::new_v4().to_string(),
        }
    }

    fn get_full_incremental_state_key(&self, key: &str) -> String {
        format!("{}.{}", self.incremental_state_id, key)
    }

    pub fn get_incremental_state(
        &self,
        incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
        key: &str,
    ) -> Option<HashMap<String, Option<Tensor>>> {
        let full_key = self.get_full_incremental_state_key(key);
        incremental_state.and_then(|state| state.get(&full_key).cloned())
    }

    pub fn set_incremental_state(
        &self,
        incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
        key: &str,
        value: HashMap<String, Option<Tensor>>,
    ) -> Option<HashMap<String, HashMap<String, Option<Tensor>>>> {
        if let Some(state) = incremental_state {
            let full_key = self.get_full_incremental_state_key(key);
            state.insert(full_key, value);
            Some(state.clone())
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub struct MultiheadAttention {
    embed_dim: i64,
    num_heads: i64,
    kdim: i64,
    vdim: i64,
    qkv_same_dim: bool,
    dropout: f64,
    head_dim: i64,
    scaling: f64,
    self_attention: bool,
    encoder_decoder_attention: bool,
    q_proj: linear::Linear,
    k_proj: linear::Linear,
    v_proj: linear::Linear,
    out_proj: linear::Linear,
    bias_k: Option<Tensor>,
    bias_v: Option<Tensor>,
    add_zero_attn: bool,
    rot_emb: Option<RotaryEmbedding>,
    onnx_trace: bool,
    enable_torch_version: bool,
    incremental_state: FairseqIncrementalState,
}

impl MultiheadAttention {
    pub fn new(
        vb: VarBuilder,
        embed_dim: i64,
        num_heads: i64,
        kdim: Option<i64>,
        vdim: Option<i64>,
        dropout: f64,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        self_attention: bool,
        encoder_decoder_attention: bool,
        use_rotary_embeddings: bool,
    ) -> Result<Self> {
        let kdim = kdim.unwrap_or(embed_dim);
        let vdim = vdim.unwrap_or(embed_dim);
        let qkv_same_dim = kdim == embed_dim && vdim == embed_dim;

        let head_dim = embed_dim / num_heads;
        assert!(
            head_dim * num_heads == embed_dim,
            "embed_dim must be divisible by num_heads"
        );
        let scaling = (head_dim as f64).powf(-0.5);

        assert!(
            !self_attention || qkv_same_dim,
            "Self-attention requires query, key and value to be of the same size"
        );

        let q_proj = linear::linear(embed_dim, embed_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear::linear(kdim, embed_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear::linear(vdim, embed_dim, bias, vb.pp("v_proj"))?;

        let out_proj = linear::linear(embed_dim, embed_dim, bias, vb.pp("out_proj"))?;

        let (bias_k, bias_v) = if add_bias_kv {
            let bias_k = vb.get_with_hints("bias_k", &[1, 1, embed_dim], init::ZEROS)?;
            let bias_v = vb.get_with_hints("bias_v", &[1, 1, embed_dim], init::ZEROS)?;
            (Some(bias_k), Some(bias_v))
        } else {
            (None, None)
        };

        let rot_emb = if use_rotary_embeddings {
            Some(RotaryEmbedding::new(head_dim)?)
        } else {
            None
        };

        Ok(Self {
            embed_dim,
            num_heads,
            kdim,
            vdim,
            qkv_same_dim,
            dropout,
            head_dim,
            scaling,
            self_attention,
            encoder_decoder_attention,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            bias_k,
            bias_v,
            add_zero_attn,
            rot_emb,
            onnx_trace: false,
            enable_torch_version: false,
            incremental_state: FairseqIncrementalState::new(),
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: Option<&Tensor>,
        value: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
        incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
        need_weights: bool,
        static_kv: bool,
        attn_mask: Option<&Tensor>,
        before_softmax: bool,
        need_head_weights: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Forward implementation
        unimplemented!()
    }
}
