// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::esm2::ESM2Config;
use super::rotary_embedding::RotaryEmbedding;
use candle_core::{Device, Module, Result, Tensor, D};
use candle_nn::{self as nn, linear, ops, VarBuilder};

#[derive(Debug)]
pub struct FairseqIncrementalState {
    incremental_state_id: String,
}

impl FairseqIncrementalState {
    // pub fn new() -> Self {
    //     Self {
    //         incremental_state_id: Uuid::new_v4().to_string(),
    //     }
    // }

    fn get_full_incremental_state_key(&self, key: &str) -> String {
        format!("{}.{}", self.incremental_state_id, key)
    }

    // pub fn get_incremental_state(
    //     &self,
    //     incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
    //     key: &str,
    // ) -> Option<HashMap<String, Option<Tensor>>> {
    //     let full_key = self.get_full_incremental_state_key(key);
    //     incremental_state.and_then(|state| state.get(&full_key).cloned())
    // }

    // pub fn set_incremental_state(
    //     &self,
    //     incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
    //     key: &str,
    //     value: HashMap<String, Option<Tensor>>,
    // ) -> Option<HashMap<String, HashMap<String, Option<Tensor>>>> {
    //     if let Some(state) = incremental_state {
    //         let full_key = self.get_full_incremental_state_key(key);
    //         state.insert(full_key, value);
    //         Some(state.clone())
    //     } else {
    //         None
    //     }
    // }
}

#[derive(Debug)]
pub struct MultiheadAttention {
    // embed_dim: i64,
    // num_heads: i64,
    // kdim: i64,
    // vdim: i64,
    // qkv_same_dim: bool,
    // head_dim: i64,
    // scaling: f64,
    // self_attention: bool,
    // encoder_decoder_attention: bool,
    // bias_k: Option<Tensor>,
    // bias_v: Option<Tensor>,
    // add_zero_attn: bool,
    // onnx_trace: bool,
    // enable_torch_version: bool,
    // dropout: f64,
    // incremental_state: FairseqIncrementalState,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    rot_emb: Option<RotaryEmbedding>,
}

impl MultiheadAttention {
    pub fn load(vb: VarBuilder, config: &ESM2Config) -> Result<Self> {
        let ESM2Config {
            hidden_size,
            num_attention_heads,
            ..
        } = config;

        //  "num_attention_heads": 20,
        let embed_dim = *hidden_size as usize;
        let num_heads = *num_attention_heads as usize;
        let head_dim = embed_dim / num_heads;

        // Todo: need to double check  this....
        let kdim = *hidden_size as usize;
        let vdim = *hidden_size as usize;
        let qkv_same_dim = true;

        assert!(
            head_dim * num_heads == embed_dim,
            "embed_dim must be divisible by num_heads"
        );
        let scaling = (head_dim as f64).powf(-0.5);
        let q_proj = nn::linear(embed_dim, embed_dim, vb.pp("self.query"))?;
        let k_proj = nn::linear(kdim, embed_dim, vb.pp("self.key"))?;
        let v_proj = nn::linear(vdim, embed_dim, vb.pp("self.value"))?;
        let out_proj = nn::linear(embed_dim, embed_dim, vb.pp("output.dense"))?;
        let rot_emb = RotaryEmbedding::load(vb.pp("rotary_embeddings"), config)?;

        //     let (bias_k, bias_v) = if add_bias_kv {
        //         let bias_k = vb.get_with_hints("bias_k", &[1, 1, embed_dim], init::ZEROS)?;
        //         let bias_v = vb.get_with_hints("bias_v", &[1, 1, embed_dim], init::ZEROS)?;
        //         (Some(bias_k), Some(bias_v))
        //     } else {
        //         (None, None)
        //     };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            rot_emb: Some(rot_emb),
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
        // Todo: heck these
        let num_heads = 10;
        let head_dim = 10;

        let need_weights = need_weights || need_head_weights;
        let (tgt_len, bsz, embed_dim) = query.dims3()?;
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(query)?;
        let v = self.v_proj.forward(query)?;
        // let q = (q * config.scaling)?;
        let q = q
            .reshape((tgt_len, bsz * num_heads as usize, head_dim as usize))?
            .transpose(0, 1)?;
        k = k
            .reshape((D::minus1, bsz * num_heads as usize, head_dim as usize))?
            .transpose(0, 1)?;
        v = v
            .reshape((-1, bsz * num_heads as usize, head_dim as usize))?
            .transpose(0, 1)?;
        let src_len = k.dim(1)?;
        let (q, k) = rot_emb.forward(&q, &k)?;
        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let attn_weights = attn_weights.softmax(2)?;
        // let attn_weights = ops::dropout(&attn_weights, self.dropout, self.training)?;
        let attn = attn_weights.matmul(&v)?;
        let attn = attn
            .transpose(0, 1)?
            .reshape((tgt_len, bsz, self.embed_dim as usize))?;
        let attn = self.out_proj.forward(&attn)?;
        let attn_weights = if need_weights {
            let attn_weights = attn_weights
                .reshape((bsz, self.num_heads as usize, tgt_len, src_len))?
                .transpose(0, 1)?;
            if !need_head_weights {
                Some(attn_weights.mean(0)?)
            } else {
                Some(attn_weights)
            }
        } else {
            None
        };

        Ok((attn, attn_weights))
    }
}
