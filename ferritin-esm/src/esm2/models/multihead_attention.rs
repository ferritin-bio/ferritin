// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::esm2::ESM2Config;
use super::rotary_embedding::RotaryEmbedding;
use candle_core::{Device, Module, Result, Tensor, D},;
use candle_nn::{self as nn, linear, ops, VarBuilder};

// pub fn utils_softmax(x: &Tensor, dim: i64, onnx_trace: bool) -> Result<Tensor> {
//     if onnx_trace {
//         x.to_dtype(candle_core::DType::F32)?.softmax(dim)
//     } else {
//         x.softmax(dim)
//     }
// }

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
        v = v.reshape((-1, bsz * num_heads as usize, head_dim as usize))?
            .transpose(0, 1)?;
        let src_len = k.dim(1)?;
        let (q, k) = rot_emb.forward(&q, &k)?;
        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let attn_weights = attn_weights.softmax(2)?;
        // let attn_weights = ops::dropout(&attn_weights, self.dropout, self.training)?;
        let attn = attn_weights.matmul(&v)?;
        let attn = attn.transpose(0, 1)?
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

    // fn _append_prev_key_padding_mask(
    //     key_padding_mask: Option<&Tensor>,
    //     prev_key_padding_mask: Option<&Tensor>,
    //     batch_size: usize,
    //     src_len: usize,
    //     static_kv: bool,
    // ) -> Result<Option<Tensor>> {
    //     let mut new_key_padding_mask = if prev_key_padding_mask.is_some() && static_kv {
    //         prev_key_padding_mask.cloned()
    //     } else if prev_key_padding_mask.is_some() && key_padding_mask.is_some() {
    //         let prev_key_padding_mask = prev_key_padding_mask
    //             .unwrap()
    //             .to_dtype(candle_core::DType::F32)?;
    //         let key_padding_mask = key_padding_mask
    //             .unwrap()
    //             .to_dtype(candle_core::DType::F32)?;
    //         Some(Tensor::cat(&[prev_key_padding_mask, key_padding_mask], 1)?)
    //     } else if prev_key_padding_mask.is_some() {
    //         let prev_key_padding_mask = prev_key_padding_mask.unwrap();
    //         let filler = Tensor::zeros(
    //             (batch_size, src_len - prev_key_padding_mask.dim(1)?),
    //             prev_key_padding_mask.device(),
    //         )?;
    //         Some(Tensor::cat(
    //             &[
    //                 prev_key_padding_mask.to_dtype(candle_core::DType::F32)?,
    //                 filler,
    //             ],
    //             1,
    //         )?)
    //     } else if key_padding_mask.is_some() {
    //         let key_padding_mask = key_padding_mask.unwrap();
    //         let filler = Tensor::zeros(
    //             (batch_size, src_len - key_padding_mask.dim(1)?),
    //             key_padding_mask.device(),
    //         )?;
    //         Some(Tensor::cat(
    //             &[filler, key_padding_mask.to_dtype(candle_core::DType::F32)?],
    //             1,
    //         )?)
    //     } else {
    //         None
    //     };

    //     Ok(new_key_padding_mask)
    // }

    // pub fn reorder_incremental_state(
    //     &self,
    //     incremental_state: &mut HashMap<String, HashMap<String, Option<Tensor>>>,
    //     new_order: &Tensor,
    // ) -> Result<()> {
    //     let mut input_buffer = self
    //         .get_incremental_state(Some(incremental_state), "attn_state")
    //         .unwrap_or_default();

    //     for (k, v) in input_buffer.iter_mut() {
    //         if let Some(tensor) = v {
    //             if self.encoder_decoder_attention && tensor.dim(0)? == new_order.dim(0)? {
    //                 break;
    //             }
    //             *v = Some(tensor.index_select(0, new_order)?);
    //         }
    //     }

    //     if !input_buffer.is_empty() {
    //         self.set_incremental_state(Some(incremental_state), "attn_state", input_buffer);
    //     }

    //     Ok(())
    // }

    // fn _get_input_buffer(
    //     &self,
    //     incremental_state: Option<&HashMap<String, HashMap<String, Option<Tensor>>>>,
    // ) -> HashMap<String, Option<Tensor>> {
    //     self.get_incremental_state(incremental_state, "attn_state")
    //         .unwrap_or_default()
    // }

    // fn _set_input_buffer(
    //     &self,
    //     incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
    //     buffer: HashMap<String, Option<Tensor>>,
    // ) -> Option<HashMap<String, HashMap<String, Option<Tensor>>>> {
    //     self.set_incremental_state(incremental_state, "attn_state", buffer)
    // }

    // fn apply_sparse_mask(
    //     attn_weights: Tensor,
    //     tgt_len: usize,
    //     src_len: usize,
    //     bsz: usize,
    // ) -> Result<Tensor> {
    //     Ok(attn_weights)
    // }

    // pub fn upgrade_state_dict_named(
    //     &self,
    //     state_dict: &mut HashMap<String, Tensor>,
    //     name: &str,
    // ) -> Result<()> {
    //     let prefix = if name.is_empty() {
    //         String::new()
    //     } else {
    //         format!("{}.", name)
    //     };

    //     let mut items_to_add = HashMap::new();
    //     let mut keys_to_remove = Vec::new();

    //     for k in state_dict.keys() {
    //         if k.ends_with(&format!("{}in_proj_weight", prefix)) {
    //             let dim = state_dict[k].dim(0)? / 3;
    //             items_to_add.insert(
    //                 format!("{}q_proj.weight", prefix),
    //                 state_dict[k].narrow(0, 0, dim)?,
    //             );
    //             items_to_add.insert(
    //                 format!("{}k_proj.weight", prefix),
    //                 state_dict[k].narrow(0, dim, dim)?,
    //             );
    //             items_to_add.insert(
    //                 format!("{}v_proj.weight", prefix),
    //                 state_dict[k].narrow(0, 2 * dim, dim)?,
    //             );

    //             keys_to_remove.push(k.clone());

    //             let k_bias = format!("{}in_proj_bias", prefix);
    //             if state_dict.contains_key(&k_bias) {
    //                 let dim = state_dict[&k_bias].dim(0)? / 3;
    //                 items_to_add.insert(
    //                     format!("{}q_proj.bias", prefix),
    //                     state_dict[&k_bias].narrow(0, 0, dim)?,
    //                 );
    //                 items_to_add.insert(
    //                     format!("{}k_proj.bias", prefix),
    //                     state_dict[&k_bias].narrow(0, dim, dim)?,
    //                 );
    //                 items_to_add.insert(
    //                     format!("{}v_proj.bias", prefix),
    //                     state_dict[&k_bias].narrow(0, 2 * dim, dim)?,
    //                 );

    //                 keys_to_remove.push(k_bias);
    //             }
    //         }
    //     }

    //     for k in keys_to_remove {
    //         state_dict.remove(&k);
    //     }

    //     for (k, v) in items_to_add {
    //         state_dict.insert(k, v);
    //     }

    //     Ok(())
    // }
}
