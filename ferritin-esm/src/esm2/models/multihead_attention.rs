// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use super::esm2::ESM2Config;
use super::rotary_embedding::RotaryEmbedding;
use candle_core::{Device, Module, Result, Tensor};
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
    // dropout: f64,
    // head_dim: i64,
    // scaling: f64,
    // self_attention: bool,
    // encoder_decoder_attention: bool,
    // bias_k: Option<Tensor>,
    // bias_v: Option<Tensor>,
    // add_zero_attn: bool,
    // onnx_trace: bool,
    // enable_torch_version: bool,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    rot_emb: Option<RotaryEmbedding>,
    // incremental_state: FairseqIncrementalState,
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

    // pub fn forward(
    //     &self,
    //     query: &Tensor,
    //     key: Option<&Tensor>,
    //     value: Option<&Tensor>,
    //     key_padding_mask: Option<&Tensor>,
    //     incremental_state: Option<&mut HashMap<String, HashMap<String, Option<Tensor>>>>,
    //     need_weights: bool,
    //     static_kv: bool,
    //     attn_mask: Option<&Tensor>,
    //     before_softmax: bool,
    //     need_head_weights: bool,
    // ) -> Result<(Tensor, Option<Tensor>)> {
    //     let need_weights = need_weights || need_head_weights;
    //     let (tgt_len, bsz, embed_dim) = query.dims3()?;
    //     // assert_eq!(embed_dim, self.embed_dim as usize);
    //     // if !self.rot_emb.is_some()
    //     //     && self.enable_torch_version
    //     //     && !self.onnx_trace
    //     //     && incremental_state.is_none()
    //     //     && !static_kv
    //     //     && !need_head_weights
    //     // {
    //     //     return multihead_attention_forward(
    //     //         query,
    //     //         key.unwrap(),
    //     //         value.unwrap(),
    //     //         self.embed_dim,
    //     //         self.num_heads,
    //     //         &self.q_proj,
    //     //         &self.k_proj,
    //     //         &self.v_proj,
    //     //         self.bias_k.as_ref(),
    //     //         self.bias_v.as_ref(),
    //     //         self.add_zero_attn,
    //     //         self.dropout,
    //     //         &self.out_proj,
    //     //         self.training,
    //     //         key_padding_mask,
    //     //         need_weights,
    //     //         attn_mask,
    //     //     );
    //     // }

    //     // let mut saved_state = None;
    //     // if let Some(inc_state) = incremental_state {
    //     //     saved_state = self.get_incremental_state(Some(inc_state), "attn_state");
    //     //     if let Some(saved_state_ref) = &saved_state {
    //     //         if saved_state_ref.contains_key("prev_key") && static_kv {
    //     //             assert!(self.encoder_decoder_attention && !self.self_attention);
    //     //         }
    //     //     }
    //     // }

    //     // let q = if self.self_attention {
    //     //     self.q_proj.forward(query)?
    //     // } else if self.encoder_decoder_attention {
    //     //     self.q_proj.forward(query)?
    //     // } else {
    //     //     assert!(key.is_some() && value.is_some());
    //     //     self.q_proj.forward(query)?
    //     // };

    //     // let k = if self.self_attention {
    //     //     self.k_proj.forward(query)?
    //     // } else if self.encoder_decoder_attention && key.is_none() {
    //     //     assert!(value.is_none());
    //     //     Tensor::zeros((), query.device())?
    //     // } else {
    //     //     assert!(key.is_some());
    //     //     self.k_proj.forward(key.unwrap())?
    //     // };

    //     // let v = if self.self_attention {
    //     //     self.v_proj.forward(query)?
    //     // } else if self.encoder_decoder_attention && value.is_none() {
    //     //     Tensor::zeros((), query.device())?
    //     // } else {
    //     //     assert!(value.is_some());
    //     //     self.v_proj.forward(value.unwrap())?
    //     // };

    //     // let q = (q * self.scaling)?;

    //     // let (mut k, mut v) = if self.bias_k.is_some() {
    //     //     assert!(self.bias_v.is_some());
    //     //     let bias_k = self.bias_k.as_ref().unwrap();
    //     //     let bias_v = self.bias_v.as_ref().unwrap();
    //     //     let bias_k = bias_k.broadcast_as((1, 1, bias_k.dim(2)?))?;
    //     //     let bias_v = bias_v.broadcast_as((1, 1, bias_v.dim(2)?))?;
    //     //     let k = Tensor::cat(&[&k, &bias_k], 1)?;
    //     //     let v = Tensor::cat(&[&v, &bias_v], 1)?;
    //     //     if let Some(attn_mask) = attn_mask {
    //     //         let attn_mask = Tensor::cat(
    //     //             &[
    //     //                 attn_mask,
    //     //                 Tensor::zeros((attn_mask.dim(0)?, 1), query.device())?,
    //     //             ],
    //     //             1,
    //     //         )?;
    //     //     }
    //     //     if let Some(key_padding_mask) = key_padding_mask {
    //     //         let key_padding_mask = Tensor::cat(
    //     //             &[
    //     //                 key_padding_mask,
    //     //                 Tensor::zeros((key_padding_mask.dim(0)?, 1), query.device())?,
    //     //             ],
    //     //             1,
    //     //         )?;
    //     //     }
    //     //     (k, v)
    //     // } else {
    //     //     (k, v)
    //     // };
    //     // let q = q
    //     //     .reshape((
    //     //         tgt_len,
    //     //         bsz * self.num_heads as usize,
    //     //         self.head_dim as usize,
    //     //     ))?
    //     //     .transpose(0, 1)?;
    //     // if k.dim()? > 0 {
    //     //     k = k
    //     //         .reshape((-1, bsz * self.num_heads as usize, self.head_dim as usize))?
    //     //         .transpose(0, 1)?;
    //     // }
    //     // if v.dim()? > 0 {
    //     //     v = v
    //     //         .reshape((-1, bsz * self.num_heads as usize, self.head_dim as usize))?
    //     //         .transpose(0, 1)?;
    //     // }
    //     // if let Some(saved_state_ref) = &mut saved_state {
    //     //     if saved_state_ref.contains_key("prev_key") {
    //     //         let prev_key = saved_state_ref.get("prev_key").unwrap().as_ref().unwrap();
    //     //         let prev_key = prev_key.reshape((
    //     //             bsz * self.num_heads as usize,
    //     //             -1,
    //     //             self.head_dim as usize,
    //     //         ))?;
    //     //         if static_kv {
    //     //             k = prev_key;
    //     //         } else {
    //     //             k = Tensor::cat(&[prev_key, &k], 1)?;
    //     //         }
    //     //     }
    //     //     if saved_state_ref.contains_key("prev_value") {
    //     //         let prev_value = saved_state_ref.get("prev_value").unwrap().as_ref().unwrap();
    //     //         let prev_value = prev_value.reshape((
    //     //             bsz * self.num_heads as usize,
    //     //             -1,
    //     //             self.head_dim as usize,
    //     //         ))?;
    //     //         if static_kv {
    //     //             v = prev_value;
    //     //         } else {
    //     //             v = Tensor::cat(&[prev_value, &v], 1)?;
    //     //         }
    //     //     }
    //     //     saved_state_ref.insert("prev_key".to_string(), Some(k.clone()));
    //     //     saved_state_ref.insert("prev_value".to_string(), Some(v.clone()));
    //     //     if let Some(inc_state) = incremental_state {
    //     //         self.set_incremental_state(Some(inc_state), "attn_state", saved_state_ref.clone());
    //     //     }
    //     // }
    //     // let src_len = k.dim(1)?;
    //     // if let Some(key_padding_mask) = key_padding_mask {
    //     //     assert_eq!(key_padding_mask.dim(0)? as i64, bsz);
    //     //     assert_eq!(key_padding_mask.dim(1)? as i64, src_len);
    //     // }
    //     // if self.add_zero_attn {
    //     //     src_len += 1;
    //     //     k = Tensor::cat(
    //     //         &[
    //     //             &k,
    //     //             Tensor::zeros((k.dim(0)?, 1, k.dim(2)?), query.device())?,
    //     //         ],
    //     //         1,
    //     //     )?;
    //     //     v = Tensor::cat(
    //     //         &[
    //     //             &v,
    //     //             Tensor::zeros((v.dim(0)?, 1, v.dim(2)?), query.device())?,
    //     //         ],
    //     //         1,
    //     //     )?;
    //     //     if let Some(attn_mask) = attn_mask {
    //     //         let attn_mask = Tensor::cat(
    //     //             &[
    //     //                 attn_mask,
    //     //                 Tensor::zeros((attn_mask.dim(0)?, 1), query.device())?,
    //     //             ],
    //     //             1,
    //     //         )?;
    //     //     }
    //     //     if let Some(key_padding_mask) = key_padding_mask {
    //     //         let key_padding_mask = Tensor::cat(
    //     //             &[
    //     //                 key_padding_mask,
    //     //                 Tensor::zeros((key_padding_mask.dim(0)?, 1), query.device())?,
    //     //             ],
    //     //             1,
    //     //         )?;
    //     //     }
    //     // }
    //     // if let Some(rot_emb) = &self.rot_emb {
    //     //     let (q, k) = rot_emb.forward(&q, &k)?;
    //     // }
    //     // let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
    //     // assert_eq!(
    //     //     attn_weights.dims(),
    //     //     &[bsz * self.num_heads as usize, tgt_len, src_len]
    //     // );
    //     // if let Some(attn_mask) = attn_mask {
    //     //     let attn_mask = attn_mask.unsqueeze(0)?;
    //     //     if self.onnx_trace {
    //     //         let attn_mask = attn_mask.repeat((attn_weights.dim(0)?, 1, 1))?;
    //     //     }
    //     //     attn_weights = (&attn_weights + attn_mask)?;
    //     // }
    //     // if let Some(key_padding_mask) = key_padding_mask {
    //     //     let attn_weights =
    //     //         attn_weights.reshape((bsz, self.num_heads as usize, tgt_len, src_len))?;

    //     //     let key_padding_mask = key_padding_mask
    //     //         .unsqueeze(1)?
    //     //         .unsqueeze(2)?
    //     //         .to_dtype(candle_core::DType::Bool)?;
    //     //     let attn_weights = attn_weights
    //     //         .masked_fill(&key_padding_mask, f32::NEG_INFINITY)?
    //     //         .reshape((bsz * self.num_heads as usize, tgt_len, src_len))?;
    //     // }
    //     // if before_softmax {
    //     //     return Ok((attn_weights, Some(v)));
    //     // }
    //     // let attn_weights = if self.onnx_trace {
    //     //     attn_weights.to_dtype(candle_core::DType::F32)?.softmax(2)?
    //     // } else {
    //     //     attn_weights.softmax(2)?
    //     // };
    //     // let attn_weights = ops::dropout(&attn_weights, self.dropout, self.training)?;
    //     // let attn = attn_weights.matmul(&v)?;
    //     // assert_eq!(
    //     //     attn.dims(),
    //     //     &[
    //     //         bsz * self.num_heads as usize,
    //     //         tgt_len,
    //     //         self.head_dim as usize
    //     //     ]
    //     // );
    //     // let attn = if self.onnx_trace && attn.dim(1)? == 1 {
    //     //     attn.reshape((tgt_len, bsz, self.embed_dim as usize))?
    //     // } else {
    //     //     attn.transpose(0, 1)?
    //     //         .reshape((tgt_len, bsz, self.embed_dim as usize))?
    //     // };
    //     // let attn = self.out_proj.forward(&attn)?;
    //     // let attn_weights = if need_weights {
    //     //     let attn_weights = attn_weights
    //     //         .reshape((bsz, self.num_heads as usize, tgt_len, src_len))?
    //     //         .transpose(0, 1)?;
    //     //     if !need_head_weights {
    //     //         Some(attn_weights.mean(0)?)
    //     //     } else {
    //     //         Some(attn_weights)
    //     //     }
    //     // } else {
    //     //     None
    //     // };

    //     Ok((attn, attn_weights))
    // }

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
