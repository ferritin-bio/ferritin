/// Mutihead attention based on the ESM2 pytorch code
///
// # Copyright (c) Meta Platforms, Inc. and affiliates.
// #
// # This source code is licensed under the MIT license found in the
// # LICENSE file in the root directory of this source tree.
//
// import math
// from typing import Dict, Optional, Tuple
// import torch
// import torch.nn.functional as F
// from torch import Tensor, nn
// from torch.nn import Parameter
// from esm.rotary_embedding import RotaryEmbedding
// import uuid
//
// def utils_softmax(x, dim: int, onnx_trace: bool = False):
//     if onnx_trace:
//         return F.softmax(x.float(), dim=dim)
//     else:
//         return F.softmax(x, dim=dim, dtype=torch.float32)
//
// class FairseqIncrementalState(object):
//     def __init__(self, *args, **kwargs):
//         super().__init__(*args, **kwargs)
//         self.init_incremental_state()

//     def init_incremental_state(self):
//         self._incremental_state_id = str(uuid.uuid4())

//     def _get_full_incremental_state_key(self, key: str) -> str:
//         return "{}.{}".format(self._incremental_state_id, key)

//     def get_incremental_state(
//         self,
//         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
//         key: str,
//     ) -> Optional[Dict[str, Optional[Tensor]]]:
//         """Helper for getting incremental state for an nn.Module."""
//         full_key = self._get_full_incremental_state_key(key)
//         if incremental_state is None or full_key not in incremental_state:
//             return None
//         return incremental_state[full_key]
//
//     def set_incremental_state(
//         self,
//         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
//         key: str,
//         value: Dict[str, Optional[Tensor]],
//     ) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
//         """Helper for setting incremental state for an nn.Module."""
//         if incremental_state is not None:
//             full_key = self._get_full_incremental_state_key(key)
//             incremental_state[full_key] = value
//         return incremental_state
//
// def with_incremental_state(cls):
//     cls.__bases__ = (FairseqIncrementalState,) + tuple(
//         b for b in cls.__bases__ if b != FairseqIncrementalState
//     )
//     return cls
//
// @with_incremental_state
// class MultiheadAttention(nn.Module):
//     """Multi-headed attention.
//
//     See "Attention Is All You Need" for more details.
//     """
//     def __init__(
//         self,
//         embed_dim,
//         num_heads,
//         kdim=None,
//         vdim=None,
//         dropout=0.0,
//         bias=True,
//         add_bias_kv: bool = False,
//         add_zero_attn: bool = False,
//         self_attention: bool = False,
//         encoder_decoder_attention: bool = False,
//         use_rotary_embeddings: bool = False,
//     ):
//         super().__init__()
//         self.embed_dim = embed_dim
//         self.kdim = kdim if kdim is not None else embed_dim
//         self.vdim = vdim if vdim is not None else embed_dim
//         self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
//         self.num_heads = num_heads
//         self.dropout = dropout
//         self.head_dim = embed_dim // num_heads
//         assert (
//             self.head_dim * num_heads == self.embed_dim
//         ), "embed_dim must be divisible by num_heads"
//         self.scaling = self.head_dim**-0.5
//         self.self_attention = self_attention
//         self.encoder_decoder_attention = encoder_decoder_attention
//         assert not self.self_attention or self.qkv_same_dim, (
//             "Self-attention requires query, key and " "value to be of the same size"
//         )
//         self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
//         self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
//         self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
//         self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
//         if add_bias_kv:
//             self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
//             self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
//         else:
//             self.bias_k = self.bias_v = None
//         self.add_zero_attn = add_zero_attn
//         self.reset_parameters()
//         self.onnx_trace = False
//         self.rot_emb = None
//         if use_rotary_embeddings:
//             self.rot_emb = RotaryEmbedding(dim=self.head_dim)
//         self.enable_torch_version = False
//         if hasattr(F, "multi_head_attention_forward"):
//             self.enable_torch_version = True
//         else:
//             self.enable_torch_version = False
//     def prepare_for_onnx_export_(self):
//         self.onnx_trace = True
//
//     def reset_parameters(self):
//         if self.qkv_same_dim:
//             # Empirically observed the convergence to be much better with
//             # the scaled initialization
//             nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
//             nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
//             nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
//         else:
//             nn.init.xavier_uniform_(self.k_proj.weight)
//             nn.init.xavier_uniform_(self.v_proj.weight)
//             nn.init.xavier_uniform_(self.q_proj.weight)
//
//         nn.init.xavier_uniform_(self.out_proj.weight)
//         if self.out_proj.bias is not None:
//             nn.init.constant_(self.out_proj.bias, 0.0)
//         if self.bias_k is not None:
//             nn.init.xavier_normal_(self.bias_k)
//         if self.bias_v is not None:
//             nn.init.xavier_normal_(self.bias_v)
//
//     def forward(
//         self,
//         query,
//         key: Optional[Tensor],
//         value: Optional[Tensor],
//         key_padding_mask: Optional[Tensor] = None,
//         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
//         need_weights: bool = True,
//         static_kv: bool = False,
//         attn_mask: Optional[Tensor] = None,
//         before_softmax: bool = False,
//         need_head_weights: bool = False,
//     ) -> Tuple[Tensor, Optional[Tensor]]:
//         """Input shape: Time x Batch x Channel
//
//         Args:
//             key_padding_mask (ByteTensor, optional): mask to exclude
//                 keys that are pads, of shape `(batch, src_len)`, where
//                 padding elements are indicated by 1s.
//             need_weights (bool, optional): return the attention weights,
//                 averaged over heads (default: False).
//             attn_mask (ByteTensor, optional): typically used to
//                 implement causal attention, where the mask prevents the
//                 attention from looking forward in time (default: None).
//             before_softmax (bool, optional): return the raw attention
//                 weights and values before the attention softmax.
//             need_head_weights (bool, optional): return the attention
//                 weights for each head. Implies *need_weights*. Default:
//                 return the average attention weights over all heads.
//         """
//         if need_head_weights:
//             need_weights = True
//
//         tgt_len, bsz, embed_dim = query.size()
//         assert embed_dim == self.embed_dim
//         assert list(query.size()) == [tgt_len, bsz, embed_dim]
//
//         if (
//             not self.rot_emb
//             and self.enable_torch_version
//             and not self.onnx_trace
//             and incremental_state is None
//             and not static_kv
//             # A workaround for quantization to work. Otherwise JIT compilation
//             # treats bias in linear module as method.
//             and not torch.jit.is_scripting()
//             and not need_head_weights
//         ):
//             assert key is not None and value is not None
//             return F.multi_head_attention_forward(
//                 query,
//                 key,
//                 value,
//                 self.embed_dim,
//                 self.num_heads,
//                 torch.empty([0]),
//                 torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
//                 self.bias_k,
//                 self.bias_v,
//                 self.add_zero_attn,
//                 self.dropout,
//                 self.out_proj.weight,
//                 self.out_proj.bias,
//                 self.training,
//                 key_padding_mask,
//                 need_weights,
//                 attn_mask,
//                 use_separate_proj_weight=True,
//                 q_proj_weight=self.q_proj.weight,
//                 k_proj_weight=self.k_proj.weight,
//                 v_proj_weight=self.v_proj.weight,
//             )
//         if incremental_state is not None:
//             saved_state = self._get_input_buffer(incremental_state)
//             if saved_state is not None and "prev_key" in saved_state:
//                 # previous time steps are cached - no need to recompute
//                 # key and value if they are static
//                 if static_kv:
//                     assert self.encoder_decoder_attention and not self.self_attention
//                     key = value = None
//         else:
//             saved_state = None
//
//         if self.self_attention:
//             q = self.q_proj(query)
//             k = self.k_proj(query)
//             v = self.v_proj(query)
//         elif self.encoder_decoder_attention:
//             # encoder-decoder attention
//             q = self.q_proj(query)
//             if key is None:
//                 assert value is None
//                 k = v = None
//             else:
//                 k = self.k_proj(key)
//                 v = self.v_proj(key)
//
//         else:
//             assert key is not None and value is not None
//             q = self.q_proj(query)
//             k = self.k_proj(key)
//             v = self.v_proj(value)
//         q *= self.scaling
//
//         if self.bias_k is not None:
//             assert self.bias_v is not None
//             k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
//             v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
//             if attn_mask is not None:
//                 attn_mask = torch.cat(
//                     [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
//                 )
//             if key_padding_mask is not None:
//                 key_padding_mask = torch.cat(
//                     [
//                         key_padding_mask,
//                         key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
//                     ],
//                     dim=1,
//                 )
//
//         q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
//         if k is not None:
//             k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
//         if v is not None:
//             v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
//         if saved_state is not None:
//             # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
//             if "prev_key" in saved_state:
//                 _prev_key = saved_state["prev_key"]
//                 assert _prev_key is not None
//                 prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
//                 if static_kv:
//                     k = prev_key
//                 else:
//                     assert k is not None
//                     k = torch.cat([prev_key, k], dim=1)
//             if "prev_value" in saved_state:
//                 _prev_value = saved_state["prev_value"]
//                 assert _prev_value is not None
//                 prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
//                 if static_kv:
//                     v = prev_value
//                 else:
//                     assert v is not None
//                     v = torch.cat([prev_value, v], dim=1)
//             prev_key_padding_mask: Optional[Tensor] = None
//             if "prev_key_padding_mask" in saved_state:
//                 prev_key_padding_mask = saved_state["prev_key_padding_mask"]
//             assert k is not None and v is not None
//             key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
//                 key_padding_mask=key_padding_mask,
//                 prev_key_padding_mask=prev_key_padding_mask,
//                 batch_size=bsz,
//                 src_len=k.size(1),
//                 static_kv=static_kv,
//             )
//             saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
//             saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
//             saved_state["prev_key_padding_mask"] = key_padding_mask
//             # In this branch incremental_state is never None
//             assert incremental_state is not None
//             incremental_state = self._set_input_buffer(incremental_state, saved_state)
//         assert k is not None
//         src_len = k.size(1)
//
//         # This is part of a workaround to get around fork/join parallelism
//         # not supporting Optional types.
//         if key_padding_mask is not None and key_padding_mask.dim() == 0:
//             key_padding_mask = None
//
//         if key_padding_mask is not None:
//             assert key_padding_mask.size(0) == bsz
//             assert key_padding_mask.size(1) == src_len
//
//         if self.add_zero_attn:
//             assert v is not None
//             src_len += 1
//             k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
//             v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
//             if attn_mask is not None:
//                 attn_mask = torch.cat(
//                     [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
//                 )
//             if key_padding_mask is not None:
//                 key_padding_mask = torch.cat(
//                     [
//                         key_padding_mask,
//                         torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask),
//                     ],
//                     dim=1,
//                 )
//
//         if self.rot_emb:
//             q, k = self.rot_emb(q, k)
//
//         attn_weights = torch.bmm(q, k.transpose(1, 2))
//         attn_weights = MultiheadAttention.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)
//
//         assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
//
//         if attn_mask is not None:
//             attn_mask = attn_mask.unsqueeze(0)
//             if self.onnx_trace:
//                 attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
//             attn_weights += attn_mask
//
//         if key_padding_mask is not None:
//             # don't attend to padding symbols
//             attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
//             attn_weights = attn_weights.masked_fill(
//                 key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
//             )
//             attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
//
//         if before_softmax:
//             return attn_weights, v
//
//         attn_weights_float = utils_softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
//         attn_weights = attn_weights_float.type_as(attn_weights)
//         attn_probs = F.dropout(
//             attn_weights_float.type_as(attn_weights),
//             p=self.dropout,
//             training=self.training,
//         )
//         assert v is not None
//         attn = torch.bmm(attn_probs, v)
//         assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
//         if self.onnx_trace and attn.size(1) == 1:
//             # when ONNX tracing a single decoder step (sequence length == 1)
//             # the transpose is a no-op copy before view, thus unnecessary
//             attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
//         else:
//             attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
//         attn = self.out_proj(attn)
//         attn_weights: Optional[Tensor] = None
//         if need_weights:
//             attn_weights = attn_weights_float.view(
//                 bsz, self.num_heads, tgt_len, src_len
//             ).type_as(attn).transpose(1, 0)
//             if not need_head_weights:
//                 # average attention weights over heads
//                 attn_weights = attn_weights.mean(dim=0)
//
//         return attn, attn_weights
//
//     @staticmethod
//     def _append_prev_key_padding_mask(
//         key_padding_mask: Optional[Tensor],
//         prev_key_padding_mask: Optional[Tensor],
//         batch_size: int,
//         src_len: int,
//         static_kv: bool,
//     ) -> Optional[Tensor]:
//         # saved key padding masks have shape (bsz, seq_len)
//         if prev_key_padding_mask is not None and static_kv:
//             new_key_padding_mask = prev_key_padding_mask
//         elif prev_key_padding_mask is not None and key_padding_mask is not None:
//             new_key_padding_mask = torch.cat(
//                 [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
//             )
//         # During incremental decoding, as the padding token enters and
//         # leaves the frame, there will be a time when prev or current
//         # is None
//         elif prev_key_padding_mask is not None:
//             filler = torch.zeros(
//                 (batch_size, src_len - prev_key_padding_mask.size(1)),
//                 device=prev_key_padding_mask.device,
//             )
//             new_key_padding_mask = torch.cat(
//                 [prev_key_padding_mask.float(), filler.float()], dim=1
//             )
//         elif key_padding_mask is not None:
//             filler = torch.zeros(
//                 (batch_size, src_len - key_padding_mask.size(1)),
//                 device=key_padding_mask.device,
//             )
//             new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
//         else:
//             new_key_padding_mask = prev_key_padding_mask
//         return new_key_padding_mask
//
//     @torch.jit.export
//     def reorder_incremental_state(
//         self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]], new_order: Tensor
//     ):
//         """Reorder buffered internal state (for incremental generation)."""
//         input_buffer = self._get_input_buffer(incremental_state)
//         if input_buffer is not None:
//             for k in input_buffer.keys():
//                 input_buffer_k = input_buffer[k]
//                 if input_buffer_k is not None:
//                     if self.encoder_decoder_attention and input_buffer_k.size(0) == new_order.size(
//                         0
//                     ):
//                         break
//                     input_buffer[k] = input_buffer_k.index_select(0, new_order)
//             incremental_state = self._set_input_buffer(incremental_state, input_buffer)
//         return incremental_state
//
//     def _get_input_buffer(
//         self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
//     ) -> Dict[str, Optional[Tensor]]:
//         result = self.get_incremental_state(incremental_state, "attn_state")
//         if result is not None:
//             return result
//         else:
//             empty_result: Dict[str, Optional[Tensor]] = {}
//             return empty_result
//
//     def _set_input_buffer(
//         self,
//         incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
//         buffer: Dict[str, Optional[Tensor]],
//     ):
//         return self.set_incremental_state(incremental_state, "attn_state", buffer)
//
//     def apply_sparse_mask(attn_weights, tgt_len: int, src_len: int, bsz: int):
//         return attn_weights
//
//     def upgrade_state_dict_named(self, state_dict, name):
//         prefix = name + "." if name != "" else ""
//         items_to_add = {}
//         keys_to_remove = []
//         for k in state_dict.keys():
//             if k.endswith(prefix + "in_proj_weight"):
//                 # in_proj_weight used to be q + k + v with same dimensions
//                 dim = int(state_dict[k].shape[0] / 3)
//                 items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
//                 items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
//                 items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]
//                 keys_to_remove.append(k)
//                 k_bias = prefix + "in_proj_bias"
//                 if k_bias in state_dict.keys():
//                     dim = int(state_dict[k].shape[0] / 3)
//                     items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
//                     items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim : 2 * dim]
//                     items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]
//                     keys_to_remove.append(prefix + "in_proj_bias")
//         for k in keys_to_remove:
//             del state_dict[k]
//         for key, value in items_to_add.items():
//             state_dict[key] = value
//
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
use super::esm2::ESM2Config;
use super::rotary_embedding::RotaryEmbedding;
use candle_core::{Result, Tensor};
use candle_nn::init;
use candle_nn::ops;
use candle_nn::{self as nn, VarBuilder};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug)]
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
    embed_dim: usize,
    num_heads: usize,
    kdim: usize,
    vdim: usize,
    qkv_same_dim: bool,
    dropout: f64,
    head_dim: usize,
    scaling: f64,
    self_attention: bool,
    encoder_decoder_attention: bool,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    out_proj: nn::Linear,
    bias_k: Option<Tensor>,
    bias_v: Option<Tensor>,
    add_zero_attn: bool,
    onnx_trace: bool,
    enable_torch_version: bool,
    rot_emb: Option<RotaryEmbedding>,
    incremental_state: FairseqIncrementalState,
    training: bool,
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
        let kdim = *hidden_size as usize;
        let vdim = *hidden_size as usize;
        let qkv_same_dim = true;
        let dropout = *attention_dropout;
        let add_bias_kv = false; // Set default value
        let self_attention = true; // Default for ESM2
        let encoder_decoder_attention = false; // Default for ESM2
        let add_zero_attn = false; // Default for ESM2
        let onnx_trace = false;
        let enable_torch_version = false;
        let training = false; // Default to eval mode

        assert_eq!(
            head_dim * num_heads,
            embed_dim,
            "embed_dim must be divisible by num_heads"
        );
        let scaling = (head_dim as f64).powf(-0.5);
        let q_proj = nn::linear(embed_dim, embed_dim, vb.pp("self.query"))?;
        let k_proj = nn::linear(kdim, embed_dim, vb.pp("self.key"))?;
        let v_proj = nn::linear(vdim, embed_dim, vb.pp("self.value"))?;
        let out_proj = nn::linear(embed_dim, embed_dim, vb.pp("output.dense"))?;
        let rot_emb = RotaryEmbedding::load(vb.pp("rotary_embeddings"), config)?;

        let (bias_k, bias_v) = if add_bias_kv {
            let bias_k = vb.get_with_hints("bias_k", &[1, 1, embed_dim], init::ZEROS)?;
            let bias_v = vb.get_with_hints("bias_v", &[1, 1, embed_dim], init::ZEROS)?;
            (Some(bias_k), Some(bias_v))
        } else {
            (None, None)
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
            onnx_trace,
            enable_torch_version,
            rot_emb: Some(rot_emb),
            incremental_state: FairseqIncrementalState::new(),
            training,
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
        let need_weights = need_weights || need_head_weights;

        let (tgt_len, bsz, embed_dim) = query.dims3()?;
        assert_eq!(embed_dim, self.embed_dim as usize);

        if !self.rot_emb.is_some()
            && self.enable_torch_version
            && !self.onnx_trace
            && incremental_state.is_none()
            && !static_kv
            && !need_head_weights
        {
            return multihead_attention_forward(
                query,
                key.unwrap(),
                value.unwrap(),
                self.embed_dim,
                self.num_heads,
                &self.q_proj,
                &self.k_proj,
                &self.v_proj,
                self.bias_k.as_ref(),
                self.bias_v.as_ref(),
                self.add_zero_attn,
                self.dropout,
                &self.out_proj,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
            );
        }

        let mut saved_state = None;
        if let Some(inc_state) = incremental_state {
            saved_state = self.get_incremental_state(Some(inc_state), "attn_state");
            if let Some(saved_state_ref) = &saved_state {
                if saved_state_ref.contains_key("prev_key") && static_kv {
                    assert!(self.encoder_decoder_attention && !self.self_attention);
                }
            }
        }

        let q = if self.self_attention {
            self.q_proj.forward(query)?
        } else if self.encoder_decoder_attention {
            self.q_proj.forward(query)?
        } else {
            assert!(key.is_some() && value.is_some());
            self.q_proj.forward(query)?
        };

        let k = if self.self_attention {
            self.k_proj.forward(query)?
        } else if self.encoder_decoder_attention && key.is_none() {
            assert!(value.is_none());
            Tensor::zeros((), query.device())?
        } else {
            assert!(key.is_some());
            self.k_proj.forward(key.unwrap())?
        };

        let v = if self.self_attention {
            self.v_proj.forward(query)?
        } else if self.encoder_decoder_attention && value.is_none() {
            Tensor::zeros((), query.device())?
        } else {
            assert!(value.is_some());
            self.v_proj.forward(value.unwrap())?
        };

        let q = (q * self.scaling)?;

        let (mut k, mut v) = if self.bias_k.is_some() {
            assert!(self.bias_v.is_some());
            let bias_k = self.bias_k.as_ref().unwrap();
            let bias_v = self.bias_v.as_ref().unwrap();

            let bias_k = bias_k.broadcast_as((1, 1, bias_k.dim(2)?))?;
            let bias_v = bias_v.broadcast_as((1, 1, bias_v.dim(2)?))?;

            let k = Tensor::cat(&[&k, &bias_k], 1)?;
            let v = Tensor::cat(&[&v, &bias_v], 1)?;

            if let Some(attn_mask) = attn_mask {
                let attn_mask = Tensor::cat(
                    &[
                        attn_mask,
                        Tensor::zeros((attn_mask.dim(0)?, 1), query.device())?,
                    ],
                    1,
                )?;
            }

            if let Some(key_padding_mask) = key_padding_mask {
                let key_padding_mask = Tensor::cat(
                    &[
                        key_padding_mask,
                        Tensor::zeros((key_padding_mask.dim(0)?, 1), query.device())?,
                    ],
                    1,
                )?;
            }

            (k, v)
        } else {
            (k, v)
        };

        let q = q
            .reshape((
                tgt_len,
                bsz * self.num_heads as usize,
                self.head_dim as usize,
            ))?
            .transpose(0, 1)?;

        if k.dim()? > 0 {
            k = k
                .reshape((-1, bsz * self.num_heads as usize, self.head_dim as usize))?
                .transpose(0, 1)?;
        }

        if v.dim()? > 0 {
            v = v
                .reshape((-1, bsz * self.num_heads as usize, self.head_dim as usize))?
                .transpose(0, 1)?;
        }

        if let Some(saved_state_ref) = &mut saved_state {
            if saved_state_ref.contains_key("prev_key") {
                let prev_key = saved_state_ref.get("prev_key").unwrap().as_ref().unwrap();
                let prev_key = prev_key.reshape((
                    bsz * self.num_heads as usize,
                    -1,
                    self.head_dim as usize,
                ))?;
                if static_kv {
                    k = prev_key;
                } else {
                    k = Tensor::cat(&[prev_key, &k], 1)?;
                }
            }

            if saved_state_ref.contains_key("prev_value") {
                let prev_value = saved_state_ref.get("prev_value").unwrap().as_ref().unwrap();
                let prev_value = prev_value.reshape((
                    bsz * self.num_heads as usize,
                    -1,
                    self.head_dim as usize,
                ))?;
                if static_kv {
                    v = prev_value;
                } else {
                    v = Tensor::cat(&[prev_value, &v], 1)?;
                }
            }

            saved_state_ref.insert("prev_key".to_string(), Some(k.clone()));
            saved_state_ref.insert("prev_value".to_string(), Some(v.clone()));

            if let Some(inc_state) = incremental_state {
                self.set_incremental_state(Some(inc_state), "attn_state", saved_state_ref.clone());
            }
        }

        let src_len = k.dim(1)?;

        if let Some(key_padding_mask) = key_padding_mask {
            assert_eq!(key_padding_mask.dim(0)? as i64, bsz);
            assert_eq!(key_padding_mask.dim(1)? as i64, src_len);
        }

        if self.add_zero_attn {
            src_len += 1;
            k = Tensor::cat(
                &[
                    &k,
                    Tensor::zeros((k.dim(0)?, 1, k.dim(2)?), query.device())?,
                ],
                1,
            )?;
            v = Tensor::cat(
                &[
                    &v,
                    Tensor::zeros((v.dim(0)?, 1, v.dim(2)?), query.device())?,
                ],
                1,
            )?;

            if let Some(attn_mask) = attn_mask {
                let attn_mask = Tensor::cat(
                    &[
                        attn_mask,
                        Tensor::zeros((attn_mask.dim(0)?, 1), query.device())?,
                    ],
                    1,
                )?;
            }

            if let Some(key_padding_mask) = key_padding_mask {
                let key_padding_mask = Tensor::cat(
                    &[
                        key_padding_mask,
                        Tensor::zeros((key_padding_mask.dim(0)?, 1), query.device())?,
                    ],
                    1,
                )?;
            }
        }

        if let Some(rot_emb) = &self.rot_emb {
            let (q, k) = rot_emb.forward(&q, &k)?;
        }

        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;

        assert_eq!(
            attn_weights.dims(),
            &[bsz * self.num_heads as usize, tgt_len, src_len]
        );

        if let Some(attn_mask) = attn_mask {
            let attn_mask = attn_mask.unsqueeze(0)?;
            if self.onnx_trace {
                let attn_mask = attn_mask.repeat((attn_weights.dim(0)?, 1, 1))?;
            }
            attn_weights = (&attn_weights + attn_mask)?;
        }

        if let Some(key_padding_mask) = key_padding_mask {
            let attn_weights =
                attn_weights.reshape((bsz, self.num_heads as usize, tgt_len, src_len))?;

            let key_padding_mask = key_padding_mask
                .unsqueeze(1)?
                .unsqueeze(2)?
                .to_dtype(candle_core::DType::Bool)?;

            let attn_weights = attn_weights
                .masked_fill(&key_padding_mask, f32::NEG_INFINITY)?
                .reshape((bsz * self.num_heads as usize, tgt_len, src_len))?;
        }

        if before_softmax {
            return Ok((attn_weights, Some(v)));
        }

        let attn_weights = if self.onnx_trace {
            attn_weights.to_dtype(candle_core::DType::F32)?.softmax(2)?
        } else {
            attn_weights.softmax(2)?
        };

        let attn_weights = ops::dropout(&attn_weights, self.dropout, self.training)?;

        let attn = attn_weights.matmul(&v)?;

        assert_eq!(
            attn.dims(),
            &[
                bsz * self.num_heads as usize,
                tgt_len,
                self.head_dim as usize
            ]
        );

        let attn = if self.onnx_trace && attn.dim(1)? == 1 {
            attn.reshape((tgt_len, bsz, self.embed_dim as usize))?
        } else {
            attn.transpose(0, 1)?
                .reshape((tgt_len, bsz, self.embed_dim as usize))?
        };

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
