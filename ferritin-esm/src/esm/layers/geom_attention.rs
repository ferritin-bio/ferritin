use crate::esm::{models::esmc::ESMCConfig, utils::constants::SQRT_3};
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{self as nn, layer_norm, LayerNorm, LayerNormConfig, Linear, VarBuilder};

pub struct GeometricReasoningOriginalImpl {
    c_s: usize,
    v_heads: usize,
    num_vector_messages: usize,
    mask_and_zero_frameless: bool,
    s_norm: LayerNorm,
    proj: Linear,
    out_proj: Linear,
    distance_scale_per_head: Tensor,
    rotation_scale_per_head: Tensor,
}

impl GeometricReasoningOriginalImpl {
    // pub fn new(
    //     c_s: i64,
    //     v_heads: i64,
    //     num_vector_messages: i64,
    //     mask_and_zero_frameless: bool,
    //     _divide_residual_by_depth: bool,
    //     bias: bool,
    //     device: &Device,
    // ) -> Result<Self> {
    //     let dim_proj = 4 * v_heads * 3 + v_heads * 3 * num_vector_messages;
    //     let channels_out = v_heads * 3 * num_vector_messages;

    //     Ok(Self {
    //         c_s,
    //         v_heads,
    //         num_vector_messages,
    //         mask_and_zero_frameless,
    //         s_norm: LayerNorm::new(c_s, bias)?,
    //         proj: Linear::new(c_s, dim_proj, bias)?,
    //         out_proj: Linear::new(channels_out, c_s, bias)?,
    //         distance_scale_per_head: Tensor::zeros((v_heads,), device)?,
    //         rotation_scale_per_head: Tensor::zeros((v_heads,), device)?,
    //     })
    // }
    pub fn load(vb: VarBuilder, config: ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            v_head_transformer,
            // num_vector_messages,
            mask_and_zero_frameless,
        } = config;

        let num_vector_messages = 1usize;
        let v_heads = v_head_transformer.unwrap();

        let dim_proj = 4 * v_heads * 3 + v_heads * 3 * num_vector_messages;
        let channels_out = v_heads * 3 * num_vector_messages;

        let ln_conf = LayerNormConfig::from(1e-5);
        let s_norm = nn::layer_norm(d_model, ln_conf, vb.pp("layer_norm"))?;

        let proj = nn::linear(d_model, dim_proj, vb.pp("linear1"))?;
        let out_proj = nn::linear(channels_out, d_model, vb.pp("outproj"))?;
        let distance_scale_per_head = Tensor::zeros((v_heads,), vb.dtype(), vb.device())?;
        let rotation_scale_per_head = Tensor::zeros((v_heads,), vb.dtype(), vb.device())?;

        Ok(Self {
            c_s: d_model as usize,
            v_heads: v_heads as usize,
            num_vector_messages,
            mask_and_zero_frameless,
            s_norm,
            proj,
            out_proj,
            distance_scale_per_head,
            rotation_scale_per_head,
        })
    }

    // pub fn forward(
    //     &self,
    //     s: &Tensor,
    //     affine: &Affine,
    //     affine_mask: &Tensor,
    //     sequence_id: Option<&Tensor>,
    //     chain_id: &Tensor,
    // ) -> Result<Tensor> {
    //     let sequence_id = match sequence_id {
    //         Some(sid) => sid.clone(),
    //         None => Tensor::zeros_like(&s.slice(s.dims()? - 1, 0, 1)?)?,
    //     };

    //     let attn_bias = sequence_id.unsqueeze(-1)?.eq(&sequence_id.unsqueeze(-2)?)?;
    //     let attn_bias = attn_bias.unsqueeze(1)?.to_dtype(s.dtype())?;
    //     let attn_bias = attn_bias.masked_fill(
    //         &affine_mask.broadcast_left(3)?.logical_not()?,
    //         f32::NEG_INFINITY,
    //     )?;

    //     let chain_id_mask = chain_id.unsqueeze(1)?.ne(&chain_id.unsqueeze(2)?)?;
    //     let attn_bias = attn_bias.masked_fill(&chain_id_mask.unsqueeze(1)?, f32::NEG_INFINITY)?;

    //     let ns = self.s_norm.forward(s)?;
    //     let proj_out = self.proj.forward(&ns)?;

    //     let (vec_rot, vec_dist) = proj_out.split_at(
    //         -1,
    //         &[
    //             self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages,
    //             self.v_heads * 2 * 3,
    //         ],
    //     )?;

    //     let vec_rot = rearrange(&vec_rot, "... (h c) -> ... h c", &[("c", 3)])?;
    //     let rot_out = affine.rot.broadcast_right(1)?.apply(&vec_rot)?;
    //     let (query_rot, key_rot, value) = rot_out.split_at(
    //         -2,
    //         &[
    //             self.v_heads,
    //             self.v_heads,
    //             self.v_heads * self.num_vector_messages,
    //         ],
    //     )?;

    //     let vec_dist = rearrange(&vec_dist, "... (h c) -> ... h c", &[("c", 3)])?;
    //     let (query_dist, key_dist) = affine.broadcast_right(1)?.apply(&vec_dist)?.chunk(2, -2)?;

    //     let query_dist = rearrange(&query_dist, "b s h d -> b h s 1 d", &[])?;
    //     let key_dist = rearrange(&key_dist, "b s h d -> b h 1 s d", &[])?;
    //     let query_rot = rearrange(&query_rot, "b s h d -> b h s d", &[])?;
    //     let key_rot = rearrange(&key_rot, "b s h d -> b h d s", &[])?;
    //     let value = rearrange(
    //         &value,
    //         "b s (h m) d -> b h s (m d)",
    //         &[("m", self.num_vector_messages)],
    //     )?;

    //     let distance_term = query_dist.sub(&key_dist)?.norm_dim(-1, true)?.div(SQRT_3)?;
    //     let rotation_term = query_rot.matmul(&key_rot)?.div(SQRT_3)?;

    //     let distance_term_weight =
    //         rearrange(&self.distance_scale_per_head.softplus()?, "h -> h 1 1", &[])?;
    //     let rotation_term_weight =
    //         rearrange(&self.rotation_scale_per_head.softplus()?, "h -> h 1 1", &[])?;

    //     let mut attn_weight = rotation_term
    //         .mul(&rotation_term_weight)?
    //         .sub(&distance_term.mul(&distance_term_weight)?)?;

    //     if let Some(bias) = attn_bias {
    //         let s_q = attn_weight.size(2)?;
    //         let s_k = attn_weight.size(3)?;
    //         let _s_q = (bias.size(2)? - s_q).max(0);
    //         let _s_k = (bias.size(3)? - s_k).max(0);
    //         let bias = bias.slice(_s_q..bias.size(2)?, _s_k..bias.size(3)?)?;
    //         attn_weight = attn_weight.add(&bias)?;
    //     }

    //     let attn_weight = attn_weight.softmax(-1)?;
    //     let mut attn_out = attn_weight.matmul(&value)?;

    //     attn_out = affine.rot.broadcast_right(1)?.invert()?.apply(&rearrange(
    //         &attn_out,
    //         "b h s (m d) -> b s (h m) d",
    //         &[("m", self.num_vector_messages)],
    //     )?)?;

    //     let mut attn_out = rearrange(
    //         &attn_out,
    //         "b s (h m) d -> b s (h m d)",
    //         &[("m", self.num_vector_messages)],
    //     )?;

    //     if self.mask_and_zero_frameless {
    //         attn_out =
    //             attn_out.masked_fill(&affine_mask.broadcast_right(1)?.logical_not()?, 0.0)?;
    //     }

    //     self.out_proj.forward(&attn_out)
    // }
}
