use crate::esm3::utils::affine3d::Affine3D;
use crate::esmc::models::esmc::ESMCConfig;
use candle_core::{D, Module, Result, Tensor};
use candle_nn::{self as nn, LayerNorm, LayerNormConfig, Linear, VarBuilder};

const SQRT_3: f64 = 1.7320508075688772;

#[allow(dead_code)]
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
    pub fn load(vb: VarBuilder, config: &ESMCConfig) -> Result<Self> {
        let ESMCConfig {
            d_model,
            v_head_transformer,
            mask_and_zero_frameless,
            ..
        } = config;

        let num_vector_messages = 1usize;

        // todo: this is a hidden param. Needs to be fixed
        let v_heads = v_head_transformer.unwrap_or(128);

        let dim_proj = 4 * v_heads * 3 + v_heads * 3 * num_vector_messages;
        let channels_out = v_heads * 3 * num_vector_messages;

        let ln_conf = LayerNormConfig::from(1e-5);
        let s_norm = nn::layer_norm(*d_model, ln_conf, vb.pp("layer_norm"))?;

        let proj = nn::linear(*d_model, dim_proj, vb.pp("linear1"))?;
        let out_proj = nn::linear(channels_out, *d_model, vb.pp("outproj"))?;
        let distance_scale_per_head = Tensor::zeros((v_heads,), vb.dtype(), vb.device())?;
        let rotation_scale_per_head = Tensor::zeros((v_heads,), vb.dtype(), vb.device())?;

        Ok(Self {
            c_s: *d_model,
            v_heads,
            num_vector_messages,
            mask_and_zero_frameless: *mask_and_zero_frameless,
            s_norm,
            proj,
            out_proj,
            distance_scale_per_head,
            rotation_scale_per_head,
        })
    }

    /// Geometric attention forward pass.
    ///
    /// - `s`:           `(B, L, d_model)` hidden states.
    /// - `affine`:      per-residue local frames `(B, L, 3, 3)` rot + `(B, L, 3)` trans.
    /// - `affine_mask`: `(B, L)` u8 — 1 where the frame is valid, 0 for frameless positions.
    /// - `sequence_id`: optional `(B, L)` int — positions with the same ID form one protein.
    /// - `chain_id`:    optional `(B, L)` int — positions in the same chain.
    ///
    /// Returns `(B, L, d_model)`.
    pub fn forward(
        &self,
        s: &Tensor,
        affine: &Affine3D,
        affine_mask: &Tensor,
        sequence_id: Option<&Tensor>,
        chain_id: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, l, _) = s.dims3()?;
        let dtype = s.dtype();
        let device = s.device();

        // ── Attention bias (sequence_id and chain_id masking) ──────────────
        // Same-sequence pairs get 1.0; cross-sequence and frameless get -inf.
        let attn_bias = if let Some(seq_id) = sequence_id {
            let seq_q = seq_id.unsqueeze(D::Minus1)?; // (B, L, 1)
            let seq_k = seq_id.unsqueeze(D::Minus2)?; // (B, 1, L)
            // (B, L, L): 1 where same sequence, 0 where different
            let same_seq = seq_q
                .broadcast_as((b, l, l))?
                .eq(&seq_k.broadcast_as((b, l, l))?)?
                .to_dtype(dtype)?;
            same_seq.unsqueeze(1)? // (B, 1, L, L)
        } else {
            Tensor::ones((b, 1, l, l), dtype, device)?
        };

        // Mask frameless key positions with -inf
        let neg_inf = Tensor::full(f32::NEG_INFINITY as f64, attn_bias.shape(), device)?
            .to_dtype(dtype)?;
        // affine_mask: (B, L) → (B, 1, 1, L)
        let frame_mask_k = affine_mask
            .unsqueeze(1)?
            .unsqueeze(1)?
            .broadcast_as(attn_bias.shape())?;
        let mut attn_bias = frame_mask_k.where_cond(&attn_bias, &neg_inf)?;

        // Mask cross-chain pairs with -inf
        if let Some(cid) = chain_id {
            let chain_q = cid.unsqueeze(D::Minus1)?.broadcast_as((b, l, l))?;
            let chain_k = cid.unsqueeze(D::Minus2)?.broadcast_as((b, l, l))?;
            let diff_chain = chain_q.ne(&chain_k)?.unsqueeze(1)?; // (B, 1, L, L)
            let diff_bias = diff_chain.broadcast_as(attn_bias.shape())?;
            attn_bias = diff_bias.where_cond(&neg_inf, &attn_bias)?;
        }

        // ── Project hidden states ──────────────────────────────────────────
        let ns = self.s_norm.forward(s)?;
        let proj_out = self.proj.forward(&ns)?;

        let vec_rot_size = self.v_heads * 2 * 3 + self.v_heads * 3 * self.num_vector_messages;
        let vec_dist_size = self.v_heads * 2 * 3;
        let vec_rot = proj_out.narrow(D::Minus1, 0, vec_rot_size)?;
        let vec_dist = proj_out.narrow(D::Minus1, vec_rot_size, vec_dist_size)?;

        // ── Rotation-only vectors: Q_rot, K_rot, V ────────────────────────
        // Reshape: (B, L, (h*c)) → (B, L, h, 3)
        let h_rot = 2 * self.v_heads + self.v_heads * self.num_vector_messages;
        let vec_rot = vec_rot.reshape((b, l, h_rot, 3))?;
        // Rotate local-frame vectors to global frame: (B, L, h_rot, 3)
        let vec_rot = Affine3D::apply_rot(&affine.rot, &vec_rot)?;

        let query_rot = vec_rot.narrow(D::Minus2, 0, self.v_heads)?; // (B, L, H, 3)
        let key_rot = vec_rot.narrow(D::Minus2, self.v_heads, self.v_heads)?;
        let value = vec_rot.narrow(D::Minus2, 2 * self.v_heads, self.v_heads * self.num_vector_messages)?;

        // ── Full-affine (rot+trans) vectors: Q_dist, K_dist ───────────────
        let vec_dist = vec_dist.reshape((b, l, self.v_heads * 2, 3))?;
        let vec_dist = affine.apply(&vec_dist)?; // (B, L, H*2, 3) in global frame
        let query_dist = vec_dist.narrow(D::Minus2, 0, self.v_heads)?; // (B, L, H, 3)
        let key_dist = vec_dist.narrow(D::Minus2, self.v_heads, self.v_heads)?;

        // ── Rearrange for attention computation ───────────────────────────
        // (B, L, H, 3) → (B, H, L, 3)
        let query_rot = query_rot.permute((0, 2, 1, 3))?;
        // (B, L, H, 3) → (B, H, 3, L)  [for matmul with query]
        let key_rot = key_rot.permute((0, 2, 3, 1))?;
        // (B, L, H, 3) → (B, H, L, 1, 3)
        let query_dist = query_dist.permute((0, 2, 1, 3))?.unsqueeze(D::Minus2)?;
        // (B, L, H, 3) → (B, H, 1, L, 3)  [unsqueeze at position 2 = -3 of 5-dim result]
        let key_dist = key_dist.permute((0, 2, 1, 3))?.unsqueeze(2)?;
        // (B, L, H*num_vm, 3) → (B, H, L, num_vm*3)
        let value = value
            .reshape((b, l, self.v_heads, self.num_vector_messages * 3))?
            .permute((0, 2, 1, 3))?;

        // ── Attention weight: rotation + distance terms ────────────────────
        // Rotation term: (B, H, L, 3) @ (B, H, 3, L) = (B, H, L, L)
        // affine(scale, 0.0) = tensor * scale (scalar multiplication)
        let rotation_term = query_rot
            .contiguous()?
            .matmul(&key_rot.contiguous()?)?
            .affine(SQRT_3.recip(), 0.0)?;

        // Distance term: ||q - k||_2 / sqrt(3) → (B, H, L, L)
        let diff = query_dist
            .broadcast_as((b, self.v_heads, l, l, 3))?
            .sub(&key_dist.broadcast_as((b, self.v_heads, l, l, 3))?)?;
        let distance_term = diff
            .sqr()?
            .sum(D::Minus1)?
            .sqrt()?
            .affine(SQRT_3.recip(), 0.0)?;

        // Learnable per-head weights: (H,) → (1, H, 1, 1); softplus = log(1 + exp(x))
        let dist_w = softplus(&self.distance_scale_per_head)?
            .reshape((1, self.v_heads, 1, 1))?
            .broadcast_as((b, self.v_heads, l, l))?;
        let rot_w = softplus(&self.rotation_scale_per_head)?
            .reshape((1, self.v_heads, 1, 1))?
            .broadcast_as((b, self.v_heads, l, l))?;

        let mut attn_weight = rotation_term
            .mul(&rot_w)?
            .sub(&distance_term.mul(&dist_w)?)?;

        // Add attention bias (already (B, 1, L, L); broadcast to (B, H, L, L))
        let attn_bias = attn_bias.broadcast_as(attn_weight.shape())?;
        attn_weight = attn_weight.add(&attn_bias)?;

        let attn_weight = candle_nn::ops::softmax(&attn_weight, D::Minus1)?;

        // ── Weighted sum of values ────────────────────────────────────────
        // (B, H, L, L) @ (B, H, L, num_vm*3) = (B, H, L, num_vm*3)
        let attn_out = attn_weight.matmul(&value.contiguous()?)?;

        // Rearrange: (B, H, L, num_vm*3) → (B, L, H*num_vm, 3)
        let attn_out = attn_out
            .permute((0, 2, 1, 3))? // (B, L, H, num_vm*3)
            .contiguous()?
            .reshape((b, l, self.v_heads * self.num_vector_messages, 3))?;

        // Rotate back from global frame to local frame
        let attn_out = Affine3D::apply_rot_inv(&affine.rot, &attn_out)?;

        // Flatten head and vector-message dims: (B, L, H*num_vm, 3) → (B, L, H*num_vm*3)
        let mut attn_out = attn_out
            .contiguous()?
            .reshape((b, l, self.v_heads * self.num_vector_messages * 3))?;

        // Zero out frameless positions if requested
        if self.mask_and_zero_frameless {
            let zeros = Tensor::zeros_like(&attn_out)?;
            let mask_exp = affine_mask.unsqueeze(D::Minus1)?.broadcast_as(attn_out.shape())?;
            attn_out = mask_exp.where_cond(&attn_out, &zeros)?;
        }

        self.out_proj.forward(&attn_out)
    }
}

/// Numerically stable softplus: `log(1 + exp(x))`.
fn softplus(t: &Tensor) -> Result<Tensor> {
    // affine(1.0, 1.0) computes tensor * 1 + 1 = tensor + 1
    t.exp()?.affine(1.0, 1.0)?.log()
}
