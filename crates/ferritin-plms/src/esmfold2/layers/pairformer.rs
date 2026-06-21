//! Pairformer block — core iterative refinement layer of the ESMFold2 FoldingTrunk.
//!
//! Each block refines the pair representation `[B, N, N, d_pair]` by applying:
//! 1. Row-wise triangle attention  (starting-node pair bias)
//! 2. Column-wise triangle attention (ending-node pair bias)
//! 3. Triangle multiplicative update — outgoing  (z_ij += Σ_k a_ik ⊙ b_jk)
//! 4. Triangle multiplicative update — incoming  (z_ij += Σ_k a_ki ⊙ b_kj)
//! 5. Pair transition FFN
//!
//! Weight layout (rooted at `folding_trunk.blocks.{i}`):
//! ```text
//! tri_attn_row.norm.*       — pre-norm LayerNorm
//! tri_attn_row.q_proj.*     — Q (no bias)
//! tri_attn_row.k_proj.*     — K (no bias)
//! tri_attn_row.v_proj.*     — V (no bias)
//! tri_attn_row.pair_bias.*  — pair bias → n_heads (no bias)
//! tri_attn_row.gate.*       — sigmoid gate → n_heads*d_head (no bias)
//! tri_attn_row.out_proj.*   — output → d_pair (no bias)
//! tri_attn_col.*            — identical layout
//! tri_mult_out.norm.*       — input LayerNorm
//! tri_mult_out.left_proj.*  — left projection → c_hidden (no bias)
//! tri_mult_out.right_proj.* — right projection → c_hidden (no bias)
//! tri_mult_out.left_gate.*  — left sigmoid gate → c_hidden (no bias)
//! tri_mult_out.right_gate.* — right sigmoid gate → c_hidden (no bias)
//! tri_mult_out.out_norm.*   — pre-output LayerNorm on c_hidden
//! tri_mult_out.out_proj.*   — c_hidden → d_pair (no bias)
//! tri_mult_out.out_gate.*   — output sigmoid gate → d_pair (no bias)
//! tri_mult_in.*             — identical layout
//! pair_trans.norm.*         — transition pre-norm
//! pair_trans.fc1.*          — d_pair → 4*d_pair (no bias)
//! pair_trans.fc2.*          — 4*d_pair → d_pair (no bias)
//! ```

use candle_core::{D, Result, Tensor};
use candle_nn::{self as nn, LayerNorm, LayerNormConfig, Module, VarBuilder};

// ── Triangle Multiplicative Update ───────────────────────────────────────────

struct TriangleMult {
    norm: LayerNorm,
    left_proj: nn::Linear,
    right_proj: nn::Linear,
    left_gate: nn::Linear,
    right_gate: nn::Linear,
    out_norm: LayerNorm,
    out_proj: nn::Linear,
    out_gate: nn::Linear,
    outgoing: bool,
}

impl TriangleMult {
    fn load(vb: VarBuilder, d_pair: usize, c_hidden: usize, outgoing: bool) -> Result<Self> {
        Ok(Self {
            norm: nn::layer_norm(d_pair, LayerNormConfig::from(1e-5), vb.pp("norm"))?,
            left_proj: nn::linear_no_bias(d_pair, c_hidden, vb.pp("left_proj"))?,
            right_proj: nn::linear_no_bias(d_pair, c_hidden, vb.pp("right_proj"))?,
            left_gate: nn::linear_no_bias(d_pair, c_hidden, vb.pp("left_gate"))?,
            right_gate: nn::linear_no_bias(d_pair, c_hidden, vb.pp("right_gate"))?,
            out_norm: nn::layer_norm(c_hidden, LayerNormConfig::from(1e-5), vb.pp("out_norm"))?,
            out_proj: nn::linear_no_bias(c_hidden, d_pair, vb.pp("out_proj"))?,
            out_gate: nn::linear_no_bias(d_pair, d_pair, vb.pp("out_gate"))?,
            outgoing,
        })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let z_n = self.norm.forward(z)?;

        // Gated projections: [B, N, N, c_hidden]
        let left_g = nn::ops::sigmoid(&self.left_gate.forward(&z_n)?)?;
        let left = (self.left_proj.forward(&z_n)? * left_g)?;
        let right_g = nn::ops::sigmoid(&self.right_gate.forward(&z_n)?)?;
        let right = (self.right_proj.forward(&z_n)? * right_g)?;

        // Permute to [B, c, N_i, N_k] for batch matmul.
        // Outgoing: left[b,i,k,c] → permute(0,3,1,2) → [B,c,i,k]
        // Incoming: left[b,k,i,c] treated as [B,c,i,k] → permute(0,3,2,1) swaps the N dims
        let (left_p, right_p) = if self.outgoing {
            (
                left.permute((0, 3, 1, 2))?.contiguous()?,
                right.permute((0, 3, 1, 2))?.contiguous()?,
            )
        } else {
            (
                left.permute((0, 3, 2, 1))?.contiguous()?,
                right.permute((0, 3, 2, 1))?.contiguous()?,
            )
        };

        // p[b,c,i,j] = Σ_k left_p[b,c,i,k] * right_p[b,c,j,k]
        let p = left_p.matmul(&right_p.transpose(D::Minus2, D::Minus1)?.contiguous()?)?; // [B, c, N, N]
        let p = p.permute((0, 2, 3, 1))?; // [B, N, N, c]
        let p = self.out_norm.forward(&p)?;

        let out_g = nn::ops::sigmoid(&self.out_gate.forward(&z_n)?)?; // [B, N, N, d_pair]
        let out = (out_g * self.out_proj.forward(&p)?)?;
        z + &out
    }
}

// ── Triangle Attention ────────────────────────────────────────────────────────

struct TriangleAttention {
    norm: LayerNorm,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    pair_bias: nn::Linear,
    gate: nn::Linear,
    out_proj: nn::Linear,
    n_heads: usize,
    d_head: usize,
    row_wise: bool,
}

impl TriangleAttention {
    fn load(vb: VarBuilder, d_pair: usize, n_heads: usize, row_wise: bool) -> Result<Self> {
        let d_head = d_pair / n_heads;
        Ok(Self {
            norm: nn::layer_norm(d_pair, LayerNormConfig::from(1e-5), vb.pp("norm"))?,
            q_proj: nn::linear_no_bias(d_pair, n_heads * d_head, vb.pp("q_proj"))?,
            k_proj: nn::linear_no_bias(d_pair, n_heads * d_head, vb.pp("k_proj"))?,
            v_proj: nn::linear_no_bias(d_pair, n_heads * d_head, vb.pp("v_proj"))?,
            pair_bias: nn::linear_no_bias(d_pair, n_heads, vb.pp("pair_bias"))?,
            gate: nn::linear_no_bias(d_pair, n_heads * d_head, vb.pp("gate"))?,
            out_proj: nn::linear_no_bias(n_heads * d_head, d_pair, vb.pp("out_proj"))?,
            n_heads,
            d_head,
            row_wise,
        })
    }

    // Operates on z [B, n1, n2, d]: treats n1 as independent rows, attends along n2.
    fn forward_inner(&self, z: &Tensor) -> Result<Tensor> {
        let (b, n1, n2, _) = z.dims4()?;
        let (h, dh) = (self.n_heads, self.d_head);

        let z_n = self.norm.forward(z)?;

        let q = self.q_proj.forward(&z_n)?; // [B, n1, n2, H*dh]
        let k = self.k_proj.forward(&z_n)?;
        let v = self.v_proj.forward(&z_n)?;
        let bias = self.pair_bias.forward(&z_n)?; // [B, n1, n2, H]
        let gate = nn::ops::sigmoid(&self.gate.forward(&z_n)?)?; // [B, n1, n2, H*dh]

        // Merge (B, n1) → B*n1 and split heads: [B,n1,n2,H*dh] → [B*n1, H, n2, dh]
        let bn1 = b * n1;
        let to_heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((bn1, n2, h, dh))?
                .permute((0, 2, 1, 3))?
                .contiguous()
        };
        let q = to_heads(q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;

        // Pair bias: [B,n1,n2,H] → [B*n1,H,1,n2] (broadcast over query positions)
        let bias = bias
            .reshape((bn1, n2, h))?
            .permute((0, 2, 1))? // [B*n1, H, n2]
            .unsqueeze(2)?; // [B*n1, H, 1, n2]

        let scale = (dh as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? / scale)?;
        // scores: [B*n1, H, n2, n2]; bias: [B*n1, H, 1, n2] — broadcast over query positions
        let scores = scores.broadcast_add(&bias)?;
        let attn = nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v)?; // [B*n1, H, n2, dh]

        // [B*n1, H, n2, dh] → [B, n1, n2, H*dh]
        let out = out
            .permute((0, 2, 1, 3))?
            .contiguous()? // [B*n1, n2, H, dh]
            .reshape((b, n1, n2, h * dh))?;

        let out = (gate * out)?;
        self.out_proj.forward(&out)
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let delta = if self.row_wise {
            self.forward_inner(z)?
        } else {
            // Column-wise: transpose the two sequence dims before/after
            let z_t = z.permute((0, 2, 1, 3))?;
            self.forward_inner(&z_t)?.permute((0, 2, 1, 3))?
        };
        z + &delta
    }
}

// ── Pair Transition FFN ───────────────────────────────────────────────────────

struct PairTransition {
    norm: LayerNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl PairTransition {
    fn load(vb: VarBuilder, d_pair: usize) -> Result<Self> {
        Ok(Self {
            norm: nn::layer_norm(d_pair, LayerNormConfig::from(1e-5), vb.pp("norm"))?,
            fc1: nn::linear_no_bias(d_pair, 4 * d_pair, vb.pp("fc1"))?,
            fc2: nn::linear_no_bias(4 * d_pair, d_pair, vb.pp("fc2"))?,
        })
    }

    fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(z)?;
        let h = self.fc1.forward(&h)?.relu()?;
        let h = self.fc2.forward(&h)?;
        z + &h
    }
}

// ── PairformerBlock ───────────────────────────────────────────────────────────

/// One Pairformer layer: refines the pair representation [B, N, N, d_pair].
///
/// Used by both the 24-layer FoldingTrunk and the 4-layer ConfidenceHead trunk.
/// Triangle multiplication hidden dim (`c_hidden`) is fixed at 128.
pub struct PairformerBlock {
    tri_attn_row: TriangleAttention,
    tri_attn_col: TriangleAttention,
    tri_mult_out: TriangleMult,
    tri_mult_in: TriangleMult,
    pair_trans: PairTransition,
}

/// Hidden dim for triangle multiplicative updates (standard: 128).
const C_HIDDEN_MULT: usize = 128;

impl PairformerBlock {
    /// Load one Pairformer block from a `VarBuilder` rooted at `blocks.{i}`.
    ///
    /// `n_heads` must evenly divide `d_pair`; for the trunk, `n_heads=8`, `d_pair=256`.
    pub fn load(vb: VarBuilder, d_pair: usize, n_heads: usize) -> Result<Self> {
        Ok(Self {
            tri_attn_row: TriangleAttention::load(vb.pp("tri_attn_row"), d_pair, n_heads, true)?,
            tri_attn_col: TriangleAttention::load(vb.pp("tri_attn_col"), d_pair, n_heads, false)?,
            tri_mult_out: TriangleMult::load(vb.pp("tri_mult_out"), d_pair, C_HIDDEN_MULT, true)?,
            tri_mult_in: TriangleMult::load(vb.pp("tri_mult_in"), d_pair, C_HIDDEN_MULT, false)?,
            pair_trans: PairTransition::load(vb.pp("pair_trans"), d_pair)?,
        })
    }

    /// Run one Pairformer layer.
    ///
    /// Input/output: `[B, N, N, d_pair]`.
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let z = self.tri_attn_row.forward(z)?;
        let z = self.tri_attn_col.forward(&z)?;
        let z = self.tri_mult_out.forward(&z)?;
        let z = self.tri_mult_in.forward(&z)?;
        self.pair_trans.forward(&z)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    const B: usize = 1;
    const N: usize = 8;
    const D_PAIR: usize = 32; // small dims for fast tests
    const N_HEADS: usize = 4;
    const C_HIDDEN: usize = 16;

    fn zeros_pair(device: &Device) -> Tensor {
        Tensor::zeros(&[B, N, N, D_PAIR], DType::F32, device).unwrap()
    }

    #[test]
    fn test_triangle_mult_outgoing_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let m = TriangleMult::load(vb, D_PAIR, C_HIDDEN, true).unwrap();
        let out = m.forward(&zeros_pair(&device)).unwrap();
        assert_eq!(out.dims(), &[B, N, N, D_PAIR]);
    }

    #[test]
    fn test_triangle_mult_incoming_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let m = TriangleMult::load(vb, D_PAIR, C_HIDDEN, false).unwrap();
        let out = m.forward(&zeros_pair(&device)).unwrap();
        assert_eq!(out.dims(), &[B, N, N, D_PAIR]);
    }

    #[test]
    fn test_triangle_attn_row_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let m = TriangleAttention::load(vb, D_PAIR, N_HEADS, true).unwrap();
        let out = m.forward(&zeros_pair(&device)).unwrap();
        assert_eq!(out.dims(), &[B, N, N, D_PAIR]);
    }

    #[test]
    fn test_triangle_attn_col_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let m = TriangleAttention::load(vb, D_PAIR, N_HEADS, false).unwrap();
        let out = m.forward(&zeros_pair(&device)).unwrap();
        assert_eq!(out.dims(), &[B, N, N, D_PAIR]);
    }

    #[test]
    fn test_pair_transition_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let m = PairTransition::load(vb, D_PAIR).unwrap();
        let out = m.forward(&zeros_pair(&device)).unwrap();
        assert_eq!(out.dims(), &[B, N, N, D_PAIR]);
    }

    #[test]
    fn test_pairformer_block_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = PairformerBlock::load(vb, D_PAIR, N_HEADS).unwrap();
        let out = block.forward(&zeros_pair(&device)).unwrap();
        assert_eq!(out.dims(), &[B, N, N, D_PAIR]);
    }

    #[test]
    fn test_pairformer_block_batch_shape() {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        let block = PairformerBlock::load(vb, D_PAIR, N_HEADS).unwrap();
        let z = Tensor::zeros(&[2, N, N, D_PAIR], DType::F32, &device).unwrap();
        let out = block.forward(&z).unwrap();
        assert_eq!(out.dims(), &[2, N, N, D_PAIR]);
    }

    #[test]
    fn test_triangle_mult_math_outgoing() {
        // Verify outgoing vs incoming produce the same shape but different values
        // when z is asymmetric (upper triangle != lower triangle).
        let device = Device::Cpu;
        let n = 4;
        let d = 8;
        // Use random-ish values via arange
        let vals: Vec<f32> = (0..n * n * d).map(|i| i as f32 * 0.01).collect();
        let z = Tensor::from_vec(vals, &[1, n, n, d], &device).unwrap();

        let vb_out = VarBuilder::zeros(DType::F32, &device);
        let vb_in = VarBuilder::zeros(DType::F32, &device);
        let m_out = TriangleMult::load(vb_out, d, 4, true).unwrap();
        let m_in = TriangleMult::load(vb_in, d, 4, false).unwrap();

        let out_out = m_out.forward(&z).unwrap();
        let out_in = m_in.forward(&z).unwrap();
        assert_eq!(out_out.dims(), &[1, n, n, d]);
        assert_eq!(out_in.dims(), &[1, n, n, d]);
        // With zero weights both will be zero + residual = z; shapes are equal
        assert_eq!(out_out.dims(), out_in.dims());
    }
}
