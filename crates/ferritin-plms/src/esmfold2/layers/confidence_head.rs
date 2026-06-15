//! Confidence head: predicts pLDDT, pAE, pDE, and distogram.
//!
//! Architecture:
//! 1. 4-layer Pairformer trunk refines (single, pair)
//! 2. Four linear heads project to logit bins:
//!    - pLDDT:     single → 50 bins  → softmax → weighted mean → scalar [0,1]
//!    - pAE:       pair   → 64 bins  → softmax → weighted mean → matrix [Å]
//!    - pDE:       pair   → 64 bins  (predicted distance error)
//!    - distogram: pair   → 39 bins
//!
//! Weight layout (rooted at `confidence_head`):
//! ```text
//! trunk.blocks.{0..3}.*   — 4-layer Pairformer (8 heads, d_pair=256)
//! plddt_head.weight        — linear d_single → num_plddt_bins (no bias)
//! pae_head.weight          — linear d_pair   → num_pae_bins   (no bias)
//! pde_head.weight          — linear d_pair   → num_pde_bins   (no bias)
//! distogram_head.weight    — linear d_pair   → distogram_bins (no bias)
//! ```

use super::pairformer::PairformerBlock;
use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder, ops::softmax};

// ── Helpers ────────────────────────────────────────────────────────────────

/// Convert bin logits to a scalar via softmax + weighted average of bin centres.
///
/// `logits`    — `[..., n_bins]`
/// `min_val`   — value of the first bin centre
/// `max_val`   — value of the last bin centre
///
/// Returns `[...]` with the same leading dims, values in `[min_val, max_val]`.
pub fn bins_to_scalar(logits: &Tensor, min_val: f64, max_val: f64) -> Result<Tensor> {
    let n_bins = logits.dim(candle_core::D::Minus1)?;
    let probs = softmax(logits, candle_core::D::Minus1)?;

    // Build bin centres tensor on the same device
    let step = (max_val - min_val) / (n_bins - 1) as f64;
    let centres: Vec<f32> = (0..n_bins)
        .map(|i| (min_val + i as f64 * step) as f32)
        .collect();
    let centres = Tensor::from_vec(centres, n_bins, logits.device())?.to_dtype(logits.dtype())?;

    // Weighted sum over last dim
    (probs * centres.broadcast_as(logits.shape())?)?.sum(candle_core::D::Minus1)
}

/// Convert pLDDT logits `[B, N, 50]` → scalar pLDDT `[B, N]` in `[0, 1]`.
pub fn plddt_from_logits(logits: &Tensor) -> Result<Tensor> {
    bins_to_scalar(logits, 0.0, 1.0)
}

// ── ConfidenceHead ─────────────────────────────────────────────────────────

/// Output of the confidence head.
pub struct ConfidenceOutput {
    /// pLDDT logits `[B, N_tok, num_plddt_bins]`.
    pub plddt_logits: Tensor,
    /// pLDDT scalar per token `[B, N_tok]`, range [0, 1].
    pub plddt: Tensor,
    /// pAE logits `[B, N_tok, N_tok, num_pae_bins]`. `None` if not produced.
    pub pae_logits: Option<Tensor>,
    /// pDE logits `[B, N_tok, N_tok, num_pde_bins]`. `None` if not produced.
    pub pde_logits: Option<Tensor>,
    /// Distogram logits `[B, N_tok, N_tok, distogram_bins]`. `None` if not produced.
    pub distogram_logits: Option<Tensor>,
}

/// 4-layer Pairformer confidence head.
pub struct ConfidenceHead {
    trunk: Vec<PairformerBlock>,
    plddt_head: nn::Linear,
    pae_head: nn::Linear,
    pde_head: nn::Linear,
    distogram_head: nn::Linear,
    num_plddt_bins: usize,
}

/// Number of attention heads in the confidence head Pairformer trunk (same as folding trunk).
const CONFIDENCE_N_HEADS: usize = 8;

impl ConfidenceHead {
    /// Load the confidence head from a `VarBuilder` rooted at `confidence_head.*`.
    pub fn load(
        vb: VarBuilder,
        d_single: usize,
        d_pair: usize,
        num_plddt_bins: usize,
        num_pae_bins: usize,
        num_pde_bins: usize,
        distogram_bins: usize,
    ) -> Result<Self> {
        let n_trunk_layers = 4;
        let trunk = (0..n_trunk_layers)
            .map(|i| {
                PairformerBlock::load(
                    vb.pp(format!("trunk.blocks.{i}")),
                    d_pair,
                    CONFIDENCE_N_HEADS,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            trunk,
            plddt_head: nn::linear_no_bias(d_single, num_plddt_bins, vb.pp("plddt_head"))?,
            pae_head: nn::linear_no_bias(d_pair, num_pae_bins, vb.pp("pae_head"))?,
            pde_head: nn::linear_no_bias(d_pair, num_pde_bins, vb.pp("pde_head"))?,
            distogram_head: nn::linear_no_bias(d_pair, distogram_bins, vb.pp("distogram_head"))?,
            num_plddt_bins,
        })
    }

    /// Compute confidence predictions from trunk representations.
    ///
    /// # Arguments
    /// * `single` — `[B, N_tok, d_single]`
    /// * `pair`   — `[B, N_tok, N_tok, d_pair]`
    pub fn forward(&self, single: &Tensor, pair: &Tensor) -> Result<ConfidenceOutput> {
        // Refine pair with 4-layer Pairformer trunk (single passes through)
        let mut pair = pair.clone();
        for block in &self.trunk {
            pair = block.forward(&pair)?;
        }

        // pLDDT: [B, N, d_single] → [B, N, 50] → scalar [B, N]
        let plddt_logits = self.plddt_head.forward(single)?;
        let plddt = plddt_from_logits(&plddt_logits)?;

        // pAE: [B, N, N, d_pair] → [B, N, N, 64]
        let pae_logits = self.pae_head.forward(&pair)?;

        // pDE: [B, N, N, d_pair] → [B, N, N, 64]
        let pde_logits = self.pde_head.forward(&pair)?;

        // Distogram: [B, N, N, d_pair] → [B, N, N, 39]
        let distogram_logits = self.distogram_head.forward(&pair)?;

        Ok(ConfidenceOutput {
            plddt_logits,
            plddt,
            pae_logits: Some(pae_logits),
            pde_logits: Some(pde_logits),
            distogram_logits: Some(distogram_logits),
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    const B: usize = 1;
    const N: usize = 12;
    const D_SINGLE: usize = 384;
    const D_PAIR: usize = 256;

    fn make_head() -> ConfidenceHead {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        ConfidenceHead::load(vb, D_SINGLE, D_PAIR, 50, 64, 64, 39).unwrap()
    }

    #[test]
    fn test_confidence_head_output_shapes() {
        let head = make_head();
        let device = Device::Cpu;
        let single = Tensor::zeros(&[B, N, D_SINGLE], DType::F32, &device).unwrap();
        let pair = Tensor::zeros(&[B, N, N, D_PAIR], DType::F32, &device).unwrap();
        let out = head.forward(&single, &pair).unwrap();

        assert_eq!(out.plddt_logits.dims(), &[B, N, 50]);
        assert_eq!(out.plddt.dims(), &[B, N]);
        assert_eq!(out.pae_logits.unwrap().dims(), &[B, N, N, 64]);
        assert_eq!(out.pde_logits.unwrap().dims(), &[B, N, N, 64]);
        assert_eq!(out.distogram_logits.unwrap().dims(), &[B, N, N, 39]);
    }

    #[test]
    fn test_plddt_range_with_uniform_logits() {
        // Uniform logits → softmax uniform → weighted mean = midpoint = 0.5
        let device = Device::Cpu;
        let logits = Tensor::zeros(&[1, 4, 50], DType::F32, &device).unwrap();
        let plddt = plddt_from_logits(&logits).unwrap();
        let vals = plddt.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in &vals {
            assert!(
                (*v - 0.5).abs() < 1e-4,
                "uniform logits → pLDDT ≈ 0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_bins_to_scalar_extremes() {
        let device = Device::Cpu;
        // All logit mass on first bin → min value
        let mut logits_vec = vec![0.0f32; 10];
        logits_vec[0] = 100.0; // very large → softmax ≈ 1.0 at bin 0
        let logits = Tensor::from_vec(logits_vec, &[1, 1, 10], &device).unwrap();
        let scalar = bins_to_scalar(&logits, 0.0, 1.0).unwrap();
        let val = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(val < 0.01, "mass on first bin → scalar ≈ 0.0, got {val}");

        // All mass on last bin → max value
        let mut logits_vec = vec![0.0f32; 10];
        logits_vec[9] = 100.0;
        let logits = Tensor::from_vec(logits_vec, &[1, 1, 10], &device).unwrap();
        let scalar = bins_to_scalar(&logits, 0.0, 1.0).unwrap();
        let val = scalar.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(val > 0.99, "mass on last bin → scalar ≈ 1.0, got {val}");
    }
}
