//! Pair representation initialization for the ESMFold2 FoldingTrunk.
//!
//! Produces the initial pair tensor `[B, N, N, d_pair]` from three sources:
//!
//! 1. **Relative position encoding** — for each (i,j), the clipped relative
//!    residue-index difference `clip(j-i, -n_bins, n_bins)` is one-hot encoded
//!    into `2*n_bins+1 = 65` bins.
//! 2. **Chain indicator** — two extra bins: same-chain / cross-chain.
//! 3. **Outer product** — the single representation `[B,N,d_single]` is
//!    projected to `d_outer`, outer-producted with itself, then projected
//!    to `d_pair`.
//!
//! The three contributions are projected to `d_pair` and summed.
//!
//! This module contains standalone math functions (`relpos_encoding`,
//! `chain_pair_features`, `outer_product`) that are testable without weights,
//! plus the [`PairInit`] struct that holds the learned projections.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, VarBuilder, encoding::one_hot, linear};

// ── Pure math ─────────────────────────────────────────────────────────────

/// Relative-residue-index one-hot encoding for every (i,j) pair.
///
/// For each pair: `d = clip(j - i, -n_bins, n_bins)`, shifted to `[0, 2*n_bins]`,
/// then one-hot encoded.
///
/// # Arguments
/// * `residue_indices` — `[B, N]` integer residue positions (any integer dtype)
/// * `n_bins`          — half-width of the window (config: 32)
///
/// # Returns
/// `[B, N, N, 2*n_bins+1]` float32
pub fn relpos_encoding(residue_indices: &Tensor, n_bins: usize) -> Result<Tensor> {
    let ri = residue_indices.to_dtype(DType::F32)?.unsqueeze(2)?; // [B, N, 1]
    let rj = residue_indices.to_dtype(DType::F32)?.unsqueeze(1)?; // [B, 1, N]
    let diff = rj.broadcast_sub(&ri)?; // [B, N, N]

    let nb = n_bins as f64;
    let clamped = diff.clamp(-nb, nb)?;
    let shifted = (clamped + nb)?; // values in [0, 2*n_bins]

    let shifted_u32 = shifted.to_dtype(DType::U32)?;
    one_hot(shifted_u32, 2 * n_bins + 1, 1.0f32, 0.0f32) // [B, N, N, 2*n_bins+1]
}

/// Per-pair same-chain / different-chain indicator.
///
/// # Arguments
/// * `chain_ids` — `[B, N]` integer chain identifiers (0, 1, 2, …)
///
/// # Returns
/// `[B, N, N, 2]` float32, where dim-3 is `[same_chain, diff_chain]`.
pub fn chain_pair_features(chain_ids: &Tensor) -> Result<Tensor> {
    let ci = chain_ids.to_dtype(DType::F32)?.unsqueeze(2)?; // [B, N, 1]
    let cj = chain_ids.to_dtype(DType::F32)?.unsqueeze(1)?; // [B, 1, N]
    let diff = ci.broadcast_sub(&cj)?; // [B, N, N]

    let same = diff.abs()?.lt(0.5_f64)?.to_dtype(DType::F32)?; // [B, N, N]
    let different = (1.0_f64 - &same)?;
    Tensor::stack(&[&same, &different], 3) // [B, N, N, 2]
}

/// Flat outer product of two `[B, N, d]` tensors.
///
/// Each (i,j) entry is the flattened outer product of row i from `a` and
/// row j from `b`.
///
/// # Returns
/// `[B, N, N, da * db]`
pub fn outer_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let (batch, n, da) = a.dims3()?;
    let db = b.dim(2)?;
    let ai = a.reshape((batch, n, 1, da, 1))?; // [B, N, 1, da, 1]
    let bj = b.reshape((batch, 1, n, 1, db))?; // [B, 1, N, 1, db]
    let prod = ai.broadcast_mul(&bj)?; // [B, N, N, da, db]
    prod.reshape((batch, n, n, da * db)) // [B, N, N, da*db]
}

// ── PairInit ──────────────────────────────────────────────────────────────

/// Learnable pair-representation initializer.
///
/// Combines relative-position + chain features and an outer-product term to
/// produce the initial `[B, N, N, d_pair]` tensor fed into the Pairformer.
pub struct PairInit {
    relpos_proj: Linear,
    single_proj: Linear,
    outer_proj: Linear,
    n_relpos_bins: usize,
}

impl PairInit {
    /// Load from a `VarBuilder` rooted at `pair_init.*`.
    ///
    /// # Arguments
    /// * `d_single`       — single-repr dim (384)
    /// * `d_pair`         — output pair dim (256)
    /// * `n_relpos_bins`  — half-window for relpos (32 → 65 bins)
    /// * `d_outer`        — inner dim for outer-product projection (32)
    pub fn load(
        vb: VarBuilder,
        d_single: usize,
        d_pair: usize,
        n_relpos_bins: usize,
        d_outer: usize,
    ) -> Result<Self> {
        let relpos_in = 2 * n_relpos_bins + 1 + 2; // 65 relpos bins + 2 chain bins
        Ok(Self {
            relpos_proj: linear::linear_no_bias(relpos_in, d_pair, vb.pp("relpos_proj"))?,
            single_proj: linear::linear_no_bias(d_single, d_outer, vb.pp("single_proj"))?,
            outer_proj: linear::linear_no_bias(d_outer * d_outer, d_pair, vb.pp("outer_proj"))?,
            n_relpos_bins,
        })
    }

    /// Build the initial pair representation.
    ///
    /// # Arguments
    /// * `single`          — `[B, N, d_single]`
    /// * `residue_indices` — `[B, N]` integer residue positions
    /// * `chain_ids`       — `[B, N]` integer chain identifiers
    ///
    /// # Returns
    /// `[B, N, N, d_pair]`
    pub fn forward(
        &self,
        single: &Tensor,
        residue_indices: &Tensor,
        chain_ids: &Tensor,
    ) -> Result<Tensor> {
        // RelPos + chain → [B, N, N, d_pair]
        let relpos = relpos_encoding(residue_indices, self.n_relpos_bins)?;
        let chain = chain_pair_features(chain_ids)?;
        let pos_feat = Tensor::cat(&[&relpos, &chain], 3)?;
        let pair_from_pos = pos_feat.apply(&self.relpos_proj)?;

        // Outer product → [B, N, N, d_pair]
        let s_proj = single.apply(&self.single_proj)?;
        let outer = outer_product(&s_proj, &s_proj)?;
        let pair_from_outer = outer.apply(&self.outer_proj)?;

        pair_from_pos + pair_from_outer
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;

    const B: usize = 1;
    const N: usize = 8;
    const N_BINS: usize = 32;

    fn arange_indices(n: usize) -> Tensor {
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Tensor::from_vec(vals, &[B, N], &Device::Cpu).unwrap()
    }

    #[test]
    fn test_relpos_encoding_shape() {
        let idx = arange_indices(N);
        let enc = relpos_encoding(&idx, N_BINS).unwrap();
        assert_eq!(enc.dims(), &[B, N, N, 2 * N_BINS + 1]);
    }

    // Helper: get a 1D slice from a 4D tensor at [b, i, j, :]
    fn get_bin_vec(t: &Tensor, b: usize, i: usize, j: usize) -> Vec<f32> {
        t.get(b)
            .unwrap()
            .get(i)
            .unwrap()
            .get(j)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    // Helper: get a 1D slice from a 3D tensor at [b, i, :]
    fn get_pair_vec(t: &Tensor, b: usize, i: usize, j: usize) -> Vec<f32> {
        t.get(b)
            .unwrap()
            .get(i)
            .unwrap()
            .get(j)
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    #[test]
    fn test_relpos_encoding_diagonal_is_centre_bin() {
        let idx = arange_indices(N);
        let enc = relpos_encoding(&idx, N_BINS).unwrap();
        // On the diagonal i==j, relative pos = 0 → shifted = n_bins → hot at index n_bins
        for i in 0..N {
            let vals = get_bin_vec(&enc, 0, i, i);
            assert!(
                (vals[N_BINS] - 1.0).abs() < 1e-5,
                "diagonal [{i},{i}] should be hot at bin {N_BINS}, got {vals:?}"
            );
            let off: f32 = vals
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != N_BINS)
                .map(|(_, &v)| v)
                .sum();
            assert!(off.abs() < 1e-5, "off bins should be zero");
        }
    }

    #[test]
    fn test_relpos_encoding_off_diagonal() {
        let idx = arange_indices(N);
        let enc = relpos_encoding(&idx, N_BINS).unwrap();

        // pair (0,1): relative pos = +1 → shifted = n_bins+1
        let row = get_bin_vec(&enc, 0, 0, 1);
        assert!((row[N_BINS + 1] - 1.0).abs() < 1e-5);

        // pair (1,0): relative pos = -1 → shifted = n_bins-1
        let row = get_bin_vec(&enc, 0, 1, 0);
        assert!((row[N_BINS - 1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_relpos_clamped_at_boundary() {
        // diff = 100, clamped to n_bins=32, shifted = 2*32 = 64 (last bin)
        let far = Tensor::from_vec(vec![0.0f32, 100.0], &[1, 2], &Device::Cpu).unwrap();
        let enc = relpos_encoding(&far, N_BINS).unwrap();
        let row = get_bin_vec(&enc, 0, 0, 1);
        assert!(
            (row[2 * N_BINS] - 1.0).abs() < 1e-5,
            "should clamp to max bin"
        );
    }

    #[test]
    fn test_chain_pair_features_shape() {
        let chain_ids = Tensor::zeros(&[B, N], DType::F32, &Device::Cpu).unwrap();
        let feats = chain_pair_features(&chain_ids).unwrap();
        assert_eq!(feats.dims(), &[B, N, N, 2]);
    }

    #[test]
    fn test_chain_pair_features_same_chain() {
        // All residues on chain 0 → every pair is same-chain
        let chain_ids = Tensor::zeros(&[B, N], DType::F32, &Device::Cpu).unwrap();
        let feats = chain_pair_features(&chain_ids).unwrap();
        let val = get_pair_vec(&feats, 0, 0, 1);
        assert!(
            (val[0] - 1.0).abs() < 1e-5,
            "same chain index 0 should be 1"
        );
        assert!(val[1].abs() < 1e-5, "same chain index 1 should be 0");
    }

    #[test]
    fn test_chain_pair_features_cross_chain() {
        // Residues 0..4 on chain 0, residues 4..8 on chain 1
        let ids: Vec<f32> = (0..N).map(|i| if i < 4 { 0.0 } else { 1.0 }).collect();
        let chain_ids = Tensor::from_vec(ids, &[B, N], &Device::Cpu).unwrap();
        let feats = chain_pair_features(&chain_ids).unwrap();

        // Same-chain pair (0,1)
        let same = get_pair_vec(&feats, 0, 0, 1);
        assert!((same[0] - 1.0).abs() < 1e-5);
        assert!(same[1].abs() < 1e-5);

        // Cross-chain pair (0,4)
        let cross = get_pair_vec(&feats, 0, 0, 4);
        assert!(cross[0].abs() < 1e-5);
        assert!((cross[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_outer_product_shape() {
        let a = Tensor::zeros(&[B, N, 16], DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::zeros(&[B, N, 16], DType::F32, &Device::Cpu).unwrap();
        let out = outer_product(&a, &b).unwrap();
        assert_eq!(out.dims(), &[B, N, N, 256]);
    }

    #[test]
    fn test_outer_product_values() {
        // [1, 2] outer [3, 4] = [3, 4, 6, 8]
        let a = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 1, 2], &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![3.0f32, 4.0], &[1, 1, 2], &Device::Cpu).unwrap();
        let out = outer_product(&a, &b).unwrap();
        let vals = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((vals[0] - 3.0).abs() < 1e-5, "1*3 = 3");
        assert!((vals[1] - 4.0).abs() < 1e-5, "1*4 = 4");
        assert!((vals[2] - 6.0).abs() < 1e-5, "2*3 = 6");
        assert!((vals[3] - 8.0).abs() < 1e-5, "2*4 = 8");
    }

    #[test]
    fn test_pair_init_forward_shape() {
        let device = Device::Cpu;
        let d_single = 32_usize; // smaller dims for fast test
        let d_pair = 16_usize;
        let n_bins = 4_usize;
        let d_outer = 4_usize;

        let vb = VarBuilder::zeros(DType::F32, &device);
        let init = PairInit::load(vb, d_single, d_pair, n_bins, d_outer).unwrap();

        let single = Tensor::zeros(&[B, N, d_single], DType::F32, &device).unwrap();
        let residue_idx = arange_indices(N);
        let chain_ids = Tensor::zeros(&[B, N], DType::F32, &device).unwrap();

        let pair = init.forward(&single, &residue_idx, &chain_ids).unwrap();
        assert_eq!(pair.dims(), &[B, N, N, d_pair]);
    }
}
