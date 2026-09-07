//! Relative position embeddings for the ESM3 structure encoder (ferritin-100.22).
//!
//! The structure encoder is *local*: each residue is encoded from its
//! k-nearest neighbours in space, not from the whole chain. Those neighbours
//! arrive in distance order, which throws away where they sat in the sequence
//! — so the only thing telling the model that a spatial neighbour was also a
//! sequence neighbour is this embedding.
//!
//! It is also the encoder's **initial hidden state**. There is no token
//! embedding: the transformer starts from these relative offsets and lets all
//! geometry enter through geometric attention.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

/// Offsets are clamped to ±`BINS` before lookup.
pub const BINS: i64 = 32;

/// Table rows: `2 * BINS + 2`. The `+2` is one row per clamped extreme plus
/// the padding index that shifts the range to be non-negative — which is why
/// the checkpoint's table is 66 rows and not 65.
pub const NUM_EMBEDDINGS: usize = (2 * BINS + 2) as usize;

/// Embeds the signed sequence offset between a query residue and each of its
/// spatial neighbours.
#[derive(Debug)]
pub struct RelativePositionEmbedding {
    embedding: Embedding,
}

impl RelativePositionEmbedding {
    /// Load the `(66, d_model)` table from `<prefix>.embedding.weight`.
    pub fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let weight = vb.get((NUM_EMBEDDINGS, d_model), "embedding.weight")?;
        Ok(Self {
            embedding: Embedding::new(weight, d_model),
        })
    }

    /// `query`: `(N,)` centre residue indices. `key`: `(N, K)` neighbour
    /// residue indices. Returns `(N, K, d_model)`.
    ///
    /// The offset is clamped to `±BINS` and then shifted by `BINS + 1`, so an
    /// offset of `-BINS` lands on row 1 and row 0 stays reserved for padding.
    pub fn forward(&self, query: &Tensor, key: &Tensor) -> Result<Tensor> {
        let query = query.to_dtype(DType::I64)?.unsqueeze(1)?; // (N, 1)
        let diff = key.to_dtype(DType::I64)?.broadcast_sub(&query)?;
        let idx = diff.clamp(-BINS, BINS)?.affine(1.0, (BINS + 1) as f64)?;
        self.embedding.forward(&idx.to_dtype(DType::U32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn table(d: usize) -> Result<RelativePositionEmbedding> {
        // Row i is filled with the value i, so a looked-up row reports its index.
        let rows: Vec<f32> = (0..NUM_EMBEDDINGS)
            .flat_map(|i| std::iter::repeat_n(i as f32, d))
            .collect();
        let w = Tensor::from_vec(rows, (NUM_EMBEDDINGS, d), &Device::Cpu)?;
        Ok(RelativePositionEmbedding {
            embedding: Embedding::new(w, d),
        })
    }

    /// The checkpoint ships 66 rows; a 65-row table would mean the padding
    /// offset was dropped and every lookup would be off by one.
    #[test]
    fn test_table_is_sixty_six_rows() {
        assert_eq!(NUM_EMBEDDINGS, 66);
    }

    #[test]
    fn test_zero_offset_lands_on_the_centre_row() -> Result<()> {
        let t = table(1)?;
        let q = Tensor::new(&[5i64], &Device::Cpu)?;
        let k = Tensor::new(&[[5i64]], &Device::Cpu)?;
        let out = t.forward(&q, &k)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(out[0], (BINS + 1) as f32, "offset 0 -> row BINS+1");
        Ok(())
    }

    /// Offsets beyond the window saturate rather than wrapping into a
    /// different residue's row, and never reach the reserved padding row 0.
    #[test]
    fn test_offsets_clamp_and_never_hit_the_padding_row() -> Result<()> {
        let t = table(1)?;
        let q = Tensor::new(&[100i64], &Device::Cpu)?;
        let k = Tensor::new(&[[0i64, 100 - 40, 100 + 40, 200]], &Device::Cpu)?;
        let out = t.forward(&q, &k)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(out[0], 1.0, "-100 clamps to -BINS -> row 1");
        assert_eq!(out[1], 1.0, "-40 clamps to -BINS -> row 1");
        assert_eq!(out[2], (2 * BINS + 1) as f32, "+40 clamps to +BINS");
        assert_eq!(out[3], (2 * BINS + 1) as f32, "+100 clamps to +BINS");
        for v in out {
            assert_ne!(v, 0.0, "row 0 is reserved for padding");
        }
        Ok(())
    }

    #[test]
    fn test_sign_is_preserved() -> Result<()> {
        let t = table(1)?;
        let q = Tensor::new(&[10i64], &Device::Cpu)?;
        let k = Tensor::new(&[[9i64, 11]], &Device::Cpu)?;
        let out = t.forward(&q, &k)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(out[0], BINS as f32, "offset -1");
        assert_eq!(out[1], (BINS + 2) as f32, "offset +1");
        Ok(())
    }
}
