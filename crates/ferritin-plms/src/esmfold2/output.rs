//! ESMFold2 output types.
//!
//! Defines [`ESMFold2Output`], the result of a full ESMFold2 forward pass.

use candle_core::Tensor;

/// Output of the ESMFold2 forward pass.
#[derive(Debug)]
pub struct ESMFold2Output {
    /// All-atom coordinates, shape `(Bm, N_atom, 3)` in Ångströms.
    pub sample_atom_coords: Tensor,
    /// Per-token pLDDT confidence, shape `(Bm, N_tok)`, range [0, 1].
    pub plddt: Tensor,
    /// Predicted TM-score, shape `(Bm,)`.
    pub ptm: Tensor,
    /// Interface pTM (for multi-chain), shape `(Bm,)`.
    pub iptm: Tensor,
    /// Predicted Aligned Error, shape `(Bm, N_tok, N_tok)` in Å. Optional.
    pub pae: Option<Tensor>,
    /// Distogram logits, shape `(Bm, N_tok, N_tok, distogram_bins)`. Optional.
    pub distogram_logits: Option<Tensor>,
}

impl ESMFold2Output {
    /// Convert this output to mmCIF text for a single protein chain.
    ///
    /// pLDDT (range [0, 1]) is written as B-factor × 100.
    pub fn to_mmcif(
        &self,
        sequence: &str,
        chain_id: &str,
        entry_id: &str,
    ) -> anyhow::Result<String> {
        super::mmcif::coords_to_mmcif(
            &self.sample_atom_coords,
            sequence,
            &self.plddt,
            chain_id,
            entry_id,
        )
    }
}
