//! ESM3 token constants.
//!
//! All values are taken directly from `esm/utils/constants/esm3.py`.

// Re-export the shared sequence vocabulary from esmc (identical alphabet).
pub use crate::esmc::utils::constants::esm3::SEQUENCE_VOCAB;

// ── Sequence track ────────────────────────────────────────────────────────────
pub const SEQUENCE_BOS_TOKEN: u32 = 0;
pub const SEQUENCE_PAD_TOKEN: u32 = 1;
pub const SEQUENCE_EOS_TOKEN: u32 = 2;
pub const SEQUENCE_CHAINBREAK_TOKEN: u32 = 31;
pub const SEQUENCE_MASK_TOKEN: u32 = 32;

// ── Structure track (VQ-VAE) ──────────────────────────────────────────────────
pub const VQVAE_CODEBOOK_SIZE: u32 = 4096;
pub const STRUCTURE_MASK_TOKEN: u32 = VQVAE_CODEBOOK_SIZE; // 4096
pub const STRUCTURE_EOS_TOKEN: u32 = VQVAE_CODEBOOK_SIZE + 1; // 4097
pub const STRUCTURE_BOS_TOKEN: u32 = VQVAE_CODEBOOK_SIZE + 2; // 4098
pub const STRUCTURE_PAD_TOKEN: u32 = VQVAE_CODEBOOK_SIZE + 3; // 4099
pub const STRUCTURE_CHAINBREAK_TOKEN: u32 = VQVAE_CODEBOOK_SIZE + 4; // 4100
pub const STRUCTURE_UNDEFINED_TOKEN: u32 = 955;

// ── SS8 secondary-structure track ────────────────────────────────────────────
// Vocab: PAD=0, MASK=1, UNK=2, then "GHITEBSC" at 3..10  (total 11)
pub const SS8_PAD_TOKEN: u32 = 0;
pub const SS8_MASK_TOKEN: u32 = 1;
pub const SS8_UNK_TOKEN: u32 = 2;
pub const SS8_VOCAB: &str = "GHITEBSC";

// ── SASA track ────────────────────────────────────────────────────────────────
// 15 boundaries → 16 bins; PAD=0, MASK=1, UNK=2, bins at 3..18  (total 19)
pub const SASA_PAD_TOKEN: u32 = 0;
pub const SASA_MASK_TOKEN: u32 = 1;
pub const SASA_UNK_TOKEN: u32 = 2;
pub const SASA_DISCRETIZATION_BOUNDARIES: &[f32] = &[
    0.8, 4.0, 9.6, 16.4, 24.5, 32.9, 42.0, 51.5, 61.2, 70.9, 81.6, 93.3, 107.2, 125.4, 151.4,
];

// ── Function / annotation tracks ─────────────────────────────────────────────
pub const INTERPRO_PAD_TOKEN: u32 = 0;
pub const RESIDUE_PAD_TOKEN: u32 = 0;
