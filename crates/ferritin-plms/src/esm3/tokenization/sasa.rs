//! ESM3 SASA (solvent-accessible surface area) tokenization.
//!
//! Continuous SASA values (Å²) are discretized into 16 bins using
//! the 15 boundary values from `esm/utils/constants/esm3.py`.
//!
//! Token layout:
//!   0 = PAD, 1 = MASK, 2 = UNK
//!   3..18 = bins 0..15   (total 19 tokens)

use crate::esm3::utils::constants::SASA_DISCRETIZATION_BOUNDARIES;

/// Discretize a slice of per-residue SASA values into token IDs.
///
/// Each value is placed in the leftmost bin where `value < boundary`.
/// Values above all boundaries fall into the highest bin.
pub fn tokenize_sasa(sasa_values: &[f32]) -> Vec<u32> {
    sasa_values
        .iter()
        .map(|&v| {
            let bin = SASA_DISCRETIZATION_BOUNDARIES
                .iter()
                .position(|&b| v < b)
                .unwrap_or(SASA_DISCRETIZATION_BOUNDARIES.len());
            bin as u32 + 3 // offset by 3 for PAD, MASK, UNK
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_sasa_lowest_bin() {
        // 0.0 < 0.8 → bin 0 → token 3
        assert_eq!(tokenize_sasa(&[0.0]), vec![3]);
    }

    #[test]
    fn test_tokenize_sasa_highest_bin() {
        // 200.0 > all boundaries → bin 15 → token 18
        assert_eq!(tokenize_sasa(&[200.0]), vec![18]);
    }

    #[test]
    fn test_tokenize_sasa_midpoint() {
        // 5.0 >= 0.8, 5.0 >= 4.0, 5.0 < 9.6 → bin 2 → token 5
        assert_eq!(tokenize_sasa(&[5.0]), vec![5]);
    }

    #[test]
    fn test_tokenize_sasa_batch() {
        let ids = tokenize_sasa(&[0.0, 200.0]);
        assert_eq!(ids, vec![3, 18]);
    }
}
