//! ESM3 SS8 secondary-structure tokenization.
//!
//! Token layout:
//!   0 = PAD, 1 = MASK, 2 = UNK
//!   3..10 = 'G','H','I','T','E','B','S','C'  (DSSP 8-class)

use crate::esm3::utils::constants::{SS8_UNK_TOKEN, SS8_VOCAB};

/// Encode an SS8 secondary-structure string to token IDs.
///
/// Valid characters are "GHITEBSC". Unknown characters map to UNK (2).
pub fn tokenize_ss8(ss8: &str) -> Vec<u32> {
    ss8.chars()
        .map(|c| match SS8_VOCAB.find(c) {
            Some(idx) => idx as u32 + 3,
            None => SS8_UNK_TOKEN,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_ss8_known() {
        // "G" is first in SS8_VOCAB → index 0 → token 3
        assert_eq!(tokenize_ss8("G"), vec![3]);
        // "C" is last → index 7 → token 10
        assert_eq!(tokenize_ss8("C"), vec![10]);
    }

    #[test]
    fn test_tokenize_ss8_unknown() {
        assert_eq!(tokenize_ss8("X"), vec![2]); // UNK
    }

    #[test]
    fn test_tokenize_ss8_full_vocab() {
        let ids = tokenize_ss8(SS8_VOCAB);
        assert_eq!(ids, vec![3, 4, 5, 6, 7, 8, 9, 10]);
    }
}
