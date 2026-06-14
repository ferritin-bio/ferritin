//! ESM3 structure tokenization.
//!
//! Structure tokens are VQ-VAE codebook indices (0..4095). This module
//! handles wrapping them with BOS/EOS special tokens for model input.

use crate::esm3::utils::constants::{STRUCTURE_BOS_TOKEN, STRUCTURE_EOS_TOKEN};

/// Wrap VQ-VAE codebook indices with optional BOS/EOS special tokens.
///
/// `codes` should contain raw VQ-VAE indices in [0, 4095]. The function
/// does not validate range — that is left to the VQ-VAE encoder output.
pub fn tokenize_structure(codes: &[u32], add_special_tokens: bool) -> Vec<u32> {
    let mut tokens = Vec::with_capacity(codes.len() + 2);
    if add_special_tokens {
        tokens.push(STRUCTURE_BOS_TOKEN);
    }
    tokens.extend_from_slice(codes);
    if add_special_tokens {
        tokens.push(STRUCTURE_EOS_TOKEN);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esm3::utils::constants::{STRUCTURE_BOS_TOKEN, STRUCTURE_EOS_TOKEN};

    #[test]
    fn test_tokenize_structure_no_special() {
        let codes = vec![0u32, 100, 4095];
        assert_eq!(tokenize_structure(&codes, false), codes);
    }

    #[test]
    fn test_tokenize_structure_with_special() {
        let codes = vec![42u32];
        let tokens = tokenize_structure(&codes, true);
        assert_eq!(tokens, vec![STRUCTURE_BOS_TOKEN, 42, STRUCTURE_EOS_TOKEN]);
    }

    #[test]
    fn test_tokenize_structure_empty() {
        let tokens = tokenize_structure(&[], true);
        assert_eq!(tokens, vec![STRUCTURE_BOS_TOKEN, STRUCTURE_EOS_TOKEN]);
    }
}
