//! ESM3 sequence tokenization.

use crate::esm3::utils::constants::{SEQUENCE_BOS_TOKEN, SEQUENCE_EOS_TOKEN, SEQUENCE_VOCAB};
use std::collections::HashMap;

fn vocab_map() -> HashMap<&'static str, u32> {
    SEQUENCE_VOCAB
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, i as u32))
        .collect()
}

/// Encode an amino-acid string to ESM3 sequence token IDs.
///
/// Unknown characters map to `<unk>` (index 3).
/// When `add_special_tokens` is true, prepends BOS and appends EOS.
pub fn tokenize_sequence(sequence: &str, add_special_tokens: bool) -> Vec<u32> {
    let vocab = vocab_map();
    let unk = *vocab.get("<unk>").unwrap_or(&3);

    let mut tokens = Vec::with_capacity(sequence.len() + 2);
    if add_special_tokens {
        tokens.push(SEQUENCE_BOS_TOKEN);
    }
    for ch in sequence.chars() {
        let s = ch.to_string();
        tokens.push(vocab.get(s.as_str()).copied().unwrap_or(unk));
    }
    if add_special_tokens {
        tokens.push(SEQUENCE_EOS_TOKEN);
    }
    tokens
}

/// Decode ESM3 sequence token IDs back to an amino-acid string.
///
/// Skips BOS (0), PAD (1), EOS (2), and MASK (32).
pub fn decode_sequence(token_ids: &[u32]) -> String {
    const SPECIAL: [u32; 4] = [0, 1, 2, 32];
    let mut out = String::new();
    for &id in token_ids {
        if SPECIAL.contains(&id) {
            continue;
        }
        if let Some(tok) = SEQUENCE_VOCAB.get(id as usize) {
            out.push_str(tok);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_sequence_no_special() {
        let ids = tokenize_sequence("MA", false);
        // M is at index 20, A is at index 5
        assert_eq!(ids, vec![20, 5]);
    }

    #[test]
    fn test_tokenize_sequence_with_special() {
        let ids = tokenize_sequence("G", true);
        assert_eq!(ids[0], SEQUENCE_BOS_TOKEN);
        assert_eq!(*ids.last().unwrap(), SEQUENCE_EOS_TOKEN);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_decode_roundtrip() {
        let seq = "ACDEFGHIKLMNPQRSTVWY";
        let ids = tokenize_sequence(seq, true);
        let decoded = decode_sequence(&ids);
        assert_eq!(decoded, seq);
    }

    #[test]
    fn test_unknown_char_maps_to_unk() {
        let ids = tokenize_sequence("?", false);
        assert_eq!(ids, vec![3]); // <unk>
    }
}
