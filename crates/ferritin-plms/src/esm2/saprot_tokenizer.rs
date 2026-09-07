//! SaProt's structure-aware tokenizer (ferritin-goh.3).
//!
//! SaProt is the ESM-2 architecture with a different alphabet. Each residue is
//! an (amino acid, 3Di structural state) pair over a 20×20 product alphabet —
//! `"Ma"`, `"Ld"`, `"Kp"` — so a sequence is read two characters at a time and
//! `"MdAaLp"` is **three** residues, not six.
//!
//! # Why not `tokenizers::Tokenizer`
//!
//! SaProt's repo ships no `tokenizer.json`. It has a bare `vocab.txt`: 446
//! lines, one token per line, the line number being the id. Reconstructing an
//! HF tokenizer from that would mean assembling a `WordLevel` model plus a
//! two-character `Split` pre-tokenizer to reproduce behaviour that is, written
//! directly, a lookup table and `chunks(2)`.
//!
//! # Vocabulary layout
//!
//! ```text
//! 0  <cls>
//! 1  <pad>
//! 2  <eos>
//! 3  <unk>
//! 4  <mask>
//! 5+ Ap, Ay, An, ...   (amino acid, 3Di state) pairs
//! ```
//!
//! Those ids match `config.json`'s `pad_token_id: 1` and `mask_token_id: 4`,
//! which is the cross-check that the file is being read as intended.

use anyhow::{Result, bail};
use std::collections::HashMap;

/// Characters per residue in SaProt's product alphabet.
pub const CHARS_PER_RESIDUE: usize = 2;

/// A tokenizer built from SaProt's `vocab.txt`.
#[derive(Debug, Clone)]
pub struct SaProtTokenizer {
    ids: HashMap<String, u32>,
    vocab: Vec<String>,
}

impl SaProtTokenizer {
    /// Parse a `vocab.txt`: one token per line, line number is the id.
    ///
    /// Rejects a file whose special tokens are absent or misplaced, since the
    /// ids are load-bearing — `<cls>` and `<eos>` wrap every sequence, and a
    /// silently-wrong `<unk>` would turn unknown residues into a real amino
    /// acid rather than an unknown.
    pub fn from_vocab_txt(contents: &str) -> Result<Self> {
        let vocab: Vec<String> = contents
            .lines()
            .map(|l| l.trim_end_matches('\r').to_string())
            .filter(|l| !l.is_empty())
            .collect();

        if vocab.is_empty() {
            bail!("SaProt vocab.txt is empty");
        }

        let ids: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        for (expected_id, token) in [(0u32, "<cls>"), (1, "<pad>"), (2, "<eos>"), (3, "<unk>")] {
            match ids.get(token) {
                Some(&id) if id == expected_id => {}
                Some(&id) => bail!(
                    "SaProt vocab.txt has {token} at id {id}, expected {expected_id}; \
                     the special-token ids are load-bearing"
                ),
                None => bail!("SaProt vocab.txt is missing {token}"),
            }
        }

        Ok(Self { ids, vocab })
    }

    /// Number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.vocab.len()
    }

    /// Whether the vocabulary is empty. Never true for a valid vocab.
    pub fn is_empty(&self) -> bool {
        self.vocab.is_empty()
    }

    /// Look up a token's id.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    /// How many residues `sequence` encodes.
    ///
    /// A trailing odd character is its own (malformed) residue rather than
    /// being dropped, so [`encode`][Self::encode] can report it as `<unk>`
    /// instead of silently shortening the sequence.
    pub fn residue_count(&self, sequence: &str) -> usize {
        sequence.len().div_ceil(CHARS_PER_RESIDUE)
    }

    /// Encode a SaProt sequence to token ids, **without** special tokens.
    ///
    /// Unknown pairs map to `<unk>` rather than erroring: a residue whose 3Di
    /// state could not be determined is written with a `#` placeholder in
    /// practice, and refusing the whole sequence for one such residue would be
    /// unhelpful.
    pub fn encode(&self, sequence: &str) -> Vec<u32> {
        let unk = self.ids["<unk>"];
        sequence
            .as_bytes()
            .chunks(CHARS_PER_RESIDUE)
            .map(|pair| {
                std::str::from_utf8(pair)
                    .ok()
                    .and_then(|t| self.ids.get(t).copied())
                    .unwrap_or(unk)
            })
            .collect()
    }

    /// Decode ids back to a SaProt sequence, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .filter(|t| !(t.starts_with('<') && t.ends_with('>')))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first lines of the real SaProt vocab.txt, plus a few residue pairs.
    fn vocab() -> &'static str {
        "<cls>\n<pad>\n<eos>\n<unk>\n<mask>\nAp\nAy\nAn\nMd\nLp\n"
    }

    #[test]
    fn test_special_token_ids_match_the_config() {
        let t = SaProtTokenizer::from_vocab_txt(vocab()).unwrap();
        assert_eq!(t.token_to_id("<cls>"), Some(0));
        assert_eq!(t.token_to_id("<pad>"), Some(1));
        assert_eq!(t.token_to_id("<eos>"), Some(2));
        assert_eq!(t.token_to_id("<unk>"), Some(3));
        // config.json declares mask_token_id 4 and pad_token_id 1.
        assert_eq!(t.token_to_id("<mask>"), Some(4));
    }

    /// Two characters per residue — the property the whole module exists for.
    #[test]
    fn test_encodes_two_characters_per_residue() {
        let t = SaProtTokenizer::from_vocab_txt(vocab()).unwrap();
        let ids = t.encode("ApAyAn");
        assert_eq!(ids.len(), 3, "six characters are three residues");
        assert_eq!(ids, vec![5, 6, 7]);
        assert_eq!(t.residue_count("ApAyAn"), 3);
    }

    #[test]
    fn test_unknown_pairs_become_unk_not_an_error() {
        let t = SaProtTokenizer::from_vocab_txt(vocab()).unwrap();
        // "Zz" is not in the vocabulary.
        assert_eq!(t.encode("ApZz"), vec![5, 3]);
    }

    /// A trailing odd character is reported as unknown rather than dropped,
    /// so a malformed sequence does not silently lose its last residue.
    #[test]
    fn test_odd_length_input_is_not_silently_truncated() {
        let t = SaProtTokenizer::from_vocab_txt(vocab()).unwrap();
        assert_eq!(t.residue_count("ApA"), 2);
        assert_eq!(t.encode("ApA"), vec![5, 3]);
    }

    #[test]
    fn test_decode_round_trips_and_drops_specials() {
        let t = SaProtTokenizer::from_vocab_txt(vocab()).unwrap();
        let ids = t.encode("ApMdLp");
        assert_eq!(t.decode(&ids), "ApMdLp");

        let wrapped = [&[0u32][..], &ids, &[2u32][..]].concat();
        assert_eq!(t.decode(&wrapped), "ApMdLp", "specials should be dropped");
    }

    /// A vocab whose specials are misplaced is rejected, because those ids are
    /// used directly rather than looked up everywhere.
    #[test]
    fn test_rejects_misplaced_special_tokens() {
        let bad = "<pad>\n<cls>\n<eos>\n<unk>\n<mask>\nAp\n";
        let err = SaProtTokenizer::from_vocab_txt(bad)
            .map(|_| ())
            .expect_err("swapped cls/pad must be rejected");
        assert!(err.to_string().contains("load-bearing"), "got: {err}");
    }

    #[test]
    fn test_rejects_a_vocab_missing_specials() {
        let err = SaProtTokenizer::from_vocab_txt("Ap\nAy\n")
            .map(|_| ())
            .expect_err("a vocab without <cls> is not a SaProt vocab");
        assert!(err.to_string().contains("missing"), "got: {err}");
    }
}
