//! ProtBert's residue tokenizer (ferritin-goh.7).
//!
//! ProtBert is BERT over an alphabet of single amino-acid letters. Upstream it
//! is driven by a `BertTokenizer` in WordPiece mode, which splits on
//! **whitespace** — so the documented way to call it is `"M L K L R V"`, not
//! `"MLKLRV"`.
//!
//! That detail is the reason this module exists rather than a `TokenizerSpec`
//! pointing at a stock tokenizer. Handing WordPiece an unspaced sequence does
//! not fail: `"MLKLRV"` is not in the vocabulary and there are no `##`
//! continuation tokens to fall back on, so the whole protein collapses to a
//! single `[UNK]`. The model then returns a confident, correctly-shaped
//! embedding of the wrong length. Nothing raises.
//!
//! So the whitespace convention is treated here as an artifact of the upstream
//! tokenizer, not as part of this crate's contract. [`encode`][Self::encode]
//! ignores whitespace entirely and reads one residue per character, which makes
//! `"MLKLRV"` and `"M L K L R V"` the same six residues. Callers pass sequences
//! in the same form as for every other model in the registry.
//!
//! # Why not `tokenizers::Tokenizer`
//!
//! Neither `Rostlab/prot_bert` nor `Rostlab/prot_bert_bfd` ships a
//! `tokenizer.json` — only a 30-line `vocab.txt`. Reconstructing an HF
//! tokenizer would mean assembling a `WordPiece` model plus a whitespace
//! pre-tokenizer in order to reproduce, at more cost, the behaviour this file
//! states directly: a lookup table over single characters.
//!
//! # Vocabulary layout
//!
//! ```text
//! 0  [PAD]
//! 1  [UNK]
//! 2  [CLS]
//! 3  [SEP]
//! 4  [MASK]
//! 5+ L A G V E S I K R D T P N Q F Y M H C W X U B Z O
//! ```

use anyhow::{Result, bail};
use std::collections::HashMap;

/// The token an unrecognised residue becomes.
///
/// `X` — "any amino acid" — rather than `[UNK]`. Rostlab's own usage
/// documentation preprocesses sequences with `re.sub(r"[UZOB]", "X", seq)`
/// before tokenizing, so `X` is the symbol this checkpoint was actually
/// trained to read for an uninformative residue. `[UNK]` exists in the
/// vocabulary but denotes a token outside the alphabet altogether, which is a
/// different statement and one the model saw far less of.
pub const UNKNOWN_RESIDUE: &str = "X";

/// A tokenizer built from ProtBert's `vocab.txt`.
#[derive(Debug, Clone)]
pub struct ProtBertTokenizer {
    ids: HashMap<String, u32>,
    vocab: Vec<String>,
}

impl ProtBertTokenizer {
    /// Parse a `vocab.txt`: one token per line, the line number being the id.
    ///
    /// The special-token ids are checked against `config.json`'s declared
    /// values rather than trusted, because they are used positionally:
    /// `[CLS]`/`[SEP]` wrap every sequence and `[PAD]` id 0 is what a batched
    /// attention mask is built against. A vocab whose order differs would
    /// otherwise load cleanly and be wrong.
    pub fn from_vocab_txt(contents: &str) -> Result<Self> {
        let vocab: Vec<String> = contents
            .lines()
            .map(|l| l.trim_end_matches('\r').to_string())
            .filter(|l| !l.is_empty())
            .collect();

        if vocab.is_empty() {
            bail!("ProtBert vocab.txt is empty");
        }

        let ids: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        for (expected_id, token) in [
            (0u32, "[PAD]"),
            (1, "[UNK]"),
            (2, "[CLS]"),
            (3, "[SEP]"),
            (4, "[MASK]"),
        ] {
            match ids.get(token) {
                Some(&id) if id == expected_id => {}
                Some(&id) => bail!(
                    "ProtBert vocab.txt has {token} at id {id}, expected {expected_id}; \
                     the special-token ids are load-bearing"
                ),
                None => bail!("ProtBert vocab.txt is missing {token}"),
            }
        }

        if !ids.contains_key(UNKNOWN_RESIDUE) {
            bail!(
                "ProtBert vocab.txt has no {UNKNOWN_RESIDUE} token to map \
                 unrecognised residues onto"
            );
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
    /// Whitespace is separator, not content, so `"M L K"` and `"MLK"` both
    /// count three.
    pub fn residue_count(&self, sequence: &str) -> usize {
        sequence.chars().filter(|c| !c.is_whitespace()).count()
    }

    /// Encode a sequence to token ids, **without** `[CLS]`/`[SEP]`.
    ///
    /// Accepts spaced and unspaced input identically; unrecognised residues
    /// become [`UNKNOWN_RESIDUE`].
    pub fn encode(&self, sequence: &str) -> Vec<u32> {
        let unknown = self.ids[UNKNOWN_RESIDUE];
        sequence
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| {
                self.ids
                    .get(c.to_ascii_uppercase().to_string().as_str())
                    .copied()
                    .unwrap_or(unknown)
            })
            .collect()
    }

    /// Wrap encoded ids in `[CLS]` … `[SEP]`, the form the model expects.
    pub fn encode_with_specials(&self, sequence: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(self.residue_count(sequence) + 2);
        ids.push(self.ids["[CLS]"]);
        ids.extend(self.encode(sequence));
        ids.push(self.ids["[SEP]"]);
        ids
    }

    /// Decode ids back to a sequence, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .filter(|t| !(t.starts_with('[') && t.ends_with(']')))
            .map(String::as_str)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real ProtBert vocab.txt: 5 specials then 25 residue symbols.
    fn vocab() -> &'static str {
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nL\nA\nG\nV\nE\nS\nI\nK\nR\nD\nT\nP\nN\nQ\nF\nY\nM\nH\nC\nW\nX\nU\nB\nZ\nO\n"
    }

    #[test]
    fn test_special_token_ids_match_the_config() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        assert_eq!(t.token_to_id("[PAD]"), Some(0));
        assert_eq!(t.token_to_id("[UNK]"), Some(1));
        assert_eq!(t.token_to_id("[CLS]"), Some(2));
        assert_eq!(t.token_to_id("[SEP]"), Some(3));
        assert_eq!(t.token_to_id("[MASK]"), Some(4));
        assert_eq!(t.len(), 30, "config.json declares vocab_size 30");
    }

    /// The property the whole module exists for: the spacing ProtBert's own
    /// tokenizer requires is not part of this crate's contract, so both forms
    /// give the same residues.
    #[test]
    fn test_spaced_and_unspaced_input_are_identical() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        assert_eq!(t.encode("MLKLRV"), t.encode("M L K L R V"));
        assert_eq!(t.residue_count("MLKLRV"), 6);
        assert_eq!(t.residue_count("M L K L R V"), 6);
    }

    /// Upstream, an unspaced sequence collapses to one `[UNK]`. If that ever
    /// starts happening here, every downstream embedding is silently the wrong
    /// length — so pin the token count explicitly.
    #[test]
    fn test_unspaced_sequence_is_not_one_unknown_token() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        let ids = t.encode("MLKLRV");
        assert_eq!(ids.len(), 6, "one token per residue, not one per word");
        assert!(
            !ids.contains(&t.token_to_id("[UNK]").unwrap()),
            "every residue here is in the vocabulary"
        );
    }

    #[test]
    fn test_unknown_residues_become_x_not_unk() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        // 'J' is not an amino-acid symbol in this alphabet.
        let ids = t.encode("MJL");
        assert_eq!(ids, vec![21, 25, 5], "J maps to X (25), not [UNK] (1)");
    }

    /// U, B, Z and O are rare but genuinely in the alphabet — they must survive
    /// rather than being folded into X.
    #[test]
    fn test_rare_residues_are_not_folded_into_x() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        let x = t.token_to_id("X").unwrap();
        for residue in ["U", "B", "Z", "O"] {
            let ids = t.encode(residue);
            assert_ne!(ids[0], x, "{residue} is in the vocabulary in its own right");
        }
    }

    #[test]
    fn test_specials_wrap_the_sequence() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        let ids = t.encode_with_specials("MLK");
        assert_eq!(ids.first(), Some(&2), "[CLS] leads");
        assert_eq!(ids.last(), Some(&3), "[SEP] trails");
        assert_eq!(ids.len(), 5, "three residues plus two specials");
    }

    #[test]
    fn test_decode_round_trips_and_drops_specials() {
        let t = ProtBertTokenizer::from_vocab_txt(vocab()).unwrap();
        assert_eq!(t.decode(&t.encode_with_specials("MLKLRV")), "MLKLRV");
    }

    #[test]
    fn test_rejects_misplaced_special_tokens() {
        let bad = "[UNK]\n[PAD]\n[CLS]\n[SEP]\n[MASK]\nL\nX\n";
        let err = ProtBertTokenizer::from_vocab_txt(bad)
            .map(|_| ())
            .expect_err("swapped pad/unk must be rejected");
        assert!(err.to_string().contains("load-bearing"), "got: {err}");
    }

    #[test]
    fn test_rejects_a_vocab_missing_specials() {
        let err = ProtBertTokenizer::from_vocab_txt("L\nA\nG\n")
            .map(|_| ())
            .expect_err("a vocab without [PAD] is not a ProtBert vocab");
        assert!(err.to_string().contains("missing"), "got: {err}");
    }
}
