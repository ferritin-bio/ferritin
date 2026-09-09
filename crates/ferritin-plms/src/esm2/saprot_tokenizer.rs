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

// ── Bridging from ProstT5 (ferritin-goh.13) ──────────────────────────────────

/// SaProt's placeholder for an unknown amino acid or unknown structural state.
///
/// The alphabet is a full 21x21 product — the 20 standard residues plus this,
/// crossed with the 20 3Di states plus this — so `#` is available on either
/// axis. `M#` is a known residue with unknown structure (the documented
/// sequence-only form) and `#a` is the converse.
pub const UNKNOWN_STATE: char = '#';

/// The residues SaProt has pairs for: the 20 standard amino acids.
///
/// Deliberately narrower than ProtT5's table, which also carries `X`, `B`,
/// `O`, `U` and `Z`. SaProt has no pair for those — `Xa` is simply not in the
/// vocabulary — so they have to become [`UNKNOWN_STATE`] rather than being
/// passed through. See [`interleave`].
const SAPROT_RESIDUES: &str = "ACDEFGHIKLMNPQRSTVWY";

/// The 20 Foldseek 3Di states, lowercase.
const SAPROT_STRUCTURE_STATES: &str = "acdefghiklmnpqrstvwy";

/// Weave an amino-acid sequence together with a 3Di structural string into
/// SaProt's product alphabet (ferritin-goh.13).
///
/// This is the join between
/// [`ProstT5Translator`][crate::t5::translator::ProstT5Translator], which emits
/// 3Di, and SaProt, which consumes `(residue, state)` pairs — the pair that
/// removes the Foldseek dependency from the structure-aware path.
///
/// # Non-standard residues become `#`
///
/// ProstT5's alphabet is wider than SaProt's: it has embeddings for `X`, `B`,
/// `O`, `U` and `Z`, and SaProt has no pair for any of them. Passing one
/// straight through would produce a token like `Xa`, which is **not in the
/// 441-entry vocabulary** and encodes as `<unk>` — losing the structural state
/// as well as the residue, silently.
///
/// Substituting [`UNKNOWN_STATE`] keeps the state: `#a` *is* in the vocabulary.
/// It is lossy about the residue either way, but it loses strictly less, and it
/// uses SaProt's own convention for exactly this case.
///
/// # Errors
///
/// On a length mismatch — ProstT5's translation is length-preserving, so a
/// disagreement means something upstream went wrong and silently truncating to
/// the shorter of the two would misalign every residue after the first gap —
/// and on any character that is neither a residue nor a state, which indicates
/// the two arguments were swapped or the structure string is not 3Di at all.
///
/// Whitespace is skipped in both arguments, matching the rest of the family.
pub fn interleave(residues: &str, structure: &str) -> Result<String> {
    let aa: Vec<char> = residues.chars().filter(|c| !c.is_whitespace()).collect();
    let states: Vec<char> = structure.chars().filter(|c| !c.is_whitespace()).collect();

    if aa.len() != states.len() {
        bail!(
            "SaProt bridge: {} residues against {} structural states. ProstT5's \
             translation is length-preserving, so these should always match.",
            aa.len(),
            states.len()
        );
    }
    if aa.is_empty() {
        bail!("SaProt bridge: nothing to interleave");
    }

    let mut out = String::with_capacity(aa.len() * CHARS_PER_RESIDUE);
    for (i, (&residue, &state)) in aa.iter().zip(states.iter()).enumerate() {
        let residue = if SAPROT_RESIDUES.contains(residue) || residue == UNKNOWN_STATE {
            residue
        } else if residue.is_ascii_uppercase() {
            // A non-standard residue ProstT5 knows and SaProt does not.
            UNKNOWN_STATE
        } else {
            bail!(
                "SaProt bridge: {residue:?} at position {i} is not an amino acid. \
                 Residues are uppercase and structural states lowercase — are the \
                 two arguments the right way round?"
            );
        };
        if !(SAPROT_STRUCTURE_STATES.contains(state) || state == UNKNOWN_STATE) {
            bail!(
                "SaProt bridge: {state:?} at position {i} is not a 3Di state. \
                 Structural states are lowercase; ProstT5 emits them directly."
            );
        }
        out.push(residue);
        out.push(state);
    }
    Ok(out)
}

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

    // ── The ProstT5 bridge (ferritin-goh.13) ──────────────────────────────

    /// The join itself: residue and state alternate, uppercase then lowercase.
    #[test]
    fn test_interleave_weaves_residues_and_states() {
        assert_eq!(interleave("MKT", "dvq").unwrap(), "MdKvTq");
        assert_eq!(interleave("M K T", "d v q").unwrap(), "MdKvTq");
    }

    /// The alphabet this bridge is allowed to emit is exactly 21x21.
    ///
    /// That every one of those pairs is really in SaProt's vocabulary is
    /// checked against the published `vocab.txt` in
    /// `tests/test_saprot_bridge.rs`; here we only pin the shape.
    #[test]
    fn test_bridge_alphabet_is_twenty_one_squared() {
        assert_eq!(SAPROT_RESIDUES.chars().count(), 20);
        assert_eq!(SAPROT_STRUCTURE_STATES.chars().count(), 20);
        assert!(!SAPROT_RESIDUES.contains(UNKNOWN_STATE));
        assert!(
            SAPROT_STRUCTURE_STATES
                .chars()
                .all(|c| c.is_ascii_lowercase())
        );
        assert!(SAPROT_RESIDUES.chars().all(|c| c.is_ascii_uppercase()));
    }

    /// ProstT5 knows X/B/O/U/Z; SaProt does not. They become `#`, which keeps
    /// the structural state — `#a` is in the vocabulary, `Xa` is not.
    #[test]
    fn test_non_standard_residues_become_the_unknown_state() {
        assert_eq!(interleave("MXBOUZ", "dvqavd").unwrap(), "Md#v#q#a#v#d");
        // That this substitution is not cosmetic — `Xa` is absent from the
        // vocabulary while `#a` is present — is checked against the real
        // vocab.txt in tests/test_saprot_bridge.rs.
    }

    /// A length mismatch is an error, not a silent truncation — every residue
    /// after the gap would otherwise be paired with the wrong state.
    #[test]
    fn test_interleave_rejects_a_length_mismatch() {
        let err = interleave("MKT", "dv").unwrap_err().to_string();
        assert!(err.contains("3 residues against 2"), "got: {err}");
        assert!(interleave("", "").is_err(), "nothing to interleave");
    }

    /// Swapping the arguments is the easy mistake, and it is caught rather
    /// than producing a fluent-looking wrong answer.
    #[test]
    fn test_interleave_rejects_swapped_arguments() {
        let err = interleave("dvq", "MKT").unwrap_err().to_string();
        assert!(
            err.contains("not an amino acid") || err.contains("right way round"),
            "got: {err}"
        );
    }

    /// `#` is legitimate on either side: a residue whose structure could not be
    /// determined is exactly what the sequence-only form uses.
    #[test]
    fn test_unknown_state_passes_through_on_either_axis() {
        assert_eq!(interleave("MKT", "d#q").unwrap(), "MdK#Tq");
        assert_eq!(interleave("M#T", "dvq").unwrap(), "Md#vTq");
    }
}
