//! ProtT5's tokenizer, as a built-in table.
//!
//! # Why this is hand-written rather than loaded
//!
//! `Rostlab/prot_t5_xl_half_uniref50-enc` ships no `tokenizer.json`. It ships
//! `spiece.model`, a SentencePiece protobuf — which would mean either a
//! SentencePiece dependency or a conversion step, for a vocabulary that turns
//! out to contain **28 pieces**:
//!
//! ```text
//!  0 <pad>   1 </s>    2 <unk>
//!  3 ▁A   4 ▁L   5 ▁G   6 ▁V   7 ▁S   8 ▁R   9 ▁E  10 ▁D  11 ▁T  12 ▁I
//! 13 ▁P  14 ▁K  15 ▁F  16 ▁Q  17 ▁N  18 ▁Y  19 ▁M  20 ▁H  21 ▁W  22 ▁C
//! 23 ▁X  24 ▁B  25 ▁O  26 ▁U  27 ▁Z
//! ```
//!
//! (Read directly out of the published `spiece.model`, not inferred.) Every
//! amino-acid piece carries SentencePiece's `▁` word-boundary marker, and
//! nothing else in the file is reachable from a protein sequence, so the whole
//! tokenizer is a 28-entry character lookup.
//!
//! `config.json` says `vocab_size: 128` and `shared.weight` really is
//! `[128, 1024]`: ids 28..=127 are T5's 100 `<extra_id_*>` sentinels, which a
//! protein sequence never produces. The embedding table is larger than the
//! reachable alphabet, which is normal for a T5 checkpoint and not a sign that
//! this table is missing something.
//!
//! # Whitespace
//!
//! Because every piece is `▁<letter>`, the reference pipeline has to space the
//! residues out — HuggingFace's `T5Tokenizer` would read `"MKT"` as one word
//! and emit `<unk>`. Rostlab's own model card does exactly that:
//!
//! ```python
//! sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
//! ```
//!
//! Mapping characters directly makes the spacing irrelevant, so this tokenizer
//! **skips whitespace** and `"MKT"` and `"M K T"` produce identical ids. That
//! removes the failure the bead was most worried about: getting the convention
//! wrong shifts every embedding by one residue and produces plausible,
//! meaningless numbers rather than an error.
//!
//! The `[UZOB] -> X` substitution in that snippet is *not* applied here. Those
//! four residues have their own ids (24..=27) and the model has embeddings for
//! them; collapsing them is a convention of the model card, not of the
//! tokenizer, and doing it silently would change a caller's input behind their
//! back. Callers reproducing published ProtT5 embeddings should apply it
//! themselves — see [`replace_rare_residues`].

/// `<pad>` — id 0. Also T5's `decoder_start_token_id`.
pub const PAD_ID: u32 = 0;
/// `</s>` — id 1. Appended after the last residue; ProtT5 has no BOS.
pub const EOS_ID: u32 = 1;
/// `<unk>` — id 2. Any character outside the table below.
pub const UNK_ID: u32 = 2;

/// The reachable alphabet, in SentencePiece id order (id = index + 3).
///
/// Ordered by frequency in UniRef, which is how SentencePiece emitted it — so
/// this is *not* alphabetical and must not be "tidied".
const RESIDUES: [char; 25] = [
    'A', 'L', 'G', 'V', 'S', 'R', 'E', 'D', 'T', 'I', 'P', 'K', 'F', 'Q', 'N', 'Y', 'M', 'H', 'W',
    'C', 'X', 'B', 'O', 'U', 'Z',
];

/// Id of the first residue piece; everything below it is a special token.
const FIRST_RESIDUE_ID: u32 = 3;

/// Full vocabulary size declared by `config.json` and matched by
/// `shared.weight`. Larger than the reachable alphabet; see the module docs.
pub const VOCAB_SIZE: usize = 128;

/// The token id for one residue character, or [`UNK_ID`] if it has none.
///
/// Case-sensitive: `tokenizer_config.json` sets `do_lower_case: false`, and
/// lowercase input in the reference pipeline would reach `<unk>` too.
pub fn residue_id(residue: char) -> u32 {
    match RESIDUES.iter().position(|&c| c == residue) {
        Some(i) => FIRST_RESIDUE_ID + i as u32,
        None => UNK_ID,
    }
}

/// How many residues `sequence` encodes, ignoring whitespace.
///
/// [`PlmRunner::residue_count`][crate::plm_runner::PlmRunner::residue_count]
/// delegates here so `embed_residues` returns one row per residue whether the
/// caller spaced the sequence out or not.
pub fn residue_count(sequence: &str) -> usize {
    sequence.chars().filter(|c| !c.is_whitespace()).count()
}

/// Token ids for `sequence`, with the trailing `</s>` and no BOS.
///
/// Whitespace is skipped, so `"MKT"` and `"M K T"` are the same input.
pub fn encode(sequence: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = sequence
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(residue_id)
        .collect();
    ids.push(EOS_ID);
    ids
}

/// Rostlab's `[UZOB] -> X` convention, for callers reproducing published
/// ProtT5 embeddings.
///
/// Offered rather than applied: see the module docs. Selenocysteine (U) and
/// pyrrolysine (O) are real residues with their own learned embeddings, so
/// collapsing them loses information that the model actually has.
pub fn replace_rare_residues(sequence: &str) -> String {
    sequence
        .chars()
        .map(|c| match c {
            'U' | 'Z' | 'O' | 'B' => 'X',
            other => other,
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The ids that matter, checked against the published `spiece.model`.
    ///
    /// Frequency-ordered, not alphabetical: `A` is 3 and `C` is 22. Sorting
    /// this table would load every residue's embedding from the wrong row.
    #[test]
    fn test_ids_match_the_sentencepiece_vocabulary() {
        for (residue, expected) in [
            ('A', 3),
            ('L', 4),
            ('G', 5),
            ('V', 6),
            ('S', 7),
            ('R', 8),
            ('E', 9),
            ('D', 10),
            ('T', 11),
            ('I', 12),
            ('P', 13),
            ('K', 14),
            ('F', 15),
            ('Q', 16),
            ('N', 17),
            ('Y', 18),
            ('M', 19),
            ('H', 20),
            ('W', 21),
            ('C', 22),
            ('X', 23),
            ('B', 24),
            ('O', 25),
            ('U', 26),
            ('Z', 27),
        ] {
            assert_eq!(residue_id(residue), expected, "id for {residue}");
        }
    }

    /// The table covers every id from 3 to 27 exactly once.
    #[test]
    fn test_residue_table_is_dense_and_unique() {
        let mut ids: Vec<u32> = RESIDUES.iter().map(|&c| residue_id(c)).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), RESIDUES.len(), "duplicate residue in the table");
        assert_eq!(ids.first().copied(), Some(FIRST_RESIDUE_ID));
        assert_eq!(ids.last().copied(), Some(27));
    }

    /// The whole point of skipping whitespace: the model card's spaced form and
    /// a bare sequence must tokenize identically.
    #[test]
    fn test_spaced_and_bare_sequences_agree() {
        assert_eq!(encode("MKT"), encode("M K T"));
        assert_eq!(encode("MKT"), vec![19, 14, 11, EOS_ID]);
        assert_eq!(residue_count("M K T"), 3);
        assert_eq!(residue_count("MKT"), 3);
    }

    /// EOS is appended and nothing is prepended — ProtT5 has no BOS.
    #[test]
    fn test_encode_appends_eos_and_no_bos() {
        let ids = encode("AC");
        assert_eq!(ids.len(), 3, "2 residues + </s>");
        assert_eq!(ids[0], residue_id('A'));
        assert_eq!(*ids.last().unwrap(), EOS_ID);
        assert_ne!(ids[0], EOS_ID, "nothing is prepended");
    }

    /// Unknown characters become `<unk>` rather than silently shifting the
    /// sequence or panicking.
    #[test]
    fn test_unknown_characters_become_unk() {
        assert_eq!(residue_id('J'), UNK_ID);
        assert_eq!(residue_id('m'), UNK_ID, "the tokenizer is case-sensitive");
        assert_eq!(encode("AJC").len(), 4);
        assert_eq!(encode("AJC")[1], UNK_ID);
    }

    /// The rare-residue convention is available but not automatic.
    #[test]
    fn test_replace_rare_residues_is_opt_in() {
        assert_eq!(replace_rare_residues("MUZOB"), "MXXXX");
        assert_ne!(
            encode("MU"),
            encode(&replace_rare_residues("MU")),
            "encode must not apply the substitution on its own"
        );
        assert_eq!(residue_id('U'), 26, "U keeps its own embedding row");
    }
}
