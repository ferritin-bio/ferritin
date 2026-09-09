//! The T5 family's tokenizer, as a built-in table.
//!
//! Shared by ProtT5, Ankh and ProstT5. The first two arrive in different
//! containers — ProtT5 in
//! a SentencePiece `spiece.model`, Ankh in a `tokenizer.json` Unigram model —
//! but the alphabet underneath is the *same letters at the same ids*, so one
//! table serves both. Ankh's pieces carry no word-boundary marker, which is
//! invisible here because this maps characters directly.
//!
//! ProstT5 extends the same vocabulary rather than replacing it: Foldseek's
//! 3Di structural alphabet occupies ids 128..=147 as *lowercase* letters, each
//! exactly [`THREE_DI_OFFSET`] above its uppercase residue. Case is therefore
//! **meaning, not formatting** — `M` is methionine and `m` is a structural
//! state — which is why [`vocab_id`] never folds it.
//!
//! That equivalence is verified rather than assumed: both parity fixtures
//! carry HuggingFace's own token ids (see `scripts/generate_t5_fixtures.py`),
//! and the Rust tests compare against them before comparing embeddings, so a
//! divergence fails on the ids rather than showing up as mysteriously wrong
//! numbers.
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

// ── ProstT5: the 3Di structural alphabet (ferritin-goh.6) ────────────────────

/// How far a 3Di token sits above its amino-acid counterpart.
///
/// ProstT5 extends this vocabulary with Foldseek's 3Di alphabet as *lowercase*
/// letters at ids 128..=147, and every one of them is exactly its uppercase
/// residue's id plus this offset — `a` = 128 = `A`(3) + 125, through
/// `c` = 147 = `C`(22) + 125. So there is no second table: 3Di is this table
/// shifted.
///
/// `test_three_di_is_the_residue_table_shifted` checks that against the ids
/// published in ProstT5's `added_tokens.json`, so the shortcut is verified
/// rather than assumed.
pub const THREE_DI_OFFSET: u32 = 125;

/// The 20 residues that have a 3Di counterpart — the standard amino acids.
///
/// `X`, `B`, `O`, `U` and `Z` are in the residue table but have no 3Di state:
/// Foldseek's alphabet is exactly 20 letters.
const THREE_DI_RESIDUES: usize = 20;

/// `<AA2fold>` — prefixed to an amino-acid sequence to translate it *to* 3Di.
pub const AA_TO_FOLD_ID: u32 = 149;
/// `<fold2AA>` — prefixed to a 3Di sequence to translate it *to* amino acids.
pub const FOLD_TO_AA_ID: u32 = 148;

/// Which way a ProstT5 translation runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Amino acids in, 3Di structural states out.
    AaToFold,
    /// 3Di structural states in, amino acids out.
    FoldToAa,
}

impl Direction {
    /// The token that prefixes the input for this direction.
    pub const fn prefix_id(&self) -> u32 {
        match self {
            Self::AaToFold => AA_TO_FOLD_ID,
            Self::FoldToAa => FOLD_TO_AA_ID,
        }
    }
}

/// The token id for one 3Di state character (lowercase), or [`UNK_ID`].
pub fn three_di_id(state: char) -> u32 {
    if !state.is_ascii_lowercase() {
        return UNK_ID;
    }
    match RESIDUES[..THREE_DI_RESIDUES]
        .iter()
        .position(|&c| c == state.to_ascii_uppercase())
    {
        Some(i) => FIRST_RESIDUE_ID + i as u32 + THREE_DI_OFFSET,
        None => UNK_ID,
    }
}

/// The 3Di state character for a token id, or `None` if the id is not one.
///
/// Used to decode a generated sequence; ids outside 128..=147 (specials, and
/// the amino-acid range) return `None` so a caller can tell "not a 3Di state"
/// from "some state".
pub fn three_di_char(id: u32) -> Option<char> {
    let base = id.checked_sub(THREE_DI_OFFSET)?;
    let index = base.checked_sub(FIRST_RESIDUE_ID)? as usize;
    RESIDUES
        .get(..THREE_DI_RESIDUES)?
        .get(index)
        .map(|c| c.to_ascii_lowercase())
}

/// The amino-acid character for a token id, or `None`.
///
/// The reverse direction's decoder: ids 3..=27 map back to residues.
pub fn residue_char(id: u32) -> Option<char> {
    let index = id.checked_sub(FIRST_RESIDUE_ID)? as usize;
    RESIDUES.get(index).copied()
}

/// The token id for any character in the shared vocabulary.
///
/// **Case is meaning here, not formatting.** ProstT5 puts amino acids and 3Di
/// states in one vocabulary and distinguishes them by case: `M` is methionine
/// (19) and `m` is a structural state (144). So this maps uppercase through
/// [`residue_id`] and lowercase through [`three_di_id`], exactly as
/// HuggingFace's tokenizer does — and deliberately does *not* case-fold.
///
/// Folding case would be an unforced deviation from the reference that changes
/// the ids: lowercasing an amino-acid sequence would silently reinterpret it as
/// a structural one, and the model would translate it fluently and wrongly.
pub fn vocab_id(c: char) -> u32 {
    if c.is_ascii_lowercase() {
        three_di_id(c)
    } else {
        residue_id(c)
    }
}

/// Token ids for a ProstT5 translation input: the direction token, the
/// sequence, then `</s>`.
///
/// The direction prefix is not decoration — ProstT5 is trained with it, and
/// omitting it or using the wrong one produces a fluent, wrong translation
/// rather than an error. The prefix is the *only* thing `direction` controls:
/// the characters themselves are mapped by [`vocab_id`], which reads case as
/// meaning. Whitespace is skipped, as everywhere else here.
pub fn encode_for_translation(sequence: &str, direction: Direction) -> Vec<u32> {
    let mut ids = Vec::with_capacity(residue_count(sequence) + 2);
    ids.push(direction.prefix_id());
    ids.extend(
        sequence
            .chars()
            .filter(|c| !c.is_whitespace())
            .map(vocab_id),
    );
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

    // ── ProstT5 3Di ───────────────────────────────────────────────────────

    /// The 3Di ids, checked against ProstT5's published `added_tokens.json`.
    ///
    /// This is the assertion the `THREE_DI_OFFSET` shortcut rests on: if the
    /// two alphabets were ordered differently, deriving one from the other
    /// would map every structural state to the wrong embedding row while
    /// looking perfectly reasonable.
    #[test]
    fn test_three_di_is_the_residue_table_shifted() {
        // Verbatim from https://huggingface.co/Rostlab/ProstT5 added_tokens.json.
        for (state, expected) in [
            ('a', 128),
            ('l', 129),
            ('g', 130),
            ('v', 131),
            ('s', 132),
            ('r', 133),
            ('e', 134),
            ('d', 135),
            ('t', 136),
            ('i', 137),
            ('p', 138),
            ('k', 139),
            ('f', 140),
            ('q', 141),
            ('n', 142),
            ('y', 143),
            ('m', 144),
            ('h', 145),
            ('w', 146),
            ('c', 147),
        ] {
            assert_eq!(three_di_id(state), expected, "3Di id for {state}");
            assert_eq!(
                three_di_id(state),
                residue_id(state.to_ascii_uppercase()) + THREE_DI_OFFSET,
                "{state} should be its uppercase id + {THREE_DI_OFFSET}"
            );
        }
    }

    /// Foldseek's alphabet is exactly 20 letters. ProtT5's table also carries
    /// X/B/O/U/Z, which have no structural counterpart and must not be
    /// silently given one — `x` would otherwise land on id 148, which is
    /// `<fold2AA>`.
    #[test]
    fn test_rare_residues_have_no_3di_state() {
        for state in ['x', 'b', 'o', 'u', 'z'] {
            assert_eq!(three_di_id(state), UNK_ID, "{state} is not a 3Di state");
        }
        assert_eq!(three_di_id('j'), UNK_ID);
        assert_eq!(three_di_id('A'), UNK_ID, "3Di states are lowercase");
    }

    /// Decoding is the inverse of encoding across the whole alphabet, and the
    /// two ranges do not overlap.
    #[test]
    fn test_three_di_round_trips_and_stays_disjoint_from_residues() {
        for state in "algvsredtipkfqnymhwc".chars() {
            let id = three_di_id(state);
            assert_eq!(three_di_char(id), Some(state), "round trip for {state}");
            assert_eq!(
                residue_char(id),
                None,
                "{state}'s id {id} must not also read as an amino acid"
            );
        }
        for residue in RESIDUES {
            let id = residue_id(residue);
            assert_eq!(residue_char(id), Some(residue));
            assert_eq!(three_di_char(id), None, "{residue} is not a 3Di state");
        }
        for special in [PAD_ID, EOS_ID, UNK_ID, AA_TO_FOLD_ID, FOLD_TO_AA_ID] {
            assert_eq!(three_di_char(special), None, "special {special}");
        }
    }

    /// The direction token leads, and the two directions read opposite cases.
    ///
    /// Getting the prefix wrong does not error — ProstT5 produces a fluent
    /// translation in the wrong direction — so this is the check that the
    /// prefix is present and correct at all.
    #[test]
    fn test_translation_input_carries_the_direction_prefix() {
        let aa = encode_for_translation("MKT", Direction::AaToFold);
        assert_eq!(aa[0], AA_TO_FOLD_ID, "AA->3Di must lead with <AA2fold>");
        assert_eq!(&aa[1..4], &[19, 14, 11], "the residues themselves");
        assert_eq!(*aa.last().unwrap(), EOS_ID);

        let fold = encode_for_translation("dvq", Direction::FoldToAa);
        assert_eq!(fold[0], FOLD_TO_AA_ID, "3Di->AA must lead with <fold2AA>");
        assert_eq!(
            &fold[1..4],
            &[three_di_id('d'), three_di_id('v'), three_di_id('q')]
        );
        assert_eq!(*fold.last().unwrap(), EOS_ID);
        assert_eq!(aa.len(), 5, "3 residues + prefix + EOS");
    }

    /// Case is meaning, not formatting: `M` and `m` are different tokens, and
    /// the encoder must not fold them together.
    ///
    /// This is the deviation that is easy to introduce by being "helpful".
    /// Lowercasing an amino-acid sequence does not normalise it — it
    /// reinterprets it as a structural one, and every id changes.
    #[test]
    fn test_case_selects_the_alphabet_and_is_never_folded() {
        assert_eq!(vocab_id('M'), 19, "uppercase M is methionine");
        assert_eq!(vocab_id('m'), 144, "lowercase m is a 3Di state");
        assert_ne!(vocab_id('M'), vocab_id('m'));

        // The direction token is the only thing `direction` changes.
        let a = encode_for_translation("MKT", Direction::AaToFold);
        let b = encode_for_translation("MKT", Direction::FoldToAa);
        assert_eq!(a[1..], b[1..], "only the prefix should differ");
        assert_ne!(a[0], b[0]);

        // A character with no counterpart in its own case is <unk>, not a
        // silent reinterpretation: `x` has no 3Di state.
        assert_eq!(vocab_id('x'), UNK_ID);
        assert_eq!(vocab_id('X'), 23, "uppercase X is a real residue");
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
