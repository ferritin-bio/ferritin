//! The ProstT5 -> SaProt bridge, against SaProt's real vocabulary
//! (ferritin-goh.13).
//!
//! ProstT5 emits Foldseek 3Di; SaProt consumes `(residue, state)` pairs over a
//! 21x21 product alphabet. Joining them removes the Foldseek C++ dependency
//! from the structure-aware path — but only if every pair the join produces is
//! actually a token SaProt knows. A pair that merely *looks* right encodes to
//! `<unk>`, losing the structural state along with the residue, and nothing
//! about that is loud.
//!
//! These read the committed `fixtures/saprot_vocab.txt` — the published
//! `westlake-repl/SaProt_35M_AF2` file, 446 lines — so they need no download.

use ferritin_plms::esm2::saprot_tokenizer::{SaProtTokenizer, interleave};

fn tokenizer() -> SaProtTokenizer {
    SaProtTokenizer::from_vocab_txt(include_str!("fixtures/saprot_vocab.txt"))
        .expect("the committed SaProt vocab.txt should parse")
}

/// The fixture really is SaProt's alphabet: 441 = 21x21 pairs plus 5 specials.
#[test]
fn test_fixture_is_the_full_saprot_vocabulary() {
    let t = tokenizer();
    assert_eq!(t.len(), 446, "5 special tokens + 21*21 residue/state pairs");
}

/// Every pair the bridge can emit is a real token.
///
/// This is the check the whole bridge rests on. It is exhaustive rather than
/// sampled because the failure is per-pair and silent: one missing combination
/// would show up as a single `<unk>` in the middle of an otherwise sensible
/// sequence.
#[test]
fn test_every_pair_the_bridge_can_emit_is_a_real_token() {
    let t = tokenizer();
    let unk = t.token_to_id("<unk>").expect("<unk> is in the vocabulary");

    let residues = "ACDEFGHIKLMNPQRSTVWY#";
    let states = "acdefghiklmnpqrstvwy#";
    let mut checked = 0;
    for residue in residues.chars() {
        for state in states.chars() {
            let woven = interleave(&residue.to_string(), &state.to_string())
                .expect("the bridge should accept its own alphabet");
            assert_eq!(
                t.encode(&woven),
                vec![t.token_to_id(&woven).unwrap()],
                "{woven} should encode as itself"
            );
            assert_ne!(t.encode(&woven)[0], unk, "{woven} encoded as <unk>");
            checked += 1;
        }
    }
    assert_eq!(checked, 21 * 21);
}

/// The substitution the bridge makes is load-bearing, not cosmetic.
///
/// ProstT5 has embeddings for `X`, `B`, `O`, `U` and `Z`; SaProt has no pair
/// for any of them. Passing one through would lose the structural state too.
#[test]
fn test_non_standard_residue_pairs_are_absent_but_hash_pairs_are_not() {
    let t = tokenizer();
    let unk = t.token_to_id("<unk>").unwrap();

    for absent in ["Xa", "Xd", "Ba", "Oa", "Ua", "Za"] {
        assert_eq!(
            t.token_to_id(absent),
            None,
            "{absent} should not be in SaProt's vocabulary"
        );
        assert_eq!(t.encode(absent)[0], unk, "{absent} should encode as <unk>");
    }
    for present in ["#a", "#d", "M#", "##"] {
        assert!(
            t.token_to_id(present).is_some(),
            "{present} should be in SaProt's vocabulary"
        );
    }

    // So the bridge's output survives encoding where a naive join would not.
    let bridged = interleave("MXKZT", "davqd").unwrap();
    assert_eq!(bridged, "Md#aKv#qTd");
    assert!(
        !t.encode(&bridged).contains(&unk),
        "the bridge's output should contain no unknown tokens: {bridged}"
    );
}

/// End to end at the tokenizer level: a bridged sequence reads back as the
/// right number of residues, and round-trips.
#[test]
fn test_bridged_sequence_round_trips_through_saprot() {
    let t = tokenizer();
    let aa = "MQIFVKTLTGK";
    let threedi = "dvvvvcvvvvd"; // ProstT5's real output for this sequence
    let bridged = interleave(aa, threedi).unwrap();

    assert_eq!(bridged.len(), aa.len() * 2);
    assert_eq!(
        t.residue_count(&bridged),
        aa.len(),
        "SaProt reads two characters per residue"
    );
    assert_eq!(t.decode(&t.encode(&bridged)), bridged);
}
