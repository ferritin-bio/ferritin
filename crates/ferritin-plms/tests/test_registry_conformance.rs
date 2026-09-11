//! The registry must describe the models as they actually load (ferritin-goh.1).
//!
//! [`REGISTRY`] is only worth having if its rows are true. A declared `d_model`
//! that disagrees with the loaded weights is worse than no declaration at all,
//! because callers will size downstream layers from it.
//!
//! The unit tests in `registry.rs` check internal consistency — unique ids,
//! well-formed repos, plausible dimensions. These check the rows against real
//! checkpoints, so they download weights and are `#[ignore]`d:
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_registry_conformance -- --include-ignored
//! ```
//!
//! Models whose cards carry `unsupported` are skipped rather than asserted on:
//! their dimensions cannot be checked against weights that will not load, and
//! pretending otherwise would make this suite red for reasons it does not own.

use anyhow::Result;
use ferritin_plms::plm_runner::PlmRunner;
use ferritin_plms::registry::{Family, REGISTRY, TokenizerSpec, VocabAlphabet, lookup};
use ferritin_plms::{
    AmplifyModels, AmplifyRunner, ESM2Models, ESM2Runner, ESMCModels, ESMCRunner, T5Models,
    T5Runner, device,
};

/// Every enum variant resolves to a registry row, and the row's source is what
/// the enum hands the loader.
///
/// This is the delegation the issue asked for: the enums are thin lookups, so
/// a repo string exists in exactly one place.
#[test]
fn test_enums_delegate_to_registry() {
    for (variant, id, repo) in [
        (
            ESM2Models::T6_8M.registry_id(),
            "esm2-t6-8m",
            "facebook/esm2_t6_8M_UR50D",
        ),
        (
            ESM2Models::T33_650M.registry_id(),
            "esm2-t33-650m",
            "facebook/esm2_t33_650M_UR50D",
        ),
    ] {
        assert_eq!(variant, id);
        assert_eq!(lookup(id).unwrap().source.repo_id, repo);
    }

    // The enum's own accessor must return the registry's source, not a copy.
    let (source, _config) = ESM2Models::T6_8M.model_info();
    assert_eq!(source.repo_id, lookup("esm2-t6-8m").unwrap().source.repo_id);

    let source = AmplifyModels::AMP120M.model_info();
    assert_eq!(
        source.repo_id,
        lookup("amplify-120m").unwrap().source.repo_id
    );
}

/// Every `vocab.txt` row states how a sequence is read from that file.
///
/// `TokenizerSpec::HfVocabTxt` used to be a bare variant that both the ESM-2
/// runner and the conformance suite treated as "SaProt": two characters per
/// residue over a 20x20 (amino acid, 3Di) alphabet. A second `vocab.txt`
/// model over a one-character alphabet would have been fed `M#Q#I#…`, made
/// every position `<unk>`, and still satisfied every shape assertion
/// (ferritin-goh.11).
///
/// Carrying [`VocabAlphabet`] in the variant makes that a compile error rather
/// than a silent wrong answer; this test pins the alphabet each current row
/// actually declares, so a row cannot be flipped to the wrong one by accident.
#[test]
fn test_vocab_txt_rows_declare_their_alphabet() {
    let vocab_txt_rows: Vec<_> = REGISTRY
        .iter()
        .filter_map(|card| match card.tokenizer {
            TokenizerSpec::HfVocabTxt(alphabet) => Some((card.id, alphabet)),
            _ => None,
        })
        .collect();

    assert!(
        !vocab_txt_rows.is_empty(),
        "no vocab.txt rows left in the registry; delete this test or the variant"
    );

    for (id, alphabet) in vocab_txt_rows {
        let expected = if id.starts_with("saprot") {
            VocabAlphabet::SaProtPairs
        } else {
            VocabAlphabet::SingleResidue
        };
        assert_eq!(
            alphabet, expected,
            "{id}: vocab.txt alphabet is {alphabet:?}, expected {expected:?}. \
             SaProt reads (amino acid, 3Di) pairs; nothing else in the registry does."
        );
    }
}

/// Loading ESM2-8M must produce exactly the dimensions its card declares.
#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_esm2_loaded_dims_match_card() -> Result<()> {
    let card = lookup("esm2-t6-8m").expect("registered");
    let runner = ESM2Runner::from_pretrained(ESM2Models::T6_8M, device(false)?)?;
    assert_card_matches(card.id, &runner)
}

#[test]
#[ignore = "requires downloading chandar-lab/AMPLIFY_120M weights"]
fn test_amplify_loaded_dims_match_card() -> Result<()> {
    let card = lookup("amplify-120m").expect("registered");
    let runner = AmplifyRunner::from_pretrained(AmplifyModels::AMP120M, device(false)?)?;
    assert_card_matches(card.id, &runner)
}

#[test]
#[ignore = "requires downloading EvolutionaryScale/esmc-300m-2024-12 weights"]
fn test_esmc_loaded_dims_match_card() -> Result<()> {
    let card = lookup("esmc-300m").expect("registered");
    let runner = ESMCRunner::from_pretrained(ESMCModels::ESMC300M, device(false)?)?;
    assert_card_matches(card.id, &runner)
}

/// ProtT5 is the one row whose `specials` is not `BOS_EOS`, so this check —
/// which compares the card's layout against the loaded runner's — is doing
/// more work here than for the ESM-family rows (ferritin-goh.5).
/// Ankh shares ProtT5's runner but not its FFN — `feed_forward_proj` is
/// `"gated-gelu"`, so this also checks that the gated path loads against a real
/// checkpoint rather than only parsing (ferritin-goh.6).
#[test]
#[ignore = "requires downloading ElnaggarLab/ankh-base (2.9 GB)"]
fn test_ankh_base_loaded_dims_match_card() -> Result<()> {
    let card = lookup("ankh-base").expect("registered");
    let runner = T5Runner::from_pretrained(T5Models::AnkhBase, device(false)?)?;
    assert_card_matches(card.id, &runner)
}

#[test]
#[ignore = "requires downloading Rostlab/prot_t5_xl_half_uniref50-enc (2.4 GB)"]
fn test_prott5_loaded_dims_match_card() -> Result<()> {
    let card = lookup("prott5-xl-half-uniref50-enc").expect("registered");
    let runner = T5Runner::from_pretrained(T5Models::ProtT5XlHalfUniref50Enc, device(false)?)?;
    assert_card_matches(card.id, &runner)
}

/// Compare a loaded runner's self-reported metadata and token layout against
/// its registry row.
fn assert_card_matches(id: &str, runner: &dyn PlmRunner) -> Result<()> {
    let card = lookup(id).expect("registered");
    let loaded = runner.metadata();

    assert_eq!(
        loaded.d_model, card.metadata.d_model,
        "{id}: card says d_model {} but the loaded model reports {}",
        card.metadata.d_model, loaded.d_model
    );
    assert_eq!(
        loaded.n_layers, card.metadata.n_layers,
        "{id}: card says n_layers {} but the loaded model reports {}",
        card.metadata.n_layers, loaded.n_layers
    );
    assert_eq!(
        loaded.vocab_size, card.metadata.vocab_size,
        "{id}: card says vocab_size {} but the loaded model reports {}",
        card.metadata.vocab_size, loaded.vocab_size
    );
    assert_eq!(
        loaded.max_positions, card.metadata.max_positions,
        "{id}: card and loaded model disagree on max_positions"
    );
    assert_eq!(
        runner.special_tokens(),
        card.specials,
        "{id}: card and runner disagree on the special-token layout"
    );

    // The declared width must also be the width actually returned.
    let embedded = runner.embed_residues("MQIFVKTLTGK")?;
    assert_eq!(
        embedded.dim(2)?,
        card.metadata.d_model,
        "{id}: embeddings are not d_model wide"
    );
    Ok(())
}

/// Every loadable embedding model has a card that can be checked this way.
///
/// Guards against a new row being added without conformance coverage: if this
/// list grows, a matching `#[ignore]`d test above should grow with it.
#[test]
fn test_loadable_embedding_models_are_covered() {
    let mut covered: Vec<&str> = REGISTRY
        .iter()
        .filter(|c| c.is_loadable() && c.is_embedding_model())
        .map(|c| c.id)
        .collect();
    covered.sort_unstable();

    assert_eq!(
        covered,
        [
            "amplify-120m",
            "amplify-350m",
            "ankh-base",
            "ankh-large",
            "dplm-650m",
            "esm1b-t33-650m-ur50s",
            "esm1v-t33-650m-ur90s-1",
            "esm1v-t33-650m-ur90s-2",
            "esm1v-t33-650m-ur90s-3",
            "esm1v-t33-650m-ur90s-4",
            "esm1v-t33-650m-ur90s-5",
            "esm2-t12-35m",
            "esm2-t30-150m",
            "esm2-t33-650m",
            "esm2-t36-3b",
            "esm2-t48-15b",
            "esm2-t6-8m",
            "esm3-sm-open-v1",
            "esmc-300m",
            "esmc-600m",
            "esmc-6b",
            "fastesm2-650",
            "pepmlm-650m",
            "prott5-xl-half-uniref50-enc",
            "saprot-35m-af2",
            "saprot-650m-af2",
        ],
        "the set of loadable embedding models changed; add or remove a \
         conformance test to match"
    );
}

/// Which loadable models have no conformance test above, and why.
///
/// Kept as an explicit list rather than a silent gap. Every entry here is a
/// model whose card is unchecked against real weights, so a wrong dimension
/// would not be caught.
#[test]
fn test_uncovered_loadable_models_are_accounted_for() {
    let covered = [
        "esm2-t6-8m",
        "amplify-120m",
        "esmc-300m",
        "prott5-xl-half-uniref50-enc",
        "ankh-base",
    ];

    let uncovered: Vec<&str> = REGISTRY
        .iter()
        .filter(|c| c.is_loadable() && c.is_embedding_model())
        .map(|c| c.id)
        .filter(|id| !covered.contains(id))
        .collect();

    // The larger ESM2 variants, AMPLIFY-350M, ESMC-600M and ESM3 are simply
    // big downloads. ESMC-6B is different in kind: at ~24 GB F32 / ~12 GB F16
    // it cannot be loaded on a modest machine at all, so its card was verified
    // against the published shard index instead — see
    // test_esmc_6b_index_contains_every_path_the_loader_requests.
    assert_eq!(
        uncovered,
        [
            "esm2-t12-35m",
            "esm2-t30-150m",
            "esm2-t33-650m",
            "esm2-t36-3b",
            "esm2-t48-15b",
            "esm1v-t33-650m-ur90s-1",
            "esm1v-t33-650m-ur90s-2",
            "esm1v-t33-650m-ur90s-3",
            "esm1v-t33-650m-ur90s-4",
            "esm1v-t33-650m-ur90s-5",
            "esm1b-t33-650m-ur50s",
            "saprot-35m-af2",
            "saprot-650m-af2",
            "fastesm2-650",
            "pepmlm-650m",
            "dplm-650m",
            "amplify-350m",
            "esmc-600m",
            "esmc-6b",
            "esm3-sm-open-v1",
            // ankh-large is a 7.5 GB download for the same architecture
            // ankh-base already covers; only its dimensions differ.
            "ankh-large",
        ],
        "a model gained or lost conformance coverage; say which and why"
    );
}

/// Every row that stays outside `PlmRunner`, and why. Recorded rather than
/// implied, because no conformance test above applies to them.
///
/// There are now two distinct reasons a model is not an embedding model, and
/// conflating them would be a mistake: the structure models *consume* structure
/// instead of sequence, while ProstT5 *emits* generated tokens instead of an
/// embedding (ferritin-goh.6). Neither fits a trait about per-residue vectors.
#[test]
fn test_non_embedding_models_are_accounted_for() {
    for (id, reason) in [
        ("proteinmpnn-v48-020", "consumes backbone coordinates"),
        (
            "ligandmpnn-v32-020-25",
            "consumes backbone coordinates and ligand atoms",
        ),
        ("esm3-structure-encoder-v0", "consumes backbone coordinates"),
        (
            "prostt5-fp16",
            "emits generated tokens, driven by ProstT5Translator",
        ),
    ] {
        let card = lookup(id).expect("registered");
        assert!(
            !card.is_embedding_model(),
            "{id} should not be treated as an embedding model ({reason})"
        );
    }
    assert_eq!(lookup("proteinmpnn-v48-020").unwrap().family, Family::Mpnn);

    // And the converse: nothing else has quietly fallen out of PlmRunner.
    let outside: Vec<&str> = REGISTRY
        .iter()
        .filter(|c| !c.is_embedding_model())
        .map(|c| c.id)
        .collect();
    assert_eq!(
        outside,
        [
            "esm3-structure-encoder-v0",
            "prostt5-fp16",
            "proteinmpnn-v48-020",
            "ligandmpnn-v32-020-25",
        ],
        "the set of non-embedding models changed; say which and why"
    );
}
