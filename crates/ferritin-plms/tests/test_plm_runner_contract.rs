//! Conformance tests for the [`PlmRunner`] special-token contract (ferritin-100.7).
//!
//! Each runner *declares* how many special tokens its tokenizer wraps a
//! sequence in. That declaration is only useful if it is true, so these tests
//! check it against the real models: `embed` must return
//! `L + leading + trailing` rows and `embed_residues` exactly `L`.
//!
//! A runner whose `SpecialTokenLayout` drifts from its tokenizer fails here
//! rather than silently handing callers misaligned residues — which is the
//! failure mode the contract exists to prevent.
//!
//! These download weights, so they are `#[ignore]`d:
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_plm_runner_contract -- --include-ignored
//! ```

use anyhow::Result;
use ferritin_plms::plm_runner::{PlmRunner, SpecialTokenLayout};
use ferritin_plms::{AmplifyModels, AmplifyRunner, ESM2Models, ESM2Runner, device};

/// Ubiquitin (76 aa).
const SEQ: &str = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

/// Assert that a runner's declared layout matches what its tokenizer actually does.
fn assert_contract_holds(runner: &dyn PlmRunner) -> Result<()> {
    let name = runner.model_name();
    let layout = runner.special_tokens();

    let raw = runner.embed(SEQ)?;
    assert_eq!(
        raw.dim(1)?,
        SEQ.len() + layout.total(),
        "{name}: embed() rows should be L + {} specials",
        layout.total()
    );

    // The contract: one row per residue, whatever the tokenizer does.
    let residues = runner.embed_residues(SEQ)?;
    assert_eq!(
        residues.dim(1)?,
        SEQ.len(),
        "{name}: embed_residues() must return exactly one row per residue"
    );

    // Width is unchanged by stripping, and matches the declared metadata.
    let metadata = runner.metadata();
    assert_eq!(
        residues.dim(2)?,
        metadata.d_model,
        "{name}: embedding width should match ModelMetadata::d_model"
    );
    assert_eq!(raw.dim(0)?, 1, "{name}: batch dimension should be 1");

    Ok(())
}

#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_esm2_special_token_contract() -> Result<()> {
    let runner = ESM2Runner::load_model(ESM2Models::T6_8M, device(false)?)?;
    assert_eq!(runner.special_tokens(), SpecialTokenLayout::BOS_EOS);
    assert_contract_holds(&runner)
}

#[test]
#[ignore = "requires downloading chandar-lab/AMPLIFY_120M weights"]
fn test_amplify_special_token_contract() -> Result<()> {
    let runner = AmplifyRunner::load_model(AmplifyModels::AMP120M, device(false)?)?;
    assert_eq!(runner.special_tokens(), SpecialTokenLayout::BOS_EOS);
    assert_contract_holds(&runner)
}

/// The point of the contract: rows from two different models line up, so a
/// caller can compare residue `i` to residue `i` without knowing either
/// tokenizer.
#[test]
#[ignore = "requires downloading both ESM2 and AMPLIFY weights"]
fn test_embed_residues_aligns_across_models() -> Result<()> {
    let esm2 = ESM2Runner::load_model(ESM2Models::T6_8M, device(false)?)?;
    let amplify = AmplifyRunner::load_model(AmplifyModels::AMP120M, device(false)?)?;

    let a = esm2.embed_residues(SEQ)?;
    let b = amplify.embed_residues(SEQ)?;

    assert_eq!(
        a.dim(1)?,
        b.dim(1)?,
        "residue counts must agree across runners"
    );
    assert_eq!(a.dim(1)?, SEQ.len());
    Ok(())
}

/// `logits` is aligned with `embed`, not `embed_residues`, and its last
/// dimension is the declared vocabulary size.
#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_logits_shape_matches_metadata() -> Result<()> {
    let runner = ESM2Runner::load_model(ESM2Models::T6_8M, device(false)?)?;
    let logits = runner.logits(SEQ)?;
    let metadata = runner.metadata();

    assert_eq!(
        logits.dim(1)?,
        SEQ.len() + runner.special_tokens().total(),
        "logits should be aligned with embed(), specials included"
    );
    assert_eq!(
        logits.dim(2)?,
        metadata.vocab_size,
        "logits width should match ModelMetadata::vocab_size"
    );
    Ok(())
}

/// Metadata should describe the model, not placeholder values.
#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_metadata_is_populated() -> Result<()> {
    let runner = ESM2Runner::load_model(ESM2Models::T6_8M, device(false)?)?;
    let metadata = runner.metadata();

    // ESM2 t6_8M: 6 layers, width 320, 33-token vocabulary.
    assert_eq!(metadata.n_layers, 6);
    assert_eq!(metadata.d_model, 320);
    assert_eq!(metadata.vocab_size, 33);
    assert!(
        metadata.max_positions.is_some_and(|m| m > SEQ.len()),
        "ESM2 has learned position embeddings, so max_positions should be reported"
    );
    Ok(())
}
