//! One conformance suite every registered model must pass (ferritin-goh.2).
//!
//! Adding a model used to mean writing a bespoke test file, and what got
//! asserted varied by author. This suite is parameterized over
//! [`REGISTRY`][ferritin_plms::registry::REGISTRY], so a new model inherits
//! coverage by being registered rather than by someone remembering to write it.
//!
//! # The checks
//!
//! | # | check | catches |
//! |---|---|---|
//! | 1 | loads, and the loaded dims equal `card.metadata` | a card that lies about the model |
//! | 3 | `embed_residues(seq).dim(1) == seq.len()` | special-token misalignment (ferritin-100.7) |
//! | 4 | the same input twice is bitwise identical | run-to-run variance |
//! | 6 | no NaN or Inf anywhere in the embedding | a bad weight-key mapping |
//! | 7 | a `Verified` card's fixture really exists and matches the model | a registry parity claim that nothing backs |
//! | 8 | every family is populated and every card has a generator script | the registry drifting ahead of the fixtures |
//!
//! Checks 4 and 6 are the cheap ones that matter most. A transposed weight
//! matrix or a wrong VarBuilder prefix usually produces *correctly shaped*
//! garbage, which shape assertions sail straight past — NaN and run-to-run
//! variance surface it long before a human notices the embeddings are
//! meaningless. Check 3 is the one that must never be relaxed: it is the
//! contract downstream code depends on.
//!
//! # Not yet implemented
//!
//! - **Check 2, tokenizer round-trip.** `PlmRunner` exposes no `encode`/
//!   `decode`, and the four families tokenize through different types. It
//!   needs the trait widened the way ferritin-100.7 widened it for embeddings.
//! - **Check 5, batch equivalence.** There is no `embed_batch`; every runner is
//!   hardcoded to batch size 1. Blocked on ferritin-100.12.
//!
//! # Tiering
//!
//! Execution is tiered by `card.approx_bytes_f32` rather than by `#[ignore]`,
//! so the tier follows the model rather than a hand-maintained attribute.
//!
//! | tier | size | runs |
//! |---|---|---|
//! | PR | ≤ [`PR_TIER_MAX_BYTES`] | every PR |
//! | nightly | ≤ [`LOADABLE_IN_CI_MAX_BYTES`] | nightly, with `FERRITIN_HF_TESTS=1` |
//! | too large | above that | nowhere — see below |
//!
//! The third tier is not in the issue's design and was added because the
//! two-tier version does not survive contact with a CI runner: `esm2-t48-15b`
//! is ~60 GB at F32 and `esmc-6b` ~24 GB, against 16 GB on a standard
//! GitHub-hosted runner. A nightly that tries to load those does not report a
//! model problem, it reports an OOM — and an unfixable red nightly is what
//! ferritin-100.20 was about. So they are excluded by size, with the excluded
//! set pinned by a test so the exclusion stays deliberate.

mod support;

use anyhow::{Result, bail};
use candle_core::DType;
use ferritin_plms::plm_runner::PlmRunner;
use ferritin_plms::registry::{ModelCard, ParityStatus, REGISTRY, TokenizerSpec};
use ferritin_plms::{
    AmplifyModels, AmplifyRunner, ESM2Models, ESM2Runner, ESM3Models, ESM3Runner, ESMCModels,
    ESMCRunner, device,
};
use support::parity::{ParityFixture, fixture_path, hf_tests_enabled};

/// Models at or below this size run on every PR; larger ones are nightly-only.
///
/// 100 MB is chosen so the PR tier is a genuine smoke test that costs seconds,
/// not a weights download that dominates the run.
const PR_TIER_MAX_BYTES: u64 = 100 * 1024 * 1024;

/// Above this, no CI runner can load the model, so nothing tries.
///
/// A standard GitHub-hosted runner has 16 GB; 8 GB leaves room for the OS, the
/// test process, and the fact that `approx_bytes_f32` is approximate. Loading
/// at F16 roughly halves the requirement, so raising this is reasonable once a
/// tier actually loads reduced-precision — see ferritin-100.9 for what that
/// costs numerically.
const LOADABLE_IN_CI_MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Ubiquitin (76 aa) — long enough to exercise attention, short enough to be fast.
const SEQ: &str = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

/// The sequence to feed a given model.
///
/// Most models take a plain amino-acid string. SaProt takes (amino acid, 3Di
/// state) pairs, so feeding it [`SEQ`] would read "MQ", "IF", … as residues —
/// none of which are in its vocabulary, making every position `<unk>`. The
/// shape assertions would still pass, which is precisely the kind of
/// meaningless-but-correctly-shaped input this suite exists to catch.
///
/// `#` is SaProt's "structure unknown" state, so `M#Q#I#…` is the documented
/// way to run it sequence-only (ferritin-goh.3).
fn test_sequence(card: &ModelCard) -> String {
    match card.tokenizer {
        TokenizerSpec::HfVocabTxt => SEQ.chars().flat_map(|c| [c, '#']).collect(),
        _ => SEQ.to_string(),
    }
}

/// Build the runner for a registry id.
///
/// A `match` rather than anything clever: the point of the registry is that
/// adding a model is a reviewed act, and a model that has no arm here fails
/// loudly rather than being silently skipped.
fn runner_for(card: &ModelCard) -> Result<Box<dyn PlmRunner>> {
    let dev = device(false)?;
    Ok(match card.id {
        "esm2-t6-8m" => Box::new(ESM2Runner::from_pretrained(ESM2Models::T6_8M, dev)?),
        "esm1v-t33-650m-ur90s-1" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Esm1v1, dev)?),
        "esm1v-t33-650m-ur90s-2" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Esm1v2, dev)?),
        "esm1v-t33-650m-ur90s-3" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Esm1v3, dev)?),
        "esm1v-t33-650m-ur90s-4" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Esm1v4, dev)?),
        "esm1v-t33-650m-ur90s-5" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Esm1v5, dev)?),
        "esm1b-t33-650m-ur50s" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Esm1b, dev)?),
        "saprot-35m-af2" => Box::new(ESM2Runner::from_pretrained(ESM2Models::SaProt35M, dev)?),
        "saprot-650m-af2" => Box::new(ESM2Runner::from_pretrained(ESM2Models::SaProt650M, dev)?),
        "fastesm2-650" => Box::new(ESM2Runner::from_pretrained(ESM2Models::FastEsm2_650, dev)?),
        "pepmlm-650m" => Box::new(ESM2Runner::from_pretrained(ESM2Models::PepMlm650M, dev)?),
        "dplm-650m" => Box::new(ESM2Runner::from_pretrained(ESM2Models::Dplm650M, dev)?),
        "esm2-t12-35m" => Box::new(ESM2Runner::from_pretrained(ESM2Models::T12_35M, dev)?),
        "esm2-t30-150m" => Box::new(ESM2Runner::from_pretrained(ESM2Models::T30_150M, dev)?),
        "esm2-t33-650m" => Box::new(ESM2Runner::from_pretrained(ESM2Models::T33_650M, dev)?),
        "esm2-t36-3b" => Box::new(ESM2Runner::from_pretrained(ESM2Models::T36_3B, dev)?),
        "esm2-t48-15b" => Box::new(ESM2Runner::from_pretrained(ESM2Models::T48_15B, dev)?),
        "amplify-120m" => Box::new(AmplifyRunner::from_pretrained(AmplifyModels::AMP120M, dev)?),
        "amplify-350m" => Box::new(AmplifyRunner::from_pretrained(AmplifyModels::AMP350M, dev)?),
        "esmc-300m" => Box::new(ESMCRunner::from_pretrained(ESMCModels::ESMC300M, dev)?),
        "esmc-600m" => Box::new(ESMCRunner::from_pretrained(ESMCModels::ESMC600M, dev)?),
        "esmc-6b" => Box::new(ESMCRunner::from_pretrained(ESMCModels::ESMC6B, dev)?),
        "esm3-sm-open-v1" => Box::new(ESM3Runner::from_pretrained(ESM3Models::SmOpen, dev)?),
        other => bail!(
            "{other} is a loadable embedding model with no arm in runner_for; \
             add one so it inherits conformance coverage"
        ),
    })
}

/// Run every implemented check against one model.
fn conform(card: &ModelCard) -> Result<()> {
    let id = card.id;
    let runner = runner_for(card)?;

    // ── 1. loads, and the loaded dims equal the card ──
    let loaded = runner.metadata();
    assert_eq!(loaded.d_model, card.metadata.d_model, "{id}: d_model");
    assert_eq!(loaded.n_layers, card.metadata.n_layers, "{id}: n_layers");
    assert_eq!(loaded.vocab_size, card.metadata.vocab_size, "{id}: vocab");
    assert_eq!(
        loaded.max_positions, card.metadata.max_positions,
        "{id}: max_positions"
    );
    assert_eq!(
        runner.special_tokens(),
        card.specials,
        "{id}: special-token layout"
    );

    // ── 3. residue alignment — the contract downstream code depends on ──
    //
    // Note this uses runner.residue_count rather than seq.len(): SaProt reads
    // two characters per residue, and hardcoding the byte length here would
    // re-introduce the very assumption residue_count exists to break.
    let seq = test_sequence(card);
    let residues_expected = runner.residue_count(&seq);
    assert_eq!(
        residues_expected,
        SEQ.len(),
        "{id}: the test sequence should encode the same 76 residues for every model"
    );

    let residues = runner.embed_residues(&seq)?;
    assert_eq!(
        residues.dim(1)?,
        residues_expected,
        "{id}: embed_residues must return exactly one row per residue"
    );
    assert_eq!(residues.dim(2)?, card.metadata.d_model, "{id}: width");

    // The raw form keeps the special tokens, per the declared layout.
    let raw = runner.embed(&seq)?;
    assert_eq!(
        raw.dim(1)?,
        residues_expected + card.specials.total(),
        "{id}: embed must keep the special-token rows"
    );

    // ── 6. finite outputs ──
    let values = residues
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if let Some((i, bad)) = values.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        bail!("{id}: embedding value at flat index {i} is {bad}, not finite");
    }
    // All-zeros is not an error but is never right for a real checkpoint, and
    // is what a silently-unloaded model produces.
    assert!(
        values.iter().any(|v| *v != 0.0),
        "{id}: every embedding value is zero, which no loaded model produces"
    );

    // ── 4. determinism ──
    let again = runner
        .embed_residues(&seq)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(
        values, again,
        "{id}: two identical inputs produced different outputs"
    );

    // ── 7. parity claim is backed by a real fixture ──
    if let ParityStatus::Verified { fixture } = card.parity {
        let path = fixture_path(fixture);
        assert!(
            path.exists(),
            "{id}: card claims parity against {fixture}, which is not committed"
        );
        let loaded_fixture = ParityFixture::load(fixture, &device(false)?)?;
        assert_eq!(loaded_fixture.name(), fixture);
        // The full numerical comparison lives in the per-model parity tests;
        // this only enforces that the registry's claim is not empty.
    }

    println!("{id}: conformance OK");
    Ok(())
}

/// Models small enough that some tier will actually load them.
fn testable() -> impl Iterator<Item = &'static ModelCard> {
    REGISTRY.iter().filter(|c| {
        c.is_loadable() && c.is_embedding_model() && c.approx_bytes_f32 <= LOADABLE_IN_CI_MAX_BYTES
    })
}

fn cards_in_tier(nightly: bool) -> impl Iterator<Item = &'static ModelCard> {
    testable().filter(move |c| (c.approx_bytes_f32 > PR_TIER_MAX_BYTES) == nightly)
}

/// Models no CI runner can load, and why each is excluded.
fn too_large_for_ci() -> impl Iterator<Item = &'static ModelCard> {
    REGISTRY.iter().filter(|c| {
        c.is_loadable() && c.is_embedding_model() && c.approx_bytes_f32 > LOADABLE_IN_CI_MAX_BYTES
    })
}

/// Small models run on every PR.
#[test]
fn test_conformance_pr_tier() -> Result<()> {
    let mut ran = 0;
    for card in cards_in_tier(false) {
        conform(card)?;
        ran += 1;
    }
    assert!(
        ran > 0,
        "the PR tier is empty; every model is now above {PR_TIER_MAX_BYTES} bytes, \
         which means no conformance coverage runs on PRs at all"
    );
    Ok(())
}

/// Everything larger runs in the nightly job.
#[test]
fn test_conformance_nightly_tier() -> Result<()> {
    if !hf_tests_enabled() {
        let ids: Vec<&str> = cards_in_tier(true).map(|c| c.id).collect();
        eprintln!(
            "skipping the nightly conformance tier ({} models: {}); \
             set FERRITIN_HF_TESTS=1 to run",
            ids.len(),
            ids.join(", ")
        );
        return Ok(());
    }
    for card in cards_in_tier(true) {
        conform(card)?;
    }
    Ok(())
}

// ── Check 8: registry completeness ────────────────────────────────────────────

/// Every loadable embedding model has an arm in `runner_for`.
///
/// Checked without loading anything, so a model added to the registry fails
/// here immediately rather than only when someone runs the nightly.
#[test]
fn test_every_loadable_model_can_be_constructed() {
    const KNOWN: &[&str] = &[
        "esm2-t6-8m",
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
        "amplify-120m",
        "amplify-350m",
        "esmc-300m",
        "esmc-600m",
        "esmc-6b",
        "esm3-sm-open-v1",
    ];
    for card in REGISTRY
        .iter()
        .filter(|c| c.is_loadable() && c.is_embedding_model())
    {
        assert!(
            KNOWN.contains(&card.id),
            "{} has no arm in runner_for, so it would inherit no conformance \
             coverage; add one",
            card.id
        );
    }
}

/// Every card names a family that has a fixture-generation script, so a model
/// can in principle be given parity coverage.
///
/// A card with no way to generate a fixture can never move off
/// `ParityStatus::Unverified`, which would make that status permanent rather
/// than provisional.
#[test]
fn test_every_family_has_a_fixture_generator() {
    let scripts = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts")
        .canonicalize()
        .expect("scripts/ should exist");

    for card in REGISTRY {
        // ESMFold2 has no generator yet and is unsupported anyway; it would be
        // dishonest to imply a fixture could be produced for a port that does
        // not load (ferritin-100.17).
        if !card.is_loadable() {
            continue;
        }
        let script = scripts.join(format!("generate_{}_fixtures.py", card.family_str()));
        assert!(
            script.exists(),
            "{}: no {} to generate a parity fixture from",
            card.id,
            script.display()
        );
    }
}

/// The models excluded for size are exactly the ones that cannot fit.
///
/// Pinned so that excluding a model is a deliberate act rather than something
/// that happens quietly when someone edits `approx_bytes_f32`. Every model
/// here has NO conformance coverage anywhere.
#[test]
fn test_models_too_large_for_ci_are_the_known_set() {
    let mut excluded: Vec<&str> = too_large_for_ci().map(|c| c.id).collect();
    excluded.sort_unstable();
    assert_eq!(
        excluded,
        ["esm2-t36-3b", "esm2-t48-15b", "esmc-6b"],
        "the set of models too large to test in CI changed; that is a \
         deliberate act, and each one loses all conformance coverage"
    );
}

/// The PR tier must stay cheap, or it stops running.
#[test]
fn test_pr_tier_is_small() {
    for card in cards_in_tier(false) {
        assert!(
            card.approx_bytes_f32 <= PR_TIER_MAX_BYTES,
            "{}: {} bytes is above the PR tier ceiling",
            card.id,
            card.approx_bytes_f32
        );
    }
}

/// SaProt loads and reads two characters per residue (ferritin-goh.3).
///
/// The residue-count property is the one worth pinning: SaProt is the first
/// model where `sequence.len()` is not the residue count, and
/// `embed_residues` would silently return twice as many rows as residues if
/// `PlmRunner::residue_count` were not overridden.
#[test]
#[ignore = "downloads westlake-repl/SaProt_35M_AF2 (~130 MB)"]
fn test_saprot_reads_two_chars_per_residue() -> Result<()> {
    use ferritin_plms::registry::lookup;

    let card = lookup("saprot-35m-af2").expect("registered");
    let runner = runner_for(card)?;

    // Three residues, six characters: (amino acid, 3Di state) pairs.
    let seq = "MdAaLp";
    assert_eq!(
        runner.residue_count(seq),
        3,
        "SaProt reads two characters per residue"
    );

    let residues = runner.embed_residues(seq)?;
    assert_eq!(
        residues.dim(1)?,
        3,
        "embed_residues must return one row per residue, not per character"
    );
    assert_eq!(residues.dim(2)?, card.metadata.d_model);
    Ok(())
}

/// FastESM2-650 ships no contact head, and asking for contacts says so
/// (ferritin-goh.12).
///
/// The absence is the point. `ESM2::load` probes for
/// `esm.contact_head.regression.weight` and stores `Option<ESM2ContactHead>`;
/// this checkpoint is the second model to exercise the `None` arm after
/// SaProt-650M. Verified against the real weights: the header lists 570
/// tensors and none of them is a contact head. Without the probe the load
/// itself would fail, so this test covers the load and the graceful refusal
/// together rather than assuming either.
#[test]
#[ignore = "downloads Synthyra/FastESM2_650 (~2.6 GB)"]
fn test_fastesm2_has_no_contact_head() -> Result<()> {
    use ferritin_plms::registry::lookup;

    let card = lookup("fastesm2-650").expect("registered");
    let runner = ESM2Runner::from_pretrained(ESM2Models::FastEsm2_650, device(false)?)?;

    // It still embeds: the missing head costs contacts, not representations.
    let residues = runner.embed_residues(SEQ)?;
    assert_eq!(residues.dim(1)?, SEQ.len(), "one row per residue");
    assert_eq!(residues.dim(2)?, card.metadata.d_model);

    let err = runner
        .predict_contacts(SEQ)
        .map(|_| ())
        .expect_err("this checkpoint ships no contact head");
    assert!(
        err.to_string().contains("no contact head"),
        "the refusal should name the cause; got: {err}"
    );
    Ok(())
}

/// ESM-1v loads with learned absolute positions and reconstructs its input.
///
/// Shape checks alone would not catch a wrong position encoding — the tensors
/// would be the right size and full of plausible numbers. Masked-LM
/// reconstruction is the cheap signal that does catch it: if positions were
/// off, or rotary were applied on top of the learned table, the model's own
/// argmax would stop agreeing with the input (ferritin-goh.4).
#[test]
#[ignore = "downloads facebook/esm1v_t33_650M_UR90S_1 (~2.6 GB)"]
fn test_esm1v_absolute_positions_reconstruct_input() -> Result<()> {
    use ferritin_plms::ESM2Runner;

    use ferritin_plms::esm2::esm2::ESM2Output;

    let runner = ESM2Runner::from_pretrained(ESM2Models::Esm1v1, device(false)?)?;
    let output = runner.run_forward(SEQ)?;

    // Strip the BOS/EOS rows before decoding. decode_logits drops only those
    // positions whose argmax happens to be a special token, so decoding the
    // raw (L+2)-row output can return more characters than there are residues.
    let residue_logits = output.logits.narrow(1, 1, SEQ.len())?;
    let decoded = runner.decode_logits(ESM2Output {
        logits: residue_logits,
    })?;

    assert_eq!(
        decoded.len(),
        SEQ.len(),
        "reconstruction should be the same length as the input"
    );

    let agree = SEQ
        .chars()
        .zip(decoded.chars())
        .filter(|(a, b)| a == b)
        .count() as f32
        / SEQ.len() as f32;

    // The existing ESM-2 smoke test uses 0.7; a wrong position encoding
    // collapses this far below chance-corrected agreement.
    assert!(
        agree > 0.7,
        "ESM-1v reconstructed only {:.0}% of ubiquitin, which suggests the \
         absolute position encoding is wrong; got {decoded}",
        agree * 100.0
    );
    println!("esm1v-1 reconstruction agreement: {:.1}%", agree * 100.0);
    Ok(())
}
