//! ProtT5 loading and contract tests (ferritin-goh.5).
//!
//! These download a 2.4 GB checkpoint, so they are `#[ignore]`d:
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_plm_prott5 -- --include-ignored
//! ```

mod support;

use anyhow::Result;
use ferritin_plms::plm_runner::{PlmRunner, SpecialTokenLayout};
use ferritin_plms::prott5::tokenizer;
use ferritin_plms::{ProtT5Models, ProtT5Runner, device};
use support::parity::{ParityFixture, assert_embeddings_close};

/// Ubiquitin (76 aa).
const SEQ: &str = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

fn runner() -> Result<ProtT5Runner> {
    ProtT5Runner::from_pretrained(ProtT5Models::XlHalfUniref50Enc, device(false)?)
}

/// It loads at all, and the shapes are what the card declares.
#[test]
#[ignore = "requires downloading Rostlab/prot_t5_xl_half_uniref50-enc (2.4 GB)"]
fn test_prott5_loads_and_embeds() -> Result<()> {
    let runner = runner()?;
    assert_eq!(runner.special_tokens(), SpecialTokenLayout::EOS_ONLY);

    let raw = runner.embed(SEQ)?;
    assert_eq!(
        raw.dims(),
        &[1, SEQ.len() + 1, 1024],
        "embed() should return L + 1 rows for the trailing </s>"
    );

    let residues = runner.embed_residues(SEQ)?;
    assert_eq!(residues.dims(), &[1, SEQ.len(), 1024]);
    Ok(())
}

/// The model is behind a `Mutex` because `T5EncoderModel::forward` takes
/// `&mut self`. That is only sound because the *encoder* never caches
/// (`T5Attention::load` sets `use_cache: cfg.use_cache && decoder`, and the
/// encoder stack loads with `decoder = false`).
///
/// If a future candle version started carrying state across calls, the second
/// embedding would differ from the first — silently, since a stale KV cache
/// concatenates rather than errors. This test is the tripwire.
#[test]
#[ignore = "requires downloading Rostlab/prot_t5_xl_half_uniref50-enc (2.4 GB)"]
fn test_repeated_embed_is_deterministic() -> Result<()> {
    let runner = runner()?;
    let first: Vec<f32> = runner
        .embed(SEQ)?
        .flatten_all()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;
    // A different, shorter sequence in between — a stale cache would change
    // both its length and the next result.
    let _ = runner.embed("MKTAYIAK")?;
    let third: Vec<f32> = runner
        .embed(SEQ)?
        .flatten_all()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;
    assert_eq!(
        first, third,
        "repeated embed() drifted — the T5 encoder is carrying state across calls"
    );
    Ok(())
}

/// The whitespace convention, end to end: the model card spaces residues out,
/// and that must not change the embedding or the row count.
#[test]
#[ignore = "requires downloading Rostlab/prot_t5_xl_half_uniref50-enc (2.4 GB)"]
fn test_spaced_input_matches_bare_input() -> Result<()> {
    let runner = runner()?;
    let spaced: String = SEQ
        .chars()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    assert_eq!(runner.residue_count(&spaced), SEQ.len());
    let bare: Vec<f32> = runner
        .embed_residues(SEQ)?
        .flatten_all()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;
    let spaced: Vec<f32> = runner
        .embed_residues(&spaced)?
        .flatten_all()?
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;
    assert_eq!(
        bare, spaced,
        "spacing the residues out changed the embedding"
    );
    Ok(())
}

/// Needs no download: the tokenizer is a built-in table.
#[test]
fn test_tokenizer_needs_no_download() {
    assert_eq!(
        tokenizer::encode("MKT"),
        vec![19, 14, 11, tokenizer::EOS_ID]
    );
    assert_eq!(tokenizer::residue_count("M K T"), 3);
}

// ── Numerical parity ──────────────────────────────────────────────────────────

/// Rust output against HuggingFace `T5EncoderModel`.
///
/// The fixture is generated at **F32** while this runner loads the checkpoint
/// at its published **F16**, so the comparison is measuring the port *plus*
/// half-precision rounding. Cosine similarity per residue is the right metric
/// for that: it is insensitive to the uniform scale drift F16 accumulates over
/// 24 layers while still catching the failures that matter — a wrong id, a
/// misread special-token layout, a transposed weight — all of which move rows
/// far more than rounding does.
///
/// Measured worst-row cosine is 0.99996 across all four fixtures, so the 0.999
/// floor leaves ~25x headroom over F16 rounding while still being tight enough
/// to catch a real regression. A one-row misalignment scores far below it.
///
/// `tests/fixtures/prott5_parity.safetensors` also carries the reference token
/// ids, checked first: a tokenizer mismatch would otherwise show up only as an
/// embedding that is inexplicably wrong.
#[test]
#[ignore = "requires downloading Rostlab/prot_t5_xl_half_uniref50-enc (2.4 GB)"]
fn test_prott5_parity_against_huggingface() -> Result<()> {
    let device = device(false)?;
    let Some(fixture) = ParityFixture::load_or_skip("prott5_parity", &device)? else {
        return Ok(());
    };
    let runner = runner()?;

    for (name, seq) in [
        ("ubiquitin_nterm", "MQIFVKTLTGK"),
        ("glycine_repeat", "GGGGGGG"),
        ("alt_charged", "KEKEKEK"),
        ("rare_residues", "MUZOBX"),
    ] {
        // Tokenizer first: it is the cheaper failure to diagnose, and a
        // mismatch here explains any embedding mismatch below.
        let expected_ids: Vec<u32> = fixture
            .tensor(&format!("{name}_input_ids"))?
            .to_dtype(candle_core::DType::U32)?
            .to_vec1()?;
        assert_eq!(
            tokenizer::encode(seq),
            expected_ids,
            "{name}: token ids disagree with SentencePiece"
        );

        let rust = runner.embed_residues(seq)?.squeeze(0)?;
        let reference = fixture.tensor(&format!("{name}_embeddings"))?;
        assert_embeddings_close(&rust.to_dtype(candle_core::DType::F32)?, reference, 0.999)
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    }
    Ok(())
}
