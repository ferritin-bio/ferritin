//! T5-family loading, contract and parity tests: ProtT5 (ferritin-goh.5) and
//! Ankh (ferritin-goh.6).
//!
//! These download multi-gigabyte checkpoints, so they are `#[ignore]`d:
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_plm_prott5 -- --include-ignored
//! ```

mod support;

use anyhow::Result;
use ferritin_plms::plm_runner::{PlmRunner, SpecialTokenLayout};
use ferritin_plms::t5::tokenizer;
use ferritin_plms::{T5Models, T5Runner, device};
use support::parity::{ParityFixture, assert_embeddings_close};

/// Ubiquitin (76 aa).
const SEQ: &str = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

fn runner() -> Result<T5Runner> {
    T5Runner::from_pretrained(T5Models::ProtT5XlHalfUniref50Enc, device(false)?)
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

// ── Batching (ferritin-100.12 meeting ferritin-goh.5) ─────────────────────────

/// ProtT5 inherits the default `embed_batch`, which loops `embed` and
/// zero-pads. Two things make it worth checking here rather than assuming:
///
/// * ProtT5 is the only `EOS_ONLY` model, so this is the only runner where
///   `embed_residues_batch` strips from column 0 instead of skipping a BOS row.
/// * `residue_count` skips whitespace, so a spaced sequence is *longer as a
///   string* than a bare one of the same length. Padding driven by
///   `sequence.len()` rather than `residue_count` would put the two rows at
///   different widths — which is exactly the mistake the spacing convention
///   invites.
#[test]
#[ignore = "requires downloading Rostlab/prot_t5_xl_half_uniref50-enc (2.4 GB)"]
fn test_prott5_batch_rows_match_single_sequences() -> Result<()> {
    let runner = runner()?;
    // Same three residues written two ways, plus a longer sequence to force
    // padding on the first two.
    let seqs = ["MKT", "M K T", "MQIFVKTLTGK"];
    let counts: Vec<usize> = seqs.iter().map(|s| runner.residue_count(s)).collect();
    assert_eq!(counts, vec![3, 3, 11], "whitespace is not a residue");

    let batch = runner.embed_residues_batch(&seqs)?;
    assert_eq!(
        batch.dims(),
        &[3, 11, 1024],
        "padding must follow residue_count, not string length"
    );

    let f32v = |t: &candle_core::Tensor| -> Result<Vec<f32>> {
        Ok(t.flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?)
    };
    for (i, seq) in seqs.iter().enumerate() {
        let single = f32v(&runner.embed_residues(seq)?)?;
        let row = f32v(&batch.narrow(0, i, 1)?.narrow(1, 0, counts[i])?)?;
        assert_eq!(single, row, "batch row {i} disagrees with embed_residues()");

        // Everything past this sequence is the documented zero padding.
        if counts[i] < 11 {
            let tail = f32v(
                &batch
                    .narrow(0, i, 1)?
                    .narrow(1, counts[i], 11 - counts[i])?,
            )?;
            assert!(
                tail.iter().all(|&v| v == 0.0),
                "row {i} has non-zero padding"
            );
        }
    }

    // The spaced and bare forms are the same input, so they must batch to
    // identical rows.
    let bare = f32v(&batch.narrow(0, 0, 1)?)?;
    let spaced = f32v(&batch.narrow(0, 1, 1)?)?;
    assert_eq!(bare, spaced, "spacing changed a batched row");
    Ok(())
}

// ── Ankh ──────────────────────────────────────────────────────────────────────

/// Ankh against HuggingFace `T5EncoderModel` (ferritin-goh.6).
///
/// Two things this pins that ProtT5's test cannot:
///
/// * **The gated FFN.** Ankh sets `feed_forward_proj: "gated-gelu"`, so candle
///   loads a `T5DenseGatedActDense` rather than ProtT5's plain ReLU dense.
///   Nothing else in the crate exercises that path.
/// * **The shared tokenizer table.** Ankh ships its own `tokenizer.json` — a
///   Unigram vocabulary with no SentencePiece boundary marker — and the runner
///   deliberately does not use it, because its alphabet sits at the same ids as
///   ProtT5's. The fixture carries HuggingFace's own ids so that reuse is
///   checked rather than assumed; if the two ever diverged this fails on the
///   ids before it fails on the embeddings.
///
/// Ankh is published at F32 and loaded at F32, so unlike ProtT5 there is no
/// half-precision rounding in this comparison.
#[test]
#[ignore = "requires downloading ElnaggarLab/ankh-base (2.9 GB)"]
fn test_ankh_parity_against_huggingface() -> Result<()> {
    let device = device(false)?;
    let Some(fixture) = ParityFixture::load_or_skip("ankh_parity", &device)? else {
        return Ok(());
    };
    let runner = T5Runner::from_pretrained(T5Models::AnkhBase, device)?;
    assert_eq!(runner.special_tokens(), SpecialTokenLayout::EOS_ONLY);

    for (name, seq) in [
        ("ubiquitin_nterm", "MQIFVKTLTGK"),
        ("glycine_repeat", "GGGGGGG"),
        ("alt_charged", "KEKEKEK"),
        ("rare_residues", "MUZOBX"),
    ] {
        let expected_ids: Vec<u32> = fixture
            .tensor(&format!("{name}_input_ids"))?
            .to_dtype(candle_core::DType::U32)?
            .to_vec1()?;
        assert_eq!(
            tokenizer::encode(seq),
            expected_ids,
            "{name}: the shared t5::tokenizer table disagrees with Ankh's own tokenizer.json"
        );

        let rust = runner.embed_residues(seq)?.squeeze(0)?;
        let reference = fixture.tensor(&format!("{name}_embeddings"))?;
        assert_embeddings_close(&rust.to_dtype(candle_core::DType::F32)?, reference, 0.999)
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
    }
    Ok(())
}

// ── ProstT5 ───────────────────────────────────────────────────────────────────

/// ProstT5's AA->3Di translation against HuggingFace (ferritin-goh.6).
///
/// The output is discrete structural states, so this asserts **exact string
/// equality** — a tolerance would be meaningless, and a near-miss 3Di string is
/// simply a wrong structure.
///
/// The fixture also carries the reference *input* ids, checked first: ProstT5's
/// vocabulary is the shared `t5::tokenizer` table plus 20 lowercase 3Di states
/// derived by an offset, plus a direction prefix. Any of those going wrong
/// yields a fluent, wrong translation rather than an error, so the ids are
/// worth pinning separately from the output.
#[test]
#[ignore = "requires downloading Rostlab/ProstT5_fp16 (5.6 GB)"]
fn test_prostt5_translation_matches_huggingface() -> Result<()> {
    use ferritin_plms::t5::tokenizer::Direction;
    use ferritin_plms::{ProstT5Models, ProstT5Translator};

    let device = device(false)?;
    let Some(fixture) = ParityFixture::load_or_skip("prostt5_parity", &device)? else {
        return Ok(());
    };
    let translator = ProstT5Translator::from_pretrained(ProstT5Models::XlFp16, device)?;

    for (name, seq) in [
        ("ubiquitin_nterm", "MQIFVKTLTGK"),
        ("glycine_repeat", "GGGGGGG"),
        ("alt_charged", "KEKEKEK"),
    ] {
        let expected_input: Vec<u32> = fixture
            .tensor(&format!("{name}_input_ids"))?
            .to_dtype(candle_core::DType::U32)?
            .to_vec1()?;
        assert_eq!(
            tokenizer::encode_for_translation(seq, Direction::AaToFold),
            expected_input,
            "{name}: translation input ids disagree with the reference"
        );

        let expected: String = fixture
            .tensor(&format!("{name}_3di_ids"))?
            .to_dtype(candle_core::DType::U32)?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|id| {
                tokenizer::three_di_char(id)
                    .unwrap_or_else(|| panic!("{name}: reference token {id} is not a 3Di state"))
            })
            .collect();

        let got = translator.translate(seq, Direction::AaToFold)?;
        assert_eq!(got, expected, "{name}: 3Di translation of {seq}");
        assert_eq!(
            got.len(),
            seq.len(),
            "{name}: translation is length-preserving"
        );
    }
    Ok(())
}

/// The decoder's KV cache must be cleared between translations.
///
/// Unlike the encoder-only path, where `use_cache` is dead, every `decode` step
/// here appends to a per-layer cache. A stale one is *concatenated onto*, not
/// overwritten, so a second translation would silently attend to the first
/// sequence's keys and return a fluent, wrong answer. This is the tripwire.
#[test]
#[ignore = "requires downloading Rostlab/ProstT5_fp16 (5.6 GB)"]
fn test_repeated_translation_is_deterministic() -> Result<()> {
    use ferritin_plms::t5::tokenizer::Direction;
    use ferritin_plms::{ProstT5Models, ProstT5Translator};

    let translator = ProstT5Translator::from_pretrained(ProstT5Models::XlFp16, device(false)?)?;
    let seq = "MQIFVKTLTGK";
    let first = translator.translate(seq, Direction::AaToFold)?;
    // A different, shorter sequence in between: a leftover cache would change
    // both its length and the next result.
    let _ = translator.translate("GGGGGGG", Direction::AaToFold)?;
    let third = translator.translate(seq, Direction::AaToFold)?;
    assert_eq!(
        first, third,
        "repeated translation drifted — the decoder is carrying KV state across calls"
    );
    Ok(())
}
