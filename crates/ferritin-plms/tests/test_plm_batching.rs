//! Batched inference: `embed_batch` must agree with `embed`, row for row
//! (ferritin-100.12).
//!
//! The whole point of a batched path is that it produces the *same numbers* as
//! the single-sequence path, and that equality is easy to break in a way that
//! raises no error: a padding mask that leaks into attention lets pad tokens
//! contribute to the real residues, so every embedding in the batch shifts
//! slightly. Without this test that is a silent quality regression rather than
//! a failure.
//!
//! Each runner's mask plumbing is exercised without weights in its own unit
//! tests (`esm2::esm2`, `amplify::amplify`, `esmc::layers::attention` — the
//! last shared with ESM-3). These tests close the remaining gap: that each
//! *runner* tokenizes, pads and masks a real checkpoint consistently with the
//! way it embeds one sequence at a time.
//!
//! **Not covered here:** ESM-C and ESM-3 have `embed_batch` overrides but no
//! runner-level test against real weights — ESM-C 300M is a ~1.2 GB download
//! and ESM-3 is licence-gated. Their masking is exercised at the layer level
//! (`esmc::layers::attention`, the `MultiHeadAttention` both models share),
//! so what is unverified for them is the runner's own tokenize/pad/mask
//! wiring, not the attention it feeds.
//!
//! They download weights, so they are `#[ignore]`d:
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_plm_batching -- --include-ignored
//! ```

use anyhow::Result;
use candle_core::Tensor;
use ferritin_plms::plm_runner::PlmRunner;
use ferritin_plms::{AmplifyModels, AmplifyRunner, ESM2Models, ESM2Runner, device};

/// Ubiquitin (76 aa) and two shorter fragments, so the batch is ragged in both
/// directions — the longest row drives the padding, and both shorter rows have
/// different amounts of it.
/// Relative agreement required between the batched and single-sequence paths.
///
/// Measured on the real checkpoints: ESM-2 t6 peaks at 3.6e-6 and AMPLIFY 120M
/// at 2.2e-6, both from float accumulation order rather than from padding (see
/// [`max_relative_diff`]). 1e-5 leaves a few times that headroom while staying
/// two orders of magnitude below anything a genuine mask leak produces — the
/// unmasked control in the unit tests moves the result by ~1e-1 relative.
const TOL: f32 = 1e-5;

const SEQS: [&str; 3] = [
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MQIFVKTLTGKTITLEVEPSD",
    "MKTAYIAKQRQISFVKSHFSRQ",
];

/// Largest elementwise difference, **relative to the scale of the activations
/// being compared**.
///
/// An absolute tolerance cannot be shared across these models. ESM-2's `embed`
/// returns post-`layer_norm_after` hidden states with `max |x| ~ 5`; AMPLIFY's
/// returns the last block's output *before* `layer_norm_2`, where `max |x| ~
/// 960`. The same relative agreement therefore looks 200x worse for AMPLIFY
/// under an absolute threshold, which says nothing about whether its padding
/// mask works.
///
/// The residual that remains after masking is float accumulation order: a row
/// embedded inside a longer batch runs its matmuls and softmax reductions at a
/// different sequence length, and 24 layers amplify that. It is *not* leakage
/// from the pad tokens — measured directly by re-running the AMPLIFY case with
/// pad ids 0, 5 and 20, which agree to the last bit (1.159668e-3 each time)
/// while a leak would move with the token.
fn max_relative_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let to_f32 = |t: &Tensor| -> Result<Vec<f32>> {
        Ok(t.flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?)
    };
    let a = to_f32(a)?;
    let b = to_f32(b)?;
    assert_eq!(a.len(), b.len(), "shape mismatch in comparison");
    let scale = a
        .iter()
        .map(|x| x.abs())
        .fold(0f32, f32::max)
        .max(f32::MIN_POSITIVE);
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
        / scale)
}

/// Exact elementwise equality, for the cases that must not drift at all.
fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    let to_f32 = |t: &Tensor| -> Result<Vec<f32>> {
        Ok(t.flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1()?)
    };
    let a = to_f32(a)?;
    let b = to_f32(b)?;
    assert_eq!(a.len(), b.len(), "shape mismatch in comparison");
    Ok(a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max))
}

/// The contract, checked against a real checkpoint.
fn assert_batch_matches_singles(runner: &dyn PlmRunner, tol: f32) -> Result<()> {
    let name = runner.model_name();
    let layout = runner.special_tokens();
    let counts: Vec<usize> = SEQS.iter().map(|s| runner.residue_count(s)).collect();
    let max_residues = counts.iter().copied().max().unwrap();

    let raw = runner.embed_batch(&SEQS)?;
    assert_eq!(
        raw.dims(),
        &[
            SEQS.len(),
            max_residues + layout.total(),
            runner.metadata().d_model
        ],
        "{name}: embed_batch shape"
    );

    let residues = runner.embed_residues_batch(&SEQS)?;
    assert_eq!(
        residues.dims(),
        &[SEQS.len(), max_residues, runner.metadata().d_model],
        "{name}: embed_residues_batch shape"
    );

    for (i, seq) in SEQS.iter().enumerate() {
        let rows = counts[i] + layout.total();

        let single = runner.embed(seq)?;
        let from_batch = raw.narrow(0, i, 1)?.narrow(1, 0, rows)?;
        let diff = max_relative_diff(&single, &from_batch)?;
        assert!(
            diff < tol,
            "{name}: embed_batch row {i} disagrees with embed(): relative diff {diff:.3e}"
        );

        let single_res = runner.embed_residues(seq)?;
        let res_from_batch = residues.narrow(0, i, 1)?.narrow(1, 0, counts[i])?;
        let diff = max_relative_diff(&single_res, &res_from_batch)?;
        assert!(
            diff < tol,
            "{name}: embed_residues_batch row {i} disagrees with embed_residues(): {diff:.3e}"
        );

        // Columns past this sequence are the documented zero padding, not
        // whatever the model happened to compute at a pad token.
        if counts[i] < max_residues {
            let tail = residues
                .narrow(0, i, 1)?
                .narrow(1, counts[i], max_residues - counts[i])?;
            let worst = tail
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec0::<f32>()?;
            assert_eq!(
                worst, 0.0,
                "{name}: padded columns of row {i} should be exactly zero"
            );
        }
    }
    Ok(())
}

/// A batch of one must be exactly what `embed` returns — no padding, no mask,
/// no drift.
fn assert_batch_of_one_is_exact(runner: &dyn PlmRunner) -> Result<()> {
    let single = runner.embed(SEQS[0])?;
    let batched = runner.embed_batch(&SEQS[..1])?;
    assert_eq!(single.dims(), batched.dims());
    let diff = max_abs_diff(&single, &batched)?;
    assert!(
        diff == 0.0,
        "{}: a batch of one drifted from embed() by {diff}",
        runner.model_name()
    );
    Ok(())
}

#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_esm2_embed_batch_matches_single_sequences() -> Result<()> {
    let runner = ESM2Runner::from_pretrained(ESM2Models::T6_8M, device(false)?)?;
    assert_batch_matches_singles(&runner, TOL)?;
    assert_batch_of_one_is_exact(&runner)
}

#[test]
#[ignore = "requires downloading chandar-lab/AMPLIFY_120M weights"]
fn test_amplify_embed_batch_matches_single_sequences() -> Result<()> {
    let runner = AmplifyRunner::from_pretrained(AmplifyModels::AMP120M, device(false)?)?;
    assert_batch_matches_singles(&runner, TOL)?;
    assert_batch_of_one_is_exact(&runner)
}

/// SaProt reads two characters per residue, so its `residue_count` is not
/// `len()`. Batching has to pad by *token* count while `embed_residues_batch`
/// strips by *residue* count; if either used the other's notion of length the
/// rows would be misaligned.
#[test]
#[ignore = "requires downloading westlake-repl/SaProt_35M_AF2 weights"]
fn test_saprot_embed_batch_respects_residue_count() -> Result<()> {
    let runner = ESM2Runner::from_pretrained(ESM2Models::SaProt35M, device(false)?)?;
    let seqs = ["MdEvVpQaLb", "MdEvVp"];
    let counts: Vec<usize> = seqs.iter().map(|s| runner.residue_count(s)).collect();
    assert_eq!(counts, vec![5, 3], "SaProt reads two chars per residue");

    let residues = runner.embed_residues_batch(&seqs)?;
    assert_eq!(residues.dims()[1], 5);
    for (i, seq) in seqs.iter().enumerate() {
        let single = runner.embed_residues(seq)?;
        let from_batch = residues.narrow(0, i, 1)?.narrow(1, 0, counts[i])?;
        let diff = max_relative_diff(&single, &from_batch)?;
        assert!(diff < TOL, "SaProt batch row {i} drifted: {diff:.3e}");
    }
    Ok(())
}
