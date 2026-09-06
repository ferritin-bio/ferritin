//! What reduced precision actually costs, measured rather than assumed
//! (ferritin-100.9).
//!
//! F16/BF16 exist here because the largest advertised variants are unloadable
//! at F32 — `ESM2Models::T48_15B` is ~60 GB, `ESMCModels::ESMC6B` ~24 GB (and
//! ~12 GB at its on-disk BF16). Halving the footprint is only useful if the
//! numerics survive, so these tests compare each model against **its own F32
//! output** and record the divergence, rather than trusting that half
//! precision is "close enough".
//!
//! # Measured on CPU, candle 0.11, ESM2 t6_8M and AMPLIFY 120M
//!
//! | model | dtype | max abs Δlogit | mean abs Δlogit | argmax agreement |
//! |---|---|---|---|---|
//! | ESM2 | F16 | 2.44e-1 | 2.25e-2 | **1.000** |
//! | AMPLIFY | F16 | 2.39e-1 | 6.87e-2 | **0.923** |
//! | either | BF16 | — | — | unsupported on CPU |
//!
//! The headline is the last column, not the first. Both models shift logits by
//! a similar amount, but that shift changes **no** ESM2 top-1 prediction and
//! roughly **one AMPLIFY position in thirteen**. So F16 is a safe swap for
//! ESM2 inference and a judgement call for AMPLIFY — worth knowing before
//! trading precision for memory.
//!
//! BF16 is refused up front on CPU: candle has no BF16 `matmul` there, so it
//! would otherwise fail partway through loading (ESM2) or on the first forward
//! pass (AMPLIFY) with a bare "unsupported dtype BF16 for op matmul".
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_plm_dtype_parity -- --include-ignored
//! ```

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use ferritin_plms::loader::LoadOptions;
use ferritin_plms::{AmplifyModels, AmplifyRunner, ESM2Models, ESM2Runner, device};

const SEQUENCES: &[&str] = &["MQIFVKTLTGK", "GGGGGGGGG", "KEKEKEKEK"];

/// `(max abs diff, mean abs diff)` between two logits tensors.
fn diff_stats(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let diffs = a.iter().zip(&b).map(|(x, y)| (x - y).abs());
    let max = diffs.clone().fold(0.0f32, f32::max);
    let mean = diffs.sum::<f32>() / a.len() as f32;
    Ok((max, mean))
}

/// Fraction of positions whose top-1 predicted token is unchanged.
///
/// The measure that actually matters: a logit shift nothing downstream can see
/// is not a regression.
fn argmax_agreement(a: &Tensor, b: &Tensor) -> Result<f32> {
    let am = a
        .to_dtype(DType::F32)?
        .argmax(D::Minus1)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let bm = b
        .to_dtype(DType::F32)?
        .argmax(D::Minus1)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let same = am.iter().zip(&bm).filter(|(x, y)| x == y).count();
    Ok(same as f32 / am.len() as f32)
}

// Bounds sit above the measured values with headroom, but tight enough that a
// real numerical regression trips them.
const ESM2_F16_MAX_DIFF: f32 = 0.5; // measured 2.44e-1
const ESM2_F16_MEAN_DIFF: f32 = 0.05; // measured 2.25e-2
const AMPLIFY_F16_MAX_DIFF: f32 = 0.5; // measured 2.39e-1
const AMPLIFY_F16_MEAN_DIFF: f32 = 0.15; // measured 6.87e-2
const AMPLIFY_F16_MIN_AGREEMENT: f32 = 0.85; // measured 0.923

/// ESM2 at F16 shifts logits slightly but changes no top-1 prediction.
#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_esm2_f16_matches_f32() -> Result<()> {
    let dev = device(false)?;
    let f32_model =
        ESM2Runner::from_pretrained_with(ESM2Models::T6_8M, &LoadOptions::new(dev.clone()))?;
    let f16_model = ESM2Runner::from_pretrained_with(
        ESM2Models::T6_8M,
        &LoadOptions::new(dev).with_dtype(DType::F16),
    )?;

    for sequence in SEQUENCES {
        let a = f32_model.run_forward(sequence)?.logits;
        let b = f16_model.run_forward(sequence)?.logits;

        let (max, mean) = diff_stats(&a, &b)?;
        assert!(
            max < ESM2_F16_MAX_DIFF,
            "{sequence}: ESM2 F16 max logit diff {max:.3e} exceeds {ESM2_F16_MAX_DIFF:.1e}"
        );
        assert!(
            mean < ESM2_F16_MEAN_DIFF,
            "{sequence}: ESM2 F16 mean logit diff {mean:.3e} exceeds {ESM2_F16_MEAN_DIFF:.1e}"
        );

        let agreement = argmax_agreement(&a, &b)?;
        assert_eq!(
            agreement, 1.0,
            "{sequence}: ESM2 F16 changed a top-1 prediction (agreement {agreement:.4}); \
             it has been exactly 1.0 — investigate before relaxing this"
        );
    }
    Ok(())
}

/// AMPLIFY at F16 changes roughly one top-1 prediction in thirteen.
///
/// Asserted as a floor rather than equality precisely because it is *not* 1.0:
/// this test exists to keep that documented and to catch it getting worse.
#[test]
#[ignore = "requires downloading chandar-lab/AMPLIFY_120M weights"]
fn test_amplify_f16_matches_f32() -> Result<()> {
    let dev = device(false)?;
    let f32_model = AmplifyRunner::from_pretrained_with(
        AmplifyModels::AMP120M,
        &LoadOptions::new(dev.clone()),
    )?;
    let f16_model = AmplifyRunner::from_pretrained_with(
        AmplifyModels::AMP120M,
        &LoadOptions::new(dev).with_dtype(DType::F16),
    )?;

    for sequence in SEQUENCES {
        let a = f32_model.run_forward(sequence)?.logits;
        let b = f16_model.run_forward(sequence)?.logits;

        let (max, mean) = diff_stats(&a, &b)?;
        assert!(
            max < AMPLIFY_F16_MAX_DIFF,
            "{sequence}: AMPLIFY F16 max logit diff {max:.3e} exceeds {AMPLIFY_F16_MAX_DIFF:.1e}"
        );
        assert!(
            mean < AMPLIFY_F16_MEAN_DIFF,
            "{sequence}: AMPLIFY F16 mean logit diff {mean:.3e} exceeds {AMPLIFY_F16_MEAN_DIFF:.1e}"
        );

        let agreement = argmax_agreement(&a, &b)?;
        assert!(
            agreement >= AMPLIFY_F16_MIN_AGREEMENT,
            "{sequence}: AMPLIFY F16 top-1 agreement {agreement:.4} below \
             {AMPLIFY_F16_MIN_AGREEMENT:.2}"
        );
    }
    Ok(())
}

/// AMPLIFY's rotary table was built at F32 regardless of the model dtype, so
/// an F16 model died in `apply_rotary_emb` with "dtype mismatch in mul, lhs:
/// F16, rhs: F32". Loading and running at F16 at all is the regression test.
#[test]
#[ignore = "requires downloading chandar-lab/AMPLIFY_120M weights"]
fn test_amplify_f16_rotary_dtype_matches_model() -> Result<()> {
    let dev = device(false)?;
    let model = AmplifyRunner::from_pretrained_with(
        AmplifyModels::AMP120M,
        &LoadOptions::new(dev).with_dtype(DType::F16),
    )?;
    let out = model.run_forward(SEQUENCES[0])?;
    assert_eq!(out.logits.dtype(), DType::F16);
    Ok(())
}

/// BF16 is refused on CPU with an explanation, not candle's bare matmul error.
///
/// Needs no weights: the check runs before any download.
#[test]
fn test_bf16_refused_on_cpu_with_explanation() {
    let err = LoadOptions::new(Device::Cpu)
        .with_dtype(DType::BF16)
        .validate()
        .expect_err("BF16 on CPU must be refused up front");
    let msg = err.to_string();
    assert!(
        msg.contains("not supported on the CPU backend"),
        "error should explain the limitation; got: {msg}"
    );
    assert!(
        msg.contains("F16"),
        "error should point at the workable alternative; got: {msg}"
    );
}
