//! What reduced precision actually costs, measured rather than assumed
//! (ferritin-100.9, extended to accelerators in ferritin-100.19).
//!
//! F16/BF16 exist here because the largest advertised variants are unloadable
//! at F32 — `ESM2Models::T48_15B` is ~60 GB, `ESMCModels::ESMC6B` ~24 GB (and
//! ~12 GB at its on-disk BF16). Halving the footprint is only useful if the
//! numerics survive, so these tests compare each model against **its own F32
//! output on the same device** and record the divergence, rather than trusting
//! that half precision is "close enough".
//!
//! Every test here runs on CPU and, when the crate is built with `--features
//! metal` (or `cuda`) on a machine that has one, again on that accelerator.
//! Nothing is skipped silently: [`measurement_devices`] reports which devices a
//! run covered.
//!
//! # Measured with candle 0.11, ESM2 t6_8M and AMPLIFY 120M
//!
//! Each row is a half-precision run against **that same device's** F32 output,
//! worst case over [`SEQUENCES`].
//!
//! | device | model | dtype | max abs Δlogit | mean abs Δlogit | argmax agreement |
//! |---|---|---|---|---|---|
//! | CPU | ESM2 | F16 | 2.44e-1 | 2.25e-2 | **1.000** |
//! | CPU | ESM2 | BF16 | — | — | refused (no BF16 matmul) |
//! | CPU | AMPLIFY | F16 | 2.39e-1 | 6.87e-2 | **0.923** |
//! | CPU | AMPLIFY | BF16 | — | — | refused (no BF16 matmul) |
//! | Metal (M1) | ESM2 | F16 | 2.14e-2 | 3.47e-3 | **1.000** |
//! | Metal (M1) | ESM2 | BF16 | 2.56e-1 | 3.11e-2 | **1.000** |
//! | Metal (M1) | AMPLIFY | F16 | 4.40e-2 | 8.85e-3 | **1.000** |
//! | Metal (M1) | AMPLIFY | BF16 | 5.19e-1 | 9.53e-2 | **1.000** |
//!
//! The headline is the last column, not the first: a logit shift nothing
//! downstream can see is not a regression. Three things fall out of the table.
//!
//! **Metal F16 is an order of magnitude tighter than CPU F16** — 2.1e-2 vs
//! 2.4e-1 on ESM2, 4.4e-2 vs 2.4e-1 on AMPLIFY — because the Metal matmul
//! accumulates in F32 while candle's CPU F16 gemm accumulates in F16. Same
//! dtype, different arithmetic.
//!
//! **AMPLIFY's F16 caveat is a CPU artifact.** The 0.923 agreement that made
//! F16 a judgement call for AMPLIFY does not reproduce on Metal, where every
//! top-1 prediction survives. On CPU it is still one position in thirteen.
//!
//! **BF16 on Metal costs about what F16 on CPU costs** and changes no top-1
//! prediction on either model. It trades three mantissa bits for exponent
//! range, so it lands between Metal F16 and CPU F16 — usable, and the reason
//! [`LoadOptions::validate`] scopes its refusal to the CPU backend rather than
//! to the dtype.
//!
//! Speed is reported, not asserted, by `test_half_precision_throughput`; see
//! its doc comment for why, and for what an M1 actually does.
//!
//! ```shell
//! cargo test -p ferritin-plms --test test_plm_dtype_parity -- --include-ignored
//! cargo test -p ferritin-plms --features metal --test test_plm_dtype_parity -- --include-ignored --nocapture
//! ```

use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use ferritin_plms::loader::LoadOptions;
use ferritin_plms::{AmplifyModels, AmplifyRunner, ESM2Models, ESM2Runner, device};
use std::time::Instant;

const SEQUENCES: &[&str] = &["MQIFVKTLTGK", "GGGGGGGGG", "KEKEKEKEK"];

/// Every device this build can reach: CPU, plus whatever accelerator
/// [`device`] selects when one is compiled in and present.
///
/// Returned with a name so failures say *where* they happened — a tolerance
/// that holds on CPU and not on Metal is the interesting case, and an
/// assertion that does not name the device buries it.
fn measurement_devices() -> Result<Vec<(&'static str, Device)>> {
    let mut devices = vec![("cpu", Device::Cpu)];
    let accelerator = device(false)?;
    if !accelerator.is_cpu() {
        let name = if accelerator.is_metal() {
            "metal"
        } else {
            "cuda"
        };
        devices.push((name, accelerator));
    }
    Ok(devices)
}

/// Half-precision dtypes worth measuring on `dev`.
///
/// BF16 is omitted on CPU because [`LoadOptions::validate`] refuses it there;
/// `test_bf16_refused_on_cpu_with_explanation` covers that path instead.
fn half_dtypes(dev: &Device) -> &'static [DType] {
    if dev.is_cpu() {
        &[DType::F16]
    } else {
        &[DType::F16, DType::BF16]
    }
}

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

/// How far a half-precision run may drift from its own-device F32 baseline.
///
/// Bounds sit above the measured values with headroom, but tight enough that a
/// real numerical regression trips them. BF16 gets its own row because it
/// trades 3 mantissa bits for exponent range: the same model diverges further
/// at BF16 than at F16, and pretending one bound covers both would either
/// fail on BF16 or stop catching anything at F16.
struct Tolerance {
    max_diff: f32,
    mean_diff: f32,
    min_agreement: f32,
}

/// ESM2 t6_8M bounds. F16 has never moved a top-1 prediction on any device.
fn esm2_tolerance(dtype: DType) -> Tolerance {
    match dtype {
        DType::F16 => Tolerance {
            max_diff: 0.5,      // measured 2.44e-1 (CPU)
            mean_diff: 0.05,    // measured 2.25e-2 (CPU)
            min_agreement: 1.0, // measured 1.000 (CPU)
        },
        DType::BF16 => Tolerance {
            max_diff: 0.6,      // measured 2.56e-1 (Metal)
            mean_diff: 0.08,    // measured 3.11e-2 (Metal)
            min_agreement: 1.0, // measured 1.000 (Metal)
        },
        other => panic!("no ESM2 tolerance recorded for {other:?}"),
    }
}

/// AMPLIFY 120M bounds. Deliberately a floor, not an equality: F16 already
/// changes roughly one AMPLIFY top-1 in thirteen, and this test exists to keep
/// that documented and catch it getting worse.
fn amplify_tolerance(dtype: DType) -> Tolerance {
    match dtype {
        DType::F16 => Tolerance {
            max_diff: 0.5,       // measured 2.39e-1 (CPU)
            mean_diff: 0.15,     // measured 6.87e-2 (CPU)
            min_agreement: 0.85, // measured 0.923 (CPU)
        },
        DType::BF16 => Tolerance {
            max_diff: 1.0,       // measured 5.19e-1 (Metal)
            mean_diff: 0.2,      // measured 9.53e-2 (Metal)
            min_agreement: 0.85, // measured 1.000 (Metal)
        },
        other => panic!("no AMPLIFY tolerance recorded for {other:?}"),
    }
}

/// Check one half-precision run against its F32 baseline, printing the row it
/// contributes to the table above so `--nocapture` on new hardware produces
/// the numbers to record.
fn assert_within(
    label: &str,
    device_name: &str,
    dtype: DType,
    sequence: &str,
    f32_logits: &Tensor,
    half_logits: &Tensor,
    tol: &Tolerance,
) -> Result<()> {
    let (max, mean) = diff_stats(f32_logits, half_logits)?;
    let agreement = argmax_agreement(f32_logits, half_logits)?;
    println!(
        "{device_name:>5} | {label:<7} | {dtype:?} | {sequence:<12} | \
         max {max:.3e} | mean {mean:.3e} | agreement {agreement:.4}"
    );

    assert!(
        max < tol.max_diff,
        "{device_name}/{label}/{dtype:?} {sequence}: max logit diff {max:.3e} \
         exceeds {:.1e}",
        tol.max_diff
    );
    assert!(
        mean < tol.mean_diff,
        "{device_name}/{label}/{dtype:?} {sequence}: mean logit diff {mean:.3e} \
         exceeds {:.1e}",
        tol.mean_diff
    );
    assert!(
        agreement >= tol.min_agreement,
        "{device_name}/{label}/{dtype:?} {sequence}: top-1 agreement {agreement:.4} \
         below {:.2}",
        tol.min_agreement
    );
    Ok(())
}

/// ESM2 in half precision, on every device this build can reach.
#[test]
#[ignore = "requires downloading facebook/esm2_t6_8M_UR50D weights"]
fn test_esm2_half_precision_matches_f32() -> Result<()> {
    for (device_name, dev) in measurement_devices()? {
        let f32_model =
            ESM2Runner::from_pretrained_with(ESM2Models::T6_8M, &LoadOptions::new(dev.clone()))?;
        let baseline: Vec<Tensor> = SEQUENCES
            .iter()
            .map(|s| Ok(f32_model.run_forward(s)?.logits))
            .collect::<Result<_>>()?;

        for &dtype in half_dtypes(&dev) {
            let half_model = ESM2Runner::from_pretrained_with(
                ESM2Models::T6_8M,
                &LoadOptions::new(dev.clone()).with_dtype(dtype),
            )?;
            for (sequence, f32_logits) in SEQUENCES.iter().zip(&baseline) {
                let half_logits = half_model.run_forward(sequence)?.logits;
                assert_eq!(
                    half_logits.dtype(),
                    dtype,
                    "{device_name}: ESM2 asked for {dtype:?} returned {:?}",
                    half_logits.dtype()
                );
                assert_within(
                    "ESM2",
                    device_name,
                    dtype,
                    sequence,
                    f32_logits,
                    &half_logits,
                    &esm2_tolerance(dtype),
                )?;
            }
        }
    }
    Ok(())
}

/// AMPLIFY in half precision, on every device this build can reach.
///
/// Also the regression test for AMPLIFY's rotary table, which was built at F32
/// regardless of the model dtype and killed an F16 model in `apply_rotary_emb`
/// with "dtype mismatch in mul, lhs: F16, rhs: F32". Getting a forward pass out
/// at all is the assertion.
#[test]
#[ignore = "requires downloading chandar-lab/AMPLIFY_120M weights"]
fn test_amplify_half_precision_matches_f32() -> Result<()> {
    for (device_name, dev) in measurement_devices()? {
        let f32_model = AmplifyRunner::from_pretrained_with(
            AmplifyModels::AMP120M,
            &LoadOptions::new(dev.clone()),
        )?;
        let baseline: Vec<Tensor> = SEQUENCES
            .iter()
            .map(|s| Ok(f32_model.run_forward(s)?.logits))
            .collect::<Result<_>>()?;

        for &dtype in half_dtypes(&dev) {
            let half_model = AmplifyRunner::from_pretrained_with(
                AmplifyModels::AMP120M,
                &LoadOptions::new(dev.clone()).with_dtype(dtype),
            )?;
            for (sequence, f32_logits) in SEQUENCES.iter().zip(&baseline) {
                let half_logits = half_model.run_forward(sequence)?.logits;
                assert_eq!(
                    half_logits.dtype(),
                    dtype,
                    "{device_name}: AMPLIFY asked for {dtype:?} returned {:?}",
                    half_logits.dtype()
                );
                assert_within(
                    "AMPLIFY",
                    device_name,
                    dtype,
                    sequence,
                    f32_logits,
                    &half_logits,
                    &amplify_tolerance(dtype),
                )?;
            }
        }
    }
    Ok(())
}

/// An accelerator at F32 must agree with the CPU at F32.
///
/// The dtype tests above each compare a device against *itself*, so a Metal or
/// CUDA kernel that is uniformly wrong would pass all of them. This is the one
/// check that would catch it.
#[test]
#[ignore = "requires an accelerator and facebook/esm2_t6_8M_UR50D weights"]
fn test_accelerator_f32_matches_cpu_f32() -> Result<()> {
    let accelerator = device(false)?;
    if accelerator.is_cpu() {
        println!("no accelerator in this build; nothing to compare against CPU");
        return Ok(());
    }

    let cpu_model =
        ESM2Runner::from_pretrained_with(ESM2Models::T6_8M, &LoadOptions::new(Device::Cpu))?;
    let gpu_model =
        ESM2Runner::from_pretrained_with(ESM2Models::T6_8M, &LoadOptions::new(accelerator))?;

    for sequence in SEQUENCES {
        let a = cpu_model.run_forward(sequence)?.logits;
        let b = gpu_model.run_forward(sequence)?.logits;
        let (max, mean) = diff_stats(&a, &b)?;
        let agreement = argmax_agreement(&a, &b)?;
        println!(
            "accel-vs-cpu F32 | {sequence:<12} | max {max:.3e} | mean {mean:.3e} | agreement {agreement:.4}"
        );
        assert!(
            max < 1e-2,
            "{sequence}: accelerator F32 differs from CPU F32 by {max:.3e} — \
             that is a kernel disagreement, not rounding"
        );
        assert_eq!(
            agreement, 1.0,
            "{sequence}: accelerator F32 changed a top-1 prediction (agreement {agreement:.4})"
        );
    }
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

/// The refusal above is scoped to CPU, not to BF16.
///
/// Metal and CUDA do have BF16 matmul, so `validate` must let them through —
/// otherwise the dtype that halves memory on the only devices big models fit
/// on would be unreachable.
#[test]
fn test_bf16_allowed_on_accelerators() -> Result<()> {
    let accelerator = device(false)?;
    if accelerator.is_cpu() {
        println!("no accelerator in this build; CPU refusal is covered separately");
        return Ok(());
    }
    LoadOptions::new(accelerator)
        .with_dtype(DType::BF16)
        .validate()
}

/// A sequence long enough that a forward pass measures arithmetic rather than
/// dispatch overhead: ubiquitin three times over, 228 residues.
const BENCH_SEQUENCE: &str = concat!(
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
);

/// Forward passes to time, after discarding warmups.
const BENCH_ITERATIONS: usize = 10;

/// Warmup passes, which pay for lazy Metal pipeline compilation and first-touch
/// buffer allocation. Timing these instead of discarding them is most of how a
/// GPU gets reported as slower than a CPU.
const BENCH_WARMUPS: usize = 3;

/// What half precision buys in wall time, per forward pass, load excluded.
///
/// **Reports, asserts nothing.** A throughput assertion on shared CI hardware
/// is a flake generator, and the number that matters — is the GPU worth it for
/// *my* model at *my* length — depends on both. So this prints a table and
/// leaves the judgement to whoever reads it.
///
/// Measured on an M1 (8 GPU cores), release build, 228 residues, ms per
/// forward:
///
/// | device | model | F32 | F16 | BF16 |
/// |---|---|---|---|---|
/// | CPU | ESM2 t6_8M | 70.4 | 74.8 | — |
/// | Metal | ESM2 t6_8M | **1.35** | **0.80** | 1.00 |
/// | CPU | AMPLIFY 120M | 724 | 406 | — |
/// | Metal | AMPLIFY 120M | 159 | 149 | 159 |
///
/// Two things to take from it. **F16 on CPU is a memory optimisation, not a
/// speed one** — ESM2 got slower (70.4 → 74.8 ms), because candle converts to
/// F32 to multiply and the conversions cost more than the narrower loads save.
/// On Metal it is a real 1.7x. And **ESM2 accelerates 52x while AMPLIFY
/// manages 4.6x**, despite AMPLIFY being only ~15x the parameters; something
/// in AMPLIFY's forward is serialising against the GPU rather than the
/// arithmetic being the limit (ferritin-100.30).
///
/// Run it in release; a debug build measures candle's un-inlined CPU loops and
/// says nothing useful about either backend.
#[test]
#[ignore = "a benchmark, and needs both models' weights"]
fn test_half_precision_throughput() -> Result<()> {
    for (device_name, dev) in measurement_devices()? {
        let mut dtypes = vec![DType::F32];
        dtypes.extend_from_slice(half_dtypes(&dev));

        for dtype in dtypes {
            let opts = LoadOptions::new(dev.clone()).with_dtype(dtype);

            let esm2 = ESM2Runner::from_pretrained_with(ESM2Models::T6_8M, &opts)?;
            let esm2_ms = time_forwards(|| esm2.run_forward(BENCH_SEQUENCE).map(|_| ()))?;

            let amplify = AmplifyRunner::from_pretrained_with(AmplifyModels::AMP120M, &opts)?;
            let amplify_ms = time_forwards(|| amplify.run_forward(BENCH_SEQUENCE).map(|_| ()))?;

            println!(
                "{device_name:>5} | {dtype:?} | ESM2 {esm2_ms:7.2} ms/forward | \
                 AMPLIFY {amplify_ms:7.2} ms/forward"
            );
        }
    }
    Ok(())
}

/// Mean milliseconds per call over [`BENCH_ITERATIONS`], after
/// [`BENCH_WARMUPS`] discarded calls.
fn time_forwards(mut forward: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..BENCH_WARMUPS {
        forward()?;
    }
    let started = Instant::now();
    for _ in 0..BENCH_ITERATIONS {
        forward()?;
    }
    Ok(started.elapsed().as_secs_f64() * 1e3 / BENCH_ITERATIONS as f64)
}
