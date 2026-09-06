//! Unit tests for the shared parity harness (`tests/support/parity.rs`).
//!
//! These need no model weights or fixtures — they build small tensors by hand
//! and check that each assertion helper accepts matching data, rejects
//! mismatched data, and names the offending position in its error.

mod support;

use candle_core::{Device, Tensor};
use support::parity::{
    SpecialTokens, align_rows, assert_distribution_close, assert_embeddings_close,
    assert_logits_close, fixture_path,
};

fn t2(rows: &[&[f32]], device: &Device) -> Tensor {
    let l = rows.len();
    let v = rows[0].len();
    let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    Tensor::from_vec(flat, (l, v), device).unwrap()
}

#[test]
fn logits_close_accepts_within_tolerance() {
    let d = Device::Cpu;
    let a = t2(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]], &d);
    let b = t2(&[&[1.0005, 2.0, 3.0], &[4.0, 5.0005, 6.0]], &d);
    assert_logits_close(&a, &b, 1e-3).expect("within tolerance should pass");
}

#[test]
fn logits_close_reports_offending_position() {
    let d = Device::Cpu;
    let a = t2(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]], &d);
    let b = t2(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.5]], &d); // off at (1, 2)
    let err = assert_logits_close(&a, &b, 1e-3).unwrap_err().to_string();
    assert!(
        err.contains("position 1") && err.contains("vocab_index 2"),
        "error should name (position 1, vocab_index 2); got: {err}"
    );
}

#[test]
fn logits_close_reports_shape_mismatch() {
    let d = Device::Cpu;
    let a = t2(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]], &d);
    let b = t2(&[&[1.0, 2.0, 3.0]], &d);
    let err = assert_logits_close(&a, &b, 1e-3).unwrap_err().to_string();
    assert!(err.contains("length mismatch"), "got: {err}");
}

#[test]
fn distribution_close_accepts_equal_distributions() {
    let d = Device::Cpu;
    let p = t2(&[&[0.5, 0.5], &[0.25, 0.75]], &d);
    assert_distribution_close(&p, &p, 1e-6).expect("identical distributions should pass");
}

#[test]
fn distribution_close_reports_offending_position() {
    let d = Device::Cpu;
    let rust = t2(&[&[0.5, 0.5], &[0.9, 0.1]], &d);
    let reference = t2(&[&[0.5, 0.5], &[0.1, 0.9]], &d); // diverges at row 1
    let err = assert_distribution_close(&rust, &reference, 0.01)
        .unwrap_err()
        .to_string();
    assert!(err.contains("position 1"), "got: {err}");
}

#[test]
fn embeddings_close_accepts_parallel_vectors() {
    let d = Device::Cpu;
    let rust = t2(&[&[1.0, 0.0], &[0.0, 2.0]], &d);
    let reference = t2(&[&[2.0, 0.0], &[0.0, 5.0]], &d); // same directions
    assert_embeddings_close(&rust, &reference, 0.999).expect("parallel rows should pass");
}

#[test]
fn embeddings_close_reports_offending_position() {
    let d = Device::Cpu;
    let rust = t2(&[&[1.0, 0.0], &[1.0, 0.0]], &d);
    let reference = t2(&[&[1.0, 0.0], &[0.0, 1.0]], &d); // orthogonal at row 1
    let err = assert_embeddings_close(&rust, &reference, 0.9)
        .unwrap_err()
        .to_string();
    assert!(err.contains("position 1"), "got: {err}");
}

#[test]
fn align_rows_strips_bos_eos() {
    let d = Device::Cpu;
    // (1, 4, 2): BOS + two residues + EOS
    let batched = t2(&[&[9.0, 9.0], &[1.0, 1.0], &[2.0, 2.0], &[8.0, 8.0]], &d)
        .unsqueeze(0)
        .unwrap();
    let aligned = align_rows(&batched, SpecialTokens::BOS_EOS).unwrap();
    assert_eq!(aligned.dims(), &[2, 2], "should drop BOS and EOS rows");
    let rows = aligned.to_vec2::<f32>().unwrap();
    assert_eq!(rows[0], vec![1.0, 1.0]);
    assert_eq!(rows[1], vec![2.0, 2.0]);
}

#[test]
fn align_rows_none_is_identity() {
    let d = Device::Cpu;
    let m = t2(&[&[1.0, 1.0], &[2.0, 2.0]], &d);
    let out = align_rows(&m, SpecialTokens::NONE).unwrap();
    assert_eq!(out.dims(), &[2, 2]);
}

#[test]
fn fixture_load_missing_names_generator_script() {
    let d = Device::Cpu;
    let err = support::parity::ParityFixture::load("nonexistent_parity", &d)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("generate_nonexistent_fixtures.py"),
        "missing-fixture error should point at the generator script; got: {err}"
    );
    // sanity: fixture_path resolves under the crate's tests/fixtures dir
    assert!(
        fixture_path("nonexistent_parity")
            .to_string_lossy()
            .ends_with("tests/fixtures/nonexistent_parity.safetensors")
    );
}

// ── Declared parity coverage (ferritin-100.20) ───────────────────────────────

/// The declared coverage table must match what is actually on disk.
///
/// This is what keeps the skip-when-absent behaviour from turning the nightly
/// into a vacuous green. Committing a fixture without flipping its entry to
/// `Committed` — or deleting a committed one — fails here.
#[test]
fn test_parity_coverage_is_accurate() {
    use support::parity::{CoverageStatus, PARITY_COVERAGE, fixture_path};

    for entry in PARITY_COVERAGE {
        let exists = fixture_path(entry.fixture).exists();
        match entry.status {
            CoverageStatus::Committed => assert!(
                exists,
                "'{}' is declared Committed but is missing from tests/fixtures/. \
                 Restore it, or change its PARITY_COVERAGE status.",
                entry.fixture
            ),
            CoverageStatus::NotGenerated { .. } => assert!(
                !exists,
                "'{}' exists on disk but is declared NotGenerated. Flip its \
                 PARITY_COVERAGE entry to Committed so its parity test runs.",
                entry.fixture
            ),
        }
    }
}

/// Every fixture in `tests/fixtures/` must be declared, so a fixture cannot be
/// added without its coverage status being stated.
#[test]
fn test_no_undeclared_fixtures_on_disk() {
    use support::parity::declared_coverage;

    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for entry in std::fs::read_dir(&dir)
        .expect("tests/fixtures should exist")
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|e| e != "safetensors") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        assert!(
            declared_coverage(&stem).is_some(),
            "fixture '{stem}' is on disk but absent from PARITY_COVERAGE; declare it there"
        );
    }
}

/// Records which ports currently have NO numerical parity coverage.
///
/// Deliberately an assertion on the exact set rather than a `println!`: the gap
/// is a reviewed fact, so closing one (or opening a new one) has to be a
/// conscious edit here. Shrinking this list is the goal.
#[test]
fn test_uncovered_ports_are_the_known_set() {
    use support::parity::{CoverageStatus, PARITY_COVERAGE};

    let mut uncovered: Vec<&str> = PARITY_COVERAGE
        .iter()
        .filter(|c| matches!(c.status, CoverageStatus::NotGenerated { .. }))
        .map(|c| c.fixture)
        .collect();
    uncovered.sort_unstable();

    assert_eq!(
        uncovered,
        ["1BC8_log_probs", "esm3_parity", "esmc_parity"],
        "the set of ports without parity coverage changed; update this test and \
         ferritin-100.20 deliberately rather than letting it drift"
    );
}
