//! Optional numerical parity for the ProGen2-small generative runner.

mod support;

use anyhow::Result;
use ferritin_plms::{GenerativeModel, ProGen2, ProGenModels, device};
use support::parity::{ParityFixture, fixture_path, hf_tests_enabled};

const SEQUENCES: &[(&str, &str)] = &[
    ("ubiquitin_nterm", "MQIFVKTLTGK"),
    ("glycine_repeat", "GGGGGGGGG"),
    ("alt_charged", "KEKEKEKEK"),
];

/// Compare Rust sequence scores against the optional Python fixture.
///
/// This is intentionally gated until a checkpoint fixture is generated: the
/// 617 MB model is not part of the ordinary test path. Once present, the test
/// is strict and compares the same summed residue log-likelihood as the
/// upstream ProGen likelihood script.
#[test]
fn test_progen2_likelihood_parity_vs_python_reference() -> Result<()> {
    if !hf_tests_enabled() {
        eprintln!("skipping ProGen2 parity: set FERRITIN_HF_TESTS=1 to enable");
        return Ok(());
    }
    if !fixture_path("progen2_parity").exists() {
        eprintln!(
            "skipping ProGen2 parity: generate it with python scripts/generate_progen2_fixtures.py \
             --output crates/ferritin-plms/tests/fixtures/"
        );
        return Ok(());
    }

    let dev = device(false)?;
    let runner = ProGen2::from_pretrained(ProGenModels::Small, dev.clone())?;
    let fixture = ParityFixture::load("progen2_parity", &dev)?;
    for (name, sequence) in SEQUENCES {
        let expected = fixture
            .tensor(&format!("{name}_log_likelihood"))?
            .to_scalar::<f32>()?;
        let actual = runner.log_likelihood(sequence)?;
        assert!(
            (actual - expected).abs() <= 1e-3,
            "{name}: {actual} vs {expected}"
        );
    }
    Ok(())
}
