//! Integration smoke tests for the AMPLIFY Candle ports.

mod support;

use ferritin_plms::amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
use ferritin_plms::device;
use support::model_harness::{TEST_SEQUENCE, run_remote_amplify_prediction_smoke};

fn load_amplify_120m() -> AmplifyRunner {
    AmplifyRunner::load_model(AmplifyModels::AMP120M, device().unwrap())
        .expect("Failed to load AMPLIFY 120M model")
}

#[test]
#[ignore = "requires downloading model files"]
fn test_load_amplify_120m() {
    run_remote_amplify_prediction_smoke(AmplifyModels::AMP120M)
        .expect("Failed to load AMPLIFY 120M model");
}

#[test]
#[ignore = "requires downloading model files"]
fn test_amplify_120m_prediction() {
    run_remote_amplify_prediction_smoke(AmplifyModels::AMP120M)
        .expect("Failed to get prediction from AMPLIFY 120M");
}

#[test]
#[ignore = "requires downloading model files"]
fn test_amplify_120m_probabilities() {
    let amplify = load_amplify_120m();
    let probabilities = amplify.get_pseudo_probabilities(TEST_SEQUENCE);
    assert!(
        probabilities.is_ok(),
        "Failed to get pseudo-probabilities: {:?}",
        probabilities.err()
    );
    let probabilities = probabilities.unwrap();
    assert!(
        !probabilities.is_empty(),
        "Probabilities should not be empty"
    );

    let unique_positions: std::collections::HashSet<_> =
        probabilities.iter().map(|p| p.position).collect();
    assert_eq!(
        unique_positions.len(),
        TEST_SEQUENCE.len(),
        "Should have predictions for each position in the sequence"
    );
}

#[test]
#[ignore = "requires downloading model files"]
fn test_amplify_120m_contact_map() {
    let amplify = load_amplify_120m();
    let contact_map = amplify.get_contact_map(TEST_SEQUENCE);
    assert!(
        contact_map.is_ok(),
        "Failed to get contact map: {:?}",
        contact_map.err()
    );

    let contacts = contact_map.unwrap();
    assert!(!contacts.is_empty(), "Contact map should not be empty");

    let sequence_len = TEST_SEQUENCE.len();
    for contact in &contacts {
        assert!(
            contact.position_1 < sequence_len,
            "Position 1 should be valid"
        );
        assert!(
            contact.position_2 < sequence_len,
            "Position 2 should be valid"
        );
        assert!(
            !contact.contact_estimate.is_nan(),
            "Contact estimate should be a valid number"
        );
    }
}
