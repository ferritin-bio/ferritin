//! Test the AMPLIFY models
//!
//! This test file validates loading and running the AMPLIFY models,
//! particularly the AMP120M model.

use candle_core::Device;
use ferritin_plms::amplify::amplify_runner::{AmplifyModels, AmplifyRunner};
use ferritin_plms::device;

const TEST_SEQUENCE: &str = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";

/// Test that we can successfully load the AMPLIFY 120M model
#[test]
#[ignore = "requires downloading model files"]
fn test_load_amplify_120m() {
    let amplify = AmplifyRunner::load_model(AmplifyModels::AMP120M, device().unwrap());
    assert!(
        amplify.is_ok(),
        "Failed to load AMPLIFY 120M model: {:?}",
        amplify.err()
    );
}

/// Test a simple sequence prediction using AMPLIFY
#[test]
#[ignore = "requires downloading model files"]
fn test_amplify_120m_prediction() {
    let amplify = AmplifyRunner::load_model(AmplifyModels::AMP120M, device().unwrap())
        .expect("Failed to load AMPLIFY 120M model");
    let sequence = TEST_SEQUENCE;
    let prediction = amplify.get_best_prediction(sequence);
    assert!(
        prediction.is_ok(),
        "Failed to get prediction: {:?}",
        prediction.err()
    );
    let prediction = prediction.unwrap();
    println!("Input: {}\nPrediction: {}", sequence, prediction);
    assert!(!prediction.is_empty(), "Prediction should not be empty");
}

/// Test getting pseudo-probabilities from the model
#[test]
#[ignore = "requires downloading model files"]
fn test_amplify_120m_probabilities() {
    let amplify = AmplifyRunner::load_model(AmplifyModels::AMP120M, device().unwrap())
        .expect("Failed to load AMPLIFY 120M model");
    let sequence = TEST_SEQUENCE;
    let probabilities = amplify.get_pseudo_probabilities(sequence);
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
    println!("Got {} probability entries", probabilities.len());

    // Check that we have entries for each position in the sequence
    let unique_positions: std::collections::HashSet<_> =
        probabilities.iter().map(|p| p.position).collect();

    // Each position should have multiple amino acid predictions
    assert_eq!(
        unique_positions.len(),
        sequence.len(),
        "Should have predictions for each position in the sequence"
    );
}

/// Test getting contact maps from the model
#[test]
#[ignore = "requires downloading model files"]
fn test_amplify_120m_contact_map() {
    let amplify = AmplifyRunner::load_model(AmplifyModels::AMP120M, device().unwrap())
        .expect("Failed to load AMPLIFY 120M model");
    let sequence = TEST_SEQUENCE;

    // Test getting contact map
    let contact_map = amplify.get_contact_map(sequence);
    assert!(
        contact_map.is_ok(),
        "Failed to get contact map: {:?}",
        contact_map.err()
    );

    let contacts = contact_map.unwrap();
    assert!(!contacts.is_empty(), "Contact map should not be empty");
    println!("Got {} contact entries", contacts.len());

    // Basic validation of contact map data
    let sequence_len = sequence.len();
    for contact in &contacts {
        assert!(
            contact.position_1 < sequence_len,
            "Position 1 should be valid"
        );
        assert!(
            contact.position_2 < sequence_len,
            "Position 2 should be valid"
        );
        // Contact estimate should be a valid float
        assert!(
            !contact.contact_estimate.is_nan(),
            "Contact estimate should be a valid number"
        );
    }
}
