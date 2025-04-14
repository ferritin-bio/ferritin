//! Test the ESM2 models
//!
//! This test file validates loading and running the ESM2 models,
//! particularly the ESM2-650M model.

use candle_core::Device;
use ferritin_plms::esm2::esm2_runner::{Esm2Models, Esm2Runner};
use ferritin_plms::device;

const TEST_SEQUENCE: &str = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";

/// Test that we can successfully load the ESM2 650M model
#[test]
#[ignore = "requires downloading model files"]
fn test_load_esm2_650m() {
    let esm2 = Esm2Runner::load_model(Esm2Models::ESM2_650M, device().unwrap());
    assert!(
        esm2.is_ok(),
        "Failed to load ESM2 650M model: {:?}",
        esm2.err()
    );
}

/// Test a simple embedding generation using ESM2
#[test]
#[ignore = "requires downloading model files"]
fn test_esm2_650m_embedding() {
    let esm2 = Esm2Runner::load_model(Esm2Models::ESM2_650M, device().unwrap())
        .expect("Failed to load ESM2 650M model");
    let sequence = TEST_SEQUENCE;
    let embedding = esm2.get_embedding(sequence);
    assert!(
        embedding.is_ok(),
        "Failed to get embedding: {:?}",
        embedding.err()
    );
    let embedding = embedding.unwrap();
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    println!("Generated embedding with {} features", embedding.len());
}

/// Test sequence masking and prediction with ESM2
#[test]
#[ignore = "requires downloading model files"]
fn test_esm2_650m_masked_prediction() {
    let esm2 = Esm2Runner::load_model(Esm2Models::ESM2_650M, device().unwrap())
        .expect("Failed to load ESM2 650M model");
    
    // Generate a masked sequence (replace some amino acids with masks)
    let original_sequence = TEST_SEQUENCE;
    let masked_positions = vec![5, 10, 15, 20]; // Example positions to mask
    let masked_sequence = esm2.create_masked_sequence(original_sequence, &masked_positions)
        .expect("Failed to create masked sequence");
    
    // Get predictions for the masked positions
    let predictions = esm2.predict_masked_positions(&masked_sequence);
    assert!(
        predictions.is_ok(),
        "Failed to get predictions for masked positions: {:?}",
        predictions.err()
    );
    
    let predictions = predictions.unwrap();
    assert!(
        !predictions.is_empty(),
        "Predictions for masked positions should not be empty"
    );
    println!("Generated {} predictions for masked positions", predictions.len());
    
    // Verify that we got predictions for each masked position
    assert_eq!(
        predictions.len(),
        masked_positions.len(),
        "Should have predictions for each masked position"
    );
}

/// Test protein classification with ESM2
#[test]
#[ignore = "requires downloading model files"]
fn test_esm2_650m_classification() {
    let esm2 = Esm2Runner::load_model(Esm2Models::ESM2_650M, device().unwrap())
        .expect("Failed to load ESM2 650M model");
    let sequence = TEST_SEQUENCE;
    
    // Get classification scores
    let classification = esm2.classify_protein(sequence);
    assert!(
        classification.is_ok(),
        "Failed to classify protein: {:?}",
        classification.err()
    );
    
    let classes = classification.unwrap();
    assert!(!classes.is_empty(), "Classification results should not be empty");
    println!("Generated {} classification scores", classes.len());
    
    // Verify classification scores format
    for class_score in &classes {
        assert!(
            !class_score.class_name.is_empty(),
            "Class name should not be empty"
        );
        assert!(
            !class_score.score.is_nan(),
            "Score should be a valid number"
        );
    }
}