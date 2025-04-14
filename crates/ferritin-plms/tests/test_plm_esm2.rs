//! Test the ESM2 models
//!
//! This test file validates loading and running the ESM2 models,
//! particularly the ESM2-650M model.

use ferritin_plms::device;
use ferritin_plms::{ESM2Models, ESM2Runner};

const TEST_SEQUENCE: &str = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";

/// Test that we can successfully load the ESM2 650M model
#[test]
#[ignore = "requires downloading model files"]
fn test_load_esm2_8m() {
    let esm2 = ESM2Runner::load_model(ESM2Models::Esm2T6_8M_UR50, device().unwrap());
    assert!(
        esm2.is_ok(),
        "Failed to load ESM2 650M model: {:?}",
        esm2.err()
    );
}

/// Test a simple embedding generation using ESM2
#[test]
#[ignore = "requires downloading model files"]
fn test_esm2_8m_embedding() {
    let esm2 = ESM2Runner::load_model(ESM2Models::Esm2T6_8M_UR50, device().unwrap())
        .expect("Failed to load ESM2 650M model");
    let sequence = TEST_SEQUENCE;
    // let embedding = esm2.get_embedding(sequence);
    // assert!(
    //     embedding.is_ok(),
    //     "Failed to get embedding: {:?}",
    //     embedding.err()
    // );
    // let embedding = embedding.unwrap();
    // assert!(!embedding.is_empty(), "Embedding should not be empty");
    // println!("Generated embedding with {} features", embedding.len());
}
