//! Test the ESM2 models
//!
//! This test file validates loading and running the ESM2 models,
//! particularly the ESM2-650M model.

// cargo test test_load_esm2_8m --features metal -- --ignored
// cargo test test_esm2_150m_embedding --features metal -- --ignored

use anyhow::Result;
use ferritin_plms::device;
use ferritin_plms::{ESM2Models, ESM2Runner};

const TEST_SEQUENCE: &str = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";
const HAMMING_CUTOFF: f32 = 0.7;

// The Hamming similarity ratio as a float
fn hamming(s1: &str, s2: &str) -> f32 {
    // if s1.len() != s2.len() {
    //     panic!(
    //         "Strings must be of equal length to calculate Hamming similarity ratio.
    //         S1: {},
    //         S2: {},
    //         ",
    //         s1, s2
    //     );
    // }
    let matches = s1
        .chars()
        .zip(s2.chars())
        .filter(|(c1, c2)| c1 == c2)
        .count();
    matches as f32 / s1.len() as f32
}

/// Test that we can successfully load the ESM2 650M model
#[test]
#[ignore = "requires downloading model files"]
fn test_load_esm2_8m() -> Result<()> {
    let esm2 = ESM2Runner::load_model(ESM2Models::T6_8M, device()?)?;
    let output = esm2.run_forward(TEST_SEQUENCE)?;
    let output_sequence = esm2.decode_logits(output)?;
    let hamming_dist = hamming(TEST_SEQUENCE, &output_sequence);
    println!("Hamming Dist: {:?}", hamming_dist);
    assert!(hamming_dist > HAMMING_CUTOFF);
    Ok(())
}

/// Test a simple embedding generation using ESM2
#[test]
#[ignore = "requires downloading model files"]
fn test_esm2_150m_embedding() -> Result<()> {
    let esm2 = ESM2Runner::load_model(ESM2Models::T30_150M, device()?)?;
    let output = esm2.run_forward(TEST_SEQUENCE)?;
    let output_sequence = esm2.decode_logits(output)?;
    let hamming_dist = hamming(TEST_SEQUENCE, &output_sequence);
    println!("Hamming Dist: {:?}", hamming_dist);
    assert!(hamming_dist > HAMMING_CUTOFF);
    Ok(())
}
