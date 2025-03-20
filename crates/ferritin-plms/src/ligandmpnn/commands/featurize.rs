use crate::ligandmpnn::proteinfeatures::LMPNNFeatures;
use candle_core;
use ferritin_core::{AtomCollection, load_structure};

// src/commands/command1.rs
pub fn execute(input: String, output: String) -> anyhow::Result<()> {
    let ac = load_structure(input).unwrap();
    let features = ac.featurize(&candle_core::Device::Cpu)?;
    features.save_to_safetensor(&output)?;
    Ok(())
}
