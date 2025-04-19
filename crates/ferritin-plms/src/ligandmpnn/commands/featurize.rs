use crate::StructureFeatures;
use candle_core;
use ferritin_core::load_structure;

// src/commands/command1.rs
pub fn execute(input: String, output: String) -> anyhow::Result<()> {
    let ac = load_structure(input)?;
    let features = ac.featurize_lmpnn(&candle_core::Device::Cpu)?;
    features.save_to_safetensor(&output)?;
    Ok(())
}
