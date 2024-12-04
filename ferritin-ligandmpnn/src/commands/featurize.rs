use crate::ligandmpnn::protinfeatures::LMPNNFeatures;
use candle_core;
use ferritin_core::AtomCollection;
use pdbtbx;

// src/commands/command1.rs
pub fn execute(input: String, output: String) -> anyhow::Result<()> {
    let (pdb, _) = pdbtbx::open(input).expect("A PDB  or CIF file");
    let ac = AtomCollection::from(&pdb);
    let features = ac.featurize(&candle_core::Device::Cpu)?;
    let _ = features.save_to_safetensor(&output)?;
    Ok(())
}
