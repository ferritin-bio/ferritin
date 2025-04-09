use anyhow::Result;
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use ferritin_core::load_structure;
use ferritin_plms::{LMPNNFeatures, ProteinMPNN, ProteinMPNNConfig, device};
use ferritin_test_data::TestFile;

fn main() -> Result<()> {
    println!("Loading the Model and Tokenizer.......");

    let deviceA = device(false)?;
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let ac = load_structure(protfile)?;

    let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
    let pth = PthTensors::new(mpnn_file, Some("model_state_dict"))?;
    let vb = VarBuilder::from_backend(Box::new(pth), DType::F32, deviceA.clone());
    let pconf = ProteinMPNNConfig::proteinmpnn();
    let pmpnn = ProteinMPNN::load(vb, &pconf)?;

    // let features = ac.featurize(&candle_core::Device::Cpu)?;
    let features = ac.featurize(&deviceA.clone())?;
    println!("Features");
    let encoded = pmpnn.encode(&features)?;
    println!("Encoded");
    // let model = LigandMPNN::new()?;
    // let logits = model.run_model(ac, 10, 0.1)?;
    // println!("{:?}", logits);

    Ok(())
}
