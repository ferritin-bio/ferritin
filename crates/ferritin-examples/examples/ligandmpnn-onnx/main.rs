use anyhow::Result;
use ferritin_core::load_structure;
use ferritin_onnx_models::LigandMPNN;
use ferritin_test_data::TestFile;

fn main() -> Result<()> {
    println!("Loading the Model and Tokenizer.......");
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let ac = load_structure(protfile).unwrap();
    let model = LigandMPNN::new()?;
    let logits = model.run_model(ac, 10, 0.1)?;
    println!("{:?}", logits);

    Ok(())
}
