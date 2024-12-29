use anyhow::Result;
use ferritin_core::AtomCollection;
use ferritin_onnx_models::LigandMPNN;
use ferritin_test_data::TestFile;

fn main() -> Result<()> {
    println!("Loading the Model and Tokenizer.......");
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let (pdb, _) = pdbtbx::open(protfile).expect("PDB/CIF");
    let ac = AtomCollection::from(&pdb);
    let model = LigandMPNN::new().unwrap();
    let logits = model.run_model(ac, 10, 0.1).unwrap();
    println!("{:?}", logits);

    Ok(())
}
