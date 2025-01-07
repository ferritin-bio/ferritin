use anyhow::Result;
use ferritin_core::AtomCollection;
use ferritin_onnx_models::LigandMPNN;
use ferritin_test_data::TestFile;
use pdbtbx::{Format, ReadOptions};
use std::io::BufReader;
use std::io::Read;
// use wonnx;

fn process_pdb_bytes(pdb_bytes: &[u8]) -> Result<Vec<f32>> {
    let reader = BufReader::new(pdb_bytes);
    let (pdb, _error) = ReadOptions::default()
        .set_format(Format::Mmcif)
        .read_raw(reader)
        .expect("Failed to parse PDB/CIF");
    let ac = AtomCollection::from(&pdb);
    let model = LigandMPNN::new()?;
    let logits = model.run_model(ac, 10, 0.1)?;
    let logits = logits.to_vec1()?;
    Ok(logits)
}

fn main() -> Result<()> {
    println!("Loading the Model and Tokenizer.......");

    let (protfile, mut handle) = TestFile::protein_01().create_temp()?;
    // println!("{:?}", handle.bytes());

    // Read the entire file into a Vec<u8>
    let mut pdb_bytes = Vec::new();
    handle.read_to_end(&mut pdb_bytes)?;

    let logits = process_pdb_bytes(&pdb_bytes);

    Ok(())
}
