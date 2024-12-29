use anyhow::Result;
use candle_core::Device;
use ferritin_core::{AtomCollection, StructureFeatures};
use ferritin_onnx_models::{
    ndarray_to_tensor_f32, tensor_to_ndarray_f32, tensor_to_ndarray_i64, LigandMPNN,
    LigandMPNNModels,
};
use ferritin_test_data::TestFile;

use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

fn main() -> Result<()> {
    println!("Loading the Model and Tokenizer.......");
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let (pdb, _) = pdbtbx::open(protfile).expect("PDB/CIF");
    let ac = AtomCollection::from(&pdb);
    let logits = LigandMPNN::run_model(ac, 10, 0.1)?;
    println!("{:?}", logits);

    Ok(())
}
