use anyhow::{Error as E, Result};
use candle_core::Device;
use clap::Parser;
use ferritin_core::{AtomCollection, StructureFeatures};
use ferritin_onnx_models::{LigandMPNN, LigandMPNNModels};
use ferritin_test_data::TestFile;
use ndarray_safetensors::parse_tensors;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use safetensors::{serialize, SafeTensors};
use std::env;

fn tensor_to_ndarray_f32(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
    let tmp_data = [("_", tensor)];
    let st = serialize(tmp_data, &None)?;
    let tensors = SafeTensors::deserialize(&st).unwrap();
    let arrays = parse_tensors::<f32>(&tensors).unwrap();
    Ok(arrays.into_iter().next().unwrap().1)
}

fn tensor_to_ndarray_i64(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
    let tmp_data = [("_", tensor)];
    let st = serialize(tmp_data, &None)?;
    let tensors = SafeTensors::deserialize(&st).unwrap();
    let arrays = parse_tensors::<i64>(&tensors).unwrap();
    Ok(arrays.into_iter().next().unwrap().1)
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Which ESM2 Model to use
    #[arg(long, value_parser = ["8M", "35M", "150M", "650M", "3B", "15B"], default_value = "35M")]
    model_id: String,

    /// Protein String
    #[arg(long)]
    protein_string: Option<String>,

    /// Path to a protein FASTA file
    #[arg(long)]
    protein_fasta: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let args = Args::parse();
    let base_path = env::current_dir()?;
    ort::init()
        .with_name("LigandMPNN")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;
    let lmpnn_model = LigandMPNNModels::LigandMPNN;
    let (encoder_path, decoder_path) = LigandMPNN::load_model_path(lmpnn_model)?;
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(encoder_path)?;

    // Print input information
    println!("\nInputs:");
    for input in &model.inputs {
        println!("Name: {}, Type: {:#?}", input.name, input);
    }

    // https://github.com/zachcp/ferritin/blob/main/ferritin-plms/src/ligandmpnn/ligandmpnn/configs.rs#L82
    println!("Loading the Model and Tokenizer.......");
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let (pdb, _) = pdbtbx::open(protfile).expect("PDB/CIF");
    let ac = AtomCollection::from(&pdb);
    let s = ac
        .encode_amino_acids(&device)
        .expect("A complete convertion to locations");
    let x_bb = ac.to_numeric_backbone_atoms(&device)?;
    let (lig_coords_array, lig_elements_array, lig_mask_array) =
        ac.to_numeric_ligand_atoms(&device)?;
    let data_nd = tensor_to_ndarray_f32(x_bb)?;
    let lig_coords_array_nd = tensor_to_ndarray_f32(lig_coords_array)?;
    let lig_elements_array_nd = tensor_to_ndarray_i64(lig_elements_array)?;
    let lig_mask_array_nd = tensor_to_ndarray_f32(lig_mask_array)?;

    let outputs = model.run(ort::inputs![
        "coords" => data_nd,
        "ligand_coords" => lig_coords_array_nd,
        "ligand_types" => lig_elements_array_nd,
        "ligand_mask" => lig_mask_array_nd
    ]?)?;

    println!("Starting the Outputs...");
    println!("Model is run!");

    // Print outputs
    for (name, tensor) in outputs.iter() {
        println!("Output {}: {:#?}", name, tensor);
    }

    Ok(())
}
