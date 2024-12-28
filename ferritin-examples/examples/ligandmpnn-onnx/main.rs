use anyhow::{Error as E, Result};
use biomodel_base::{LigandMPNN, LigandMPNNModels};
use candle_core::Device;
use clap::Parser;
use ferritin_core::{AtomCollection, StructureFeatures};
use ferritin_test_data::TestFile;
use ndarray::{Array, Array2, Array4};
use ndarray_safetensors::parse_tensors;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use safetensors::{serialize, SafeTensors};
use std::env;

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

    // Coords: background 4:
    // ligand_coords: -1,-1,-1 3
    // ligand types. btach sequence, num atoms
    // ligand_mask
    //
    // https://github.com/zachcp/ferritin/blob/main/ferritin-plms/src/ligandmpnn/ligandmpnn/configs.rs#L82
    println!("Loading the Model and Tokenizer.......");
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let (pdb, _) = pdbtbx::open(protfile).expect("PDB/CIF");
    let ac = AtomCollection::from(&pdb);
    let s = ac
        .encode_amino_acids(&device)
        .expect("A complete convertion to locations");

    let x_bb = ac.to_numeric_backbone_atoms(&device)?;
    println!("XBB: {:?}", x_bb);
    let data = [("name", x_bb)];
    let st = serialize(data, &None)?;
    let tensors = safetensors::SafeTensors::deserialize(&st).unwrap();
    let arrays = parse_tensors::<f32>(&tensors).unwrap();
    println!("{:?}", arrays);

    // let lig_coord = ac.to_numeric_ligand_atoms()?;
    let (lig_coords_array, lig_elements_array, lig_mask_array) =
        ac.to_numeric_ligand_atoms(&device)?;

    // let data = x_bb.to_vec4()?;
    // let x_bb_nd = Array::try_from(x_bb);

    // let outputs = model.run(ort::inputs![
    //     "coords" => arrays,
    //     // "ligand_coords" => lig_coords_array,
    //     // "ligand_types" => lig_elements_array,
    //     // "ligand_mask" => lig_mask_array
    // ]?)?;

    println!("Starting the Outputs...");
    // Run inference
    // Error: Invalid rank for input:
    //  coords Got: 2 Expected:
    //  4 Please fix either the inputs/outputs or the model.
    println!("Model is run!");

    // Print outputs
    // for (name, tensor) in outputs.iter() {
    //     println!("Output {}: {:?}", name, tensor);
    // }
    // let res_idx = ac.get_res_index();
    // let res_idx_len = res_idx.len();
    // let res_idx_tensor = Tensor::from_vec(res_idx, (1, res_idx_len), &device)?;
    // coords = x_37
    // ligand_coord =
    // // # Prepare inputs for ONNX
    // ort_inputs = {
    //     'coords': coords.cpu().numpy(),
    //     'ligand_coords': ligand_coords.cpu().numpy(),
    //     'ligand_types': ligand_types.cpu().numpy(),
    //     'ligand_mask': ligand_mask.cpu().numpy()
    // }
    // println!("{}", protfile);

    // # Prepare inputs for ONNX
    // ort_inputs = {
    //     'coords': coords.cpu().numpy(),
    //     'ligand_coords': ligand_coords.cpu().numpy(),
    //     'ligand_types': ligand_types.cpu().numpy(),
    //     'ligand_mask': ligand_mask.cpu().numpy()
    // }

    // # Run ONNX inference
    // ort_outputs = ort_session.run(None, ort_inputs)

    // # Compare outputs
    // print("\nComparing PyTorch and ONNX outputs:")
    // torch_outputs = [h_V, h_E, E_idx]
    // for torch_out, onnx_out, name in zip(torch_outputs, ort_outputs, ['h_V', 'h_E', 'E_idx']):
    //     max_diff = np.abs(torch_out.cpu().numpy() - onnx_out).max()
    //     print(f"{name} max difference: {max_diff:.6f}")

    // let tokenizer = ESM2::load_tokenizer()?;
    // let protein = args.protein_string.as_ref().unwrap().as_str();
    // let tokens = tokenizer
    //     .encode(protein.to_string(), false)
    //     .map_err(E::msg)?
    //     .get_ids()
    //     .iter()
    //     .map(|&x| x as i64)
    //     .collect::<Vec<_>>();

    // // since we are taking a single string we set the first <batch> dimension == 1.
    // let shape = (1, tokens.len());
    // let mask_array: Array2<i64> = Array2::from_shape_vec(shape, vec![0; tokens.len()])?;
    // let tokens_array: Array2<i64> = Array2::from_shape_vec(shape, tokens)?;

    // // Input name: input_ids
    // // Input type: Tensor { ty: Int64, dimensions: [-1, -1], dimension_symbols: [Some("batch_size"), Some("sequence_length")] }
    // // Input name: attention_mask
    // // Input type: Tensor { ty: Int64, dimensions: [-1, -1], dimension_symbols: [Some("batch_size"), Some("sequence_length")] }
    // for input in &model.inputs {
    //     println!("Input name: {}", input.name);
    //     println!("Input type: {:?}", input.input_type);
    // }
    // let outputs =
    //     model.run(ort::inputs!["input_ids" => tokens_array,"attention_mask" => mask_array]?)?;
    // // Print output names and shapes
    // // Output name: logits
    // for (name, tensor) in outputs.iter() {
    //     println!("Output name: {}", name);
    //     if let Ok(tensor) = tensor.try_extract_tensor::<f32>() {
    //         //     <Batch> <SeqLength> <Vocab>
    //         // Shape: [1, 256, 33]
    //         println!("Shape: {:?}", tensor.shape());
    //         println!(
    //             "Sample values: {:?}",
    //             &tensor.view().as_slice().unwrap()[..5]
    //         ); // First 5 values
    //     }
    // }
    Ok(())
}
