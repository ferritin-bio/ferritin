use anyhow::{Error as E, Result};
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
    let device = Device::Cpu;
    let lmpnn_model = LigandMPNNModels::LigandMPNN;
    let (encoder_path, decoder_path) = LigandMPNN::load_model_path(lmpnn_model)?;

    ort::init()
        .with_name("LigandMPNN")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let encoder_model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(encoder_path)?;

    // https://github.com/zachcp/ferritin/blob/main/ferritin-plms/src/ligandmpnn/ligandmpnn/configs.rs#L82

    println!("Loading the Model and Tokenizer.......");
    let (protfile, _handle) = TestFile::protein_01().create_temp()?;
    let (pdb, _) = pdbtbx::open(protfile).expect("PDB/CIF");
    let ac = AtomCollection::from(&pdb);

    println!("Creating the input Tensors.......");
    let x_bb = ac.to_numeric_backbone_atoms(&device)?;
    let (lig_coords_array, lig_elements_array, lig_mask_array) =
        ac.to_numeric_ligand_atoms(&device)?;
    let data_nd = tensor_to_ndarray_f32(x_bb)?;
    let lig_coords_array_nd = tensor_to_ndarray_f32(lig_coords_array)?;
    let lig_elements_array_nd = tensor_to_ndarray_i64(lig_elements_array)?;
    let lig_mask_array_nd = tensor_to_ndarray_f32(lig_mask_array)?;

    println!("Runnning the Encoder Model.......");
    let encoder_outputs = encoder_model.run(ort::inputs![
        "coords" => data_nd,
        "ligand_coords" => lig_coords_array_nd,
        "ligand_types" => lig_elements_array_nd,
        "ligand_mask" => lig_mask_array_nd
    ]?)?;

    println!("Spinning up the Dccoder Model.......");
    let decoder_model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(decoder_path)?;

    println!("Creating the Inpute to the Decoder.......");
    let h_V = encoder_outputs["h_V"].try_extract_tensor::<f32>()?;
    let h_E = encoder_outputs["h_E"].try_extract_tensor::<f32>()?;
    let E_idx = encoder_outputs["E_idx"].try_extract_tensor::<i64>()?;
    let position_tensor = {
        let data = vec![10 as i64]; // Single value
        let array = ndarray::Array::from_shape_vec([1], data)?; // Shape [1]
        Tensor::from_array(array)?
    };

    println!("Temp and Position Are Hardcoded........");
    let temp_tensor = {
        let data = vec![0.1 as f32]; // Single value
        let array = ndarray::Array::from_shape_vec([1], data)?
        Tensor::from_array(array)?
    };

    let decoder_outputs = decoder_model.run(ort::inputs![
        "h_V" => h_V,
        "h_E" => h_E,
        "E_idx" => E_idx,
        "position" => position_tensor,
        "temperature" => temp_tensor,
    ]?)?;

    println!("Decoder Outputs are logits.");
    let logits = decoder_outputs["logits"]
        .try_extract_tensor::<f32>()?
        .to_owned();

    println!("Converted to Candle.");
    let logit_tensor = ndarray_to_tensor_f32(logits);
    println!("{:?}", logit_tensor);

    Ok(())
}
