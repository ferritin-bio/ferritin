use anyhow::{Error as E, Result};
use clap::Parser;
use ferritin_plms::ESM2;
use ndarray::Array2;
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
};
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
    let args = Args::parse();
    let base_path = env::current_dir()?;
    let model_path = base_path
        .join("ferritin-examples")
        .join("examples")
        .join("esm2-onnx")
        .join("esm2_t6_8M_UR50D_onnx")
        .join("model.onnx");

    // Create the ONNX Runtime environment, enabling CUDA execution providers for all sessions created in this process.
    ort::init()
        .with_name("ESM2")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    println!("Current directory: {:?}", env::current_dir()?);
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .commit_from_file(model_path)?;

    println!("Loading the Model and Tokenizer.......");
    let tokenizer = ESM2::load_tokenizer()?;

    let protein = args.protein_string.as_ref().unwrap().as_str();
    let tokens = tokenizer
        .encode(protein.to_string(), false)
        .map_err(E::msg)?
        .get_ids()
        .iter()
        .map(|&x| x as i64)
        .collect::<Vec<_>>();

    // since we are taking a single string we set the first <batch> dimension == 1.
    let shape = (1, tokens.len());
    let mask_array: Array2<i64> = Array2::from_shape_vec(shape, vec![0; tokens.len()])?;
    let tokens_array: Array2<i64> = Array2::from_shape_vec(shape, tokens)?;

    // Input name: input_ids
    // Input type: Tensor { ty: Int64, dimensions: [-1, -1], dimension_symbols: [Some("batch_size"), Some("sequence_length")] }
    // Input name: attention_mask
    // Input type: Tensor { ty: Int64, dimensions: [-1, -1], dimension_symbols: [Some("batch_size"), Some("sequence_length")] }
    for input in &model.inputs {
        println!("Input name: {}", input.name);
        println!("Input type: {:?}", input.input_type);
    }
    let outputs =
        model.run(ort::inputs!["input_ids" => tokens_array,"attention_mask" => mask_array]?)?;
    // Print output names and shapes
    // Output name: logits
    for (name, tensor) in outputs.iter() {
        println!("Output name: {}", name);
        if let Ok(tensor) = tensor.try_extract_tensor::<f32>() {
            //     <Batch> <SeqLength> <Vocab>
            // Shape: [1, 256, 33]
            println!("Shape: {:?}", tensor.shape());
            println!(
                "Sample values: {:?}",
                &tensor.view().as_slice().unwrap()[..5]
            ); // First 5 values
        }
    }
    Ok(())
}
