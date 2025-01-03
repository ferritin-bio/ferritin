use anyhow::{Error as E, Result};
use clap::Parser;
use ferritin_onnx_models::{ESM2Models, ESM2};
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
    let esm_model = ESM2Models::ESM2_T6_8M;
    // let esm_model = ESM2Models::ESM2_T12_35M;
    // let esm_model = ESM2Models::ESM2_T30_150M;
    // let esm_model = ESM2Models::ESM2_T33_650M;
    let esm2 = ESM2::new(esm_model)?;
    let protein = args.protein_string.as_ref().unwrap().as_str();

    let logits = esm2.run_model(protein)?;
    let normed = esm2.extract_logits(&logits)?;
    Ok(())
}
