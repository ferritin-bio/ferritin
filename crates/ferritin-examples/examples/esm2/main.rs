use anyhow::{Error as E, Result};
use candle_core::DType;
use clap::Parser;
use ferritin_plms::{ESM2Models, ESM2Runner, device};
pub const DTYPE: DType = DType::F32;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Which ESM2 Model to use
    #[arg(long, value_parser = ["8M", "35M", "150M", "650M", "3B", "15B"], default_value = "35M")]
    model_id: String,
    /// Protein String
    #[arg(long)]
    protein_string: Option<String>,
    // /// Path to a protein FASTA file
    // #[arg(long)]
    // protein_fasta: Option<std::path::PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Loading the Model and Tokenizer.......");
    let model_enum = match args.model_id.as_str() {
        "8M" => ESM2Models::T6_8M,
        "35M" => ESM2Models::T12_35M,
        "150M" => ESM2Models::T30_150M,
        "650M" => ESM2Models::T33_650M,
        "3B" => ESM2Models::T36_3B,
        "15B" => ESM2Models::T48_15B,
        _ => return Err(E::msg("Invalid model ID")),
    };
    let modelrunner = ESM2Runner::load_model(model_enum, device()?)?;
    println!("Encoding.......");
    let prot_string = args
        .protein_string
        .expect("a protein sting must be provided");
    let output = modelrunner.run_forward(&prot_string)?;
    println!("Predicting.......");
    let output_sequence = modelrunner.decode_logits(output)?;
    println!("Decoded sequence: {:?}", output_sequence);

    Ok(())
}
