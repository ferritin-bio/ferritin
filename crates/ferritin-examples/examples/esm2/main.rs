use anyhow::{Error as E, Result};
use candle_core::DType;
use clap::Parser;
use ferritin_plms::{ESM2Models, ESM2Runner, device};
pub const DTYPE: DType = DType::F32;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
    /// Which ESM2 Model to use
    #[arg(long, value_parser = ["8M", "35M", "150M", "650M", "3B", "15B"], default_value = "35M")]
    model_id: String,
    /// Protein sequence to evaluate. Defaults to a short demo sequence.
    #[arg(long)]
    protein_string: Option<String>,
}

// A short real protein sequence (Ubiquitin, human) used as the default demo.
const DEFAULT_SEQUENCE: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

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
    let modelrunner = ESM2Runner::load_model(model_enum, device(args.cpu)?)?;
    println!("Encoding.......");
    let prot_string = args.protein_string.unwrap_or_else(|| {
        println!("No --protein-string provided, using default demo sequence.");
        DEFAULT_SEQUENCE.to_string()
    });
    let output = modelrunner.run_forward(&prot_string)?;
    println!("Predicting.......");
    let output_sequence = modelrunner.decode_logits(output)?;
    println!("Decoded sequence: {:?}", output_sequence);

    Ok(())
}
