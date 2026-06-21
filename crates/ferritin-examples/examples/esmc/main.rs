//! ESMC sequence embedding example.
//!
//! Downloads ESMC-300M (or 600M / 6B) from HuggingFace and embeds a protein
//! sequence, printing the embedding shape and first logit values.
//!
//! ```shell
//! cargo run --example esmc
//! cargo run --example esmc -- --model-id 600M
//! cargo run --example esmc -- --model-id 6B --cpu
//! cargo run --example esmc --features metal
//! ```

use anyhow::{Error as E, Result};
use clap::Parser;
use ferritin_plms::{ESMCModels, ESMCRunner, device};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Which ESMC model to use: "300M", "600M", or "6B".
    #[arg(long, value_parser = ["300M", "600M", "6B"], default_value = "300M")]
    model_id: String,

    /// Protein sequence to embed. Defaults to GFP.
    #[arg(long)]
    protein_string: Option<String>,
}

// GFP — the canonical demo sequence from the biohub/ESMC model card.
const DEFAULT_SEQUENCE: &str = concat!(
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCF",
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE",
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE",
    "KRDHMVLLEFVTAAGITHGMDELYK"
);

fn main() -> Result<()> {
    let args = Args::parse();

    let model_enum = match args.model_id.as_str() {
        "300M" => ESMCModels::ESMC300M,
        "600M" => ESMCModels::ESMC600M,
        "6B" => ESMCModels::ESMC6B,
        other => return Err(E::msg(format!("Unknown model id: {other}"))),
    };

    println!("Loading ESMC-{}...", args.model_id);
    let device = device(args.cpu)?;
    let runner = ESMCRunner::from_pretrained(model_enum, device)?;

    let sequence = args.protein_string.unwrap_or_else(|| {
        println!("No --protein-string provided, using GFP (default demo sequence).");
        DEFAULT_SEQUENCE.to_string()
    });

    println!("Embedding sequence of length {}...", sequence.len());
    let embeddings = runner.embed_sequence(&sequence)?;

    let shape = embeddings.shape();
    println!("Embedding shape: {:?}", shape.dims());

    // Print first few values of the first residue embedding
    let first_residue = embeddings.get(0)?.get(0)?;
    let vals: Vec<f32> = first_residue.to_vec1()?;
    println!(
        "First residue embedding (first 8 values): {:?}",
        &vals[..8.min(vals.len())]
    );

    Ok(())
}
