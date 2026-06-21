//! ESM3 sequence embedding example.
//!
//! Downloads esm3-sm-open-v1 (1.4B params) from HuggingFace and embeds a protein
//! sequence, printing the embedding shape and first values of the first residue.
//!
//! **Note**: ESM3 weights are gated. Before running:
//! 1. Accept the Cambrian Non-Commercial license at
//!    <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1>
//! 2. Run `huggingface-cli login` (or set the `HF_TOKEN` environment variable)
//!
//! ```shell
//! cargo run --example esm3
//! cargo run --example esm3 -- --protein-string ACDEFGHIKLM
//! cargo run --example esm3 --features metal
//! ```

use anyhow::Result;
use clap::Parser;
use ferritin_plms::{ESM3Models, ESM3Runner, device};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    /// Protein sequence to embed. Defaults to GFP.
    #[arg(long)]
    protein_string: Option<String>,
}

// GFP — the canonical demo sequence.
const DEFAULT_SEQUENCE: &str = concat!(
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCF",
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE",
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE",
    "KRDHMVLLEFVTAAGITHGMDELYK"
);

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading ESM3 sm-open-v1 (1.4B)...");
    let device = device(args.cpu)?;
    let runner = ESM3Runner::from_pretrained(ESM3Models::SmOpen, device)?;

    let sequence = args.protein_string.unwrap_or_else(|| {
        println!("No --protein-string provided; using GFP (default demo sequence).");
        DEFAULT_SEQUENCE.to_string()
    });

    println!("Embedding sequence of length {}...", sequence.len());
    let embeddings = runner.embed_sequence(&sequence)?;

    let shape = embeddings.shape();
    println!("Embedding shape: {:?}", shape.dims());
    // Shape should be (1, L+2, 1536): batch=1, L+2 includes BOS+EOS, d_model=1536

    let first_residue = embeddings.get(0)?.get(1)?; // index 1 = first real residue (skip BOS)
    let vals: Vec<f32> = first_residue.to_vec1()?;
    println!(
        "First residue embedding (first 8 dims): {:?}",
        &vals[..8.min(vals.len())]
    );

    Ok(())
}
