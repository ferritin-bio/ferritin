//! ESMFold2 structure-prediction demo.
//!
//! **Currently non-functional.** ESMFold2 pretrained weights cannot be loaded:
//! the ported architecture does not match the released `biohub/ESMFold2-Fast`
//! checkpoint, so `from_pretrained` refuses (ferritin-100.16). This example
//! parses and reports the input, then prints that refusal and exits — it does
//! not write an mmCIF file, because any coordinates it produced would be
//! meaningless. See `docs/decisions/esmfold2-port-mismatch.md`.
//!
//! ```shell
//! cargo run --example esmfold2 -p ferritin-examples
//! cargo run --example esmfold2 -p ferritin-examples -- --sequence MKTAYIAKQRQISFVKSHFSRQLEERLK
//! cargo run --example esmfold2 -p ferritin-examples -- --out my_protein.cif
//! cargo run --example esmfold2 -p ferritin-examples --features metal
//! ```

use anyhow::Result;
use clap::Parser;
use ferritin_plms::{
    ESMFold2Models, ESMFold2Runner, ProteinInput, StructurePredictionInput, device,
};

#[derive(Parser, Debug)]
#[command(about = "ESMFold2 structure-prediction demo")]
struct Args {
    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    /// Protein sequence to fold (default: ubiquitin).
    #[arg(
        long,
        default_value = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )]
    sequence: String,

    /// Output mmCIF file path.
    #[arg(long, default_value = "esmfold2_pred.cif")]
    out: String,

    /// Recycling iterations.
    #[arg(long, default_value_t = 1)]
    num_loops: usize,

    /// Diffusion denoising steps (14 = fast, 50 = quality).
    #[arg(long, default_value_t = 14)]
    num_sampling_steps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;

    let protein = ProteinInput::new("A", &args.sequence)
        .map_err(|e| anyhow::anyhow!("Invalid sequence: {e}"))?;
    let input = StructurePredictionInput::new().add_protein(protein);
    println!("Input:    {input}");

    println!("Loading ESMFold2-Fast…");
    match ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device) {
        Ok(_runner) => {
            // Unreachable while the port is mismatched; kept so this example
            // starts working again the moment loading is restored.
            unreachable!(
                "ESMFold2 loading unexpectedly succeeded — restore the folding \
                 path in this example (ferritin-100.16)"
            );
        }
        Err(e) => {
            eprintln!("\nESMFold2 cannot fold sequences yet:\n\n{e}\n");
            eprintln!(
                "Requested: {} residues, loops={}, steps={} → would have written '{}'.",
                args.sequence.len(),
                args.num_loops,
                args.num_sampling_steps,
                args.out
            );
            eprintln!(
                "No mmCIF was written: coordinates from a mismatched checkpoint \
                 would be meaningless."
            );
        }
    }

    Ok(())
}
