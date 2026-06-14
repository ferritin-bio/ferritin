//! ESMFold2 structure-prediction demo.
//!
//! Downloads ESMFold2-Fast weights from HuggingFace (~755 MB on first run,
//! cached locally thereafter) and folds the given protein sequence.
//!
//! Note: the ESMC-6B backbone is not yet loaded, so coordinates will not be
//! physically meaningful until that wiring is complete. The mmCIF output and
//! pLDDT tensors are exercised end-to-end regardless.
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

    println!("Loading ESMFold2-Fast (downloads ~755 MB on first run)…");
    let runner = ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device)?;

    println!(
        "Folding {} residues (loops={} steps={})…",
        args.sequence.len(),
        args.num_loops,
        args.num_sampling_steps
    );
    let output = runner.fold_protein(&args.sequence, args.num_loops, args.num_sampling_steps)?;

    let mmcif = output.to_mmcif(&args.sequence, "A", "esmfold2_pred")?;
    std::fs::write(&args.out, &mmcif)?;

    let atom_count = mmcif.lines().filter(|l| l.starts_with("ATOM")).count();
    println!("Wrote {atom_count} ATOM records → '{}'", args.out);

    println!("\n--- mmCIF preview ---");
    for line in mmcif.lines().take(20) {
        println!("{line}");
    }
    if mmcif.lines().count() > 20 {
        println!("… ({} total lines)", mmcif.lines().count());
    }

    Ok(())
}
