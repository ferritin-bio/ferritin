//! ESMFold2 structure-prediction demo.
//!
//! The ESMFold2 forward pass is not yet fully implemented, so this demo builds a
//! synthetic ESMFold2Output (random-walk Cα trace) and exercises the full
//! input → output → mmCIF pipeline that will be used once the model is ready.
//!
//! When `fold_protein` is implemented, swap the `synthetic_output` call for:
//!
//! ```rust,ignore
//! let runner = ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device)?;
//! let output = runner.fold_protein(&args.sequence, 3, 14)?;
//! ```
//!
//! ```shell
//! cargo run --example esmfold2
//! cargo run --example esmfold2 -- --sequence MKTAYIAKQRQISFVKSHFSRQLEERLK
//! cargo run --example esmfold2 -- --out my_protein.cif
//! cargo run --example esmfold2 --features metal
//! ```

use anyhow::Result;
use candle_core::{Device, Tensor};
use clap::Parser;
use ferritin_plms::{
    ESMFold2Config, ESMFold2Output,
    esmfold2::{ProteinInput, StructurePredictionInput},
    device,
};

#[derive(Parser, Debug)]
#[command(about = "ESMFold2 structure-prediction demo (synthetic output)")]
struct Args {
    /// Run on CPU rather than GPU.
    #[arg(long)]
    cpu: bool,

    /// Protein sequence to fold.
    #[arg(long, default_value = "MKTAYIAKQRQISFVKSHFSRQLEERLKK")]
    sequence: String,

    /// Output mmCIF file path.
    #[arg(long, default_value = "esmfold2_demo.cif")]
    out: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;

    // --- 1. Validate input and print summary ---
    let protein = ProteinInput::new("A", &args.sequence)
        .map_err(|e| anyhow::anyhow!("Invalid sequence: {e}"))?;

    let input = StructurePredictionInput::new().add_protein(protein);
    println!("Input:    {input}");

    let config = ESMFold2Config::fast();
    println!(
        "Config:   d_single={} d_pair={} trunk_layers={} diffusion_steps={}",
        config.d_single,
        config.d_pair,
        config.trunk_n_layers,
        config.inference_num_steps,
    );

    let l = args.sequence.len();

    // --- 2. Build synthetic output (Cα-only random walk) ---
    //
    // Replace this with the real runner once the forward pass is ready:
    //
    //   let runner = ESMFold2Runner::from_pretrained(ESMFold2Models::Fast, device.clone())?;
    //   let output = runner.fold_protein(&args.sequence, 3, 14)?;
    //
    println!("Generating synthetic Cα trace ({l} residues)…");
    let output = synthetic_output(&args.sequence, &device)?;

    // --- 3. Convert to mmCIF and write ---
    let mmcif = output.to_mmcif(&args.sequence, "A", "esmfold2_demo")?;
    std::fs::write(&args.out, &mmcif)?;

    let atom_count = mmcif.lines().filter(|l| l.starts_with("ATOM")).count();
    println!("Wrote {atom_count} ATOM records → '{}'", args.out);

    // --- 4. Print a short preview ---
    println!("\n--- mmCIF preview ---");
    for line in mmcif.lines().take(20) {
        println!("{line}");
    }
    if mmcif.lines().count() > 20 {
        println!("… ({} total lines)", mmcif.lines().count());
    }

    Ok(())
}

/// Build a synthetic [`ESMFold2Output`] with a Cα-only random-walk backbone.
///
/// Each Cα–Cα step is 3.8 Å along X (ideal extended chain) with small random
/// perturbations in Y and Z. pLDDT is set to a constant 0.70.
fn synthetic_output(sequence: &str, device: &Device) -> Result<ESMFold2Output> {
    let l = sequence.len();

    let mut coords: Vec<f32> = Vec::with_capacity(l * 3);
    let mut x = 0.0_f32;
    let mut y = 0.0_f32;
    let mut z = 0.0_f32;
    let mut rng: u64 = 0xDEAD_BEEF_1234_5678;

    for _ in 0..l {
        x += 3.8;
        y += (lcg(&mut rng) - 0.5) * 2.0;
        z += (lcg(&mut rng) - 0.5) * 2.0;
        coords.extend_from_slice(&[x, y, z]);
    }

    Ok(ESMFold2Output {
        sample_atom_coords: Tensor::from_vec(coords, &[l, 3], device)?,
        plddt: Tensor::from_vec(vec![0.70_f32; l], l, device)?,
        ptm: Tensor::from_vec(vec![0.65_f32], 1_usize, device)?,
        iptm: Tensor::from_vec(vec![0.65_f32], 1_usize, device)?,
        pae: None,
        distogram_logits: None,
    })
}

/// Minimal deterministic LCG for fake coordinate noise.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}
