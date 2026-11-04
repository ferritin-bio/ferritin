//! Rank amino acids at a masked antibody position with AbLang2.
//!
//! ```shell
//! cargo run --example ablang2 -- --chain heavy --position 99
//! # reuse an unpacked ablang2-weights.tar.gz instead of downloading it:
//! cargo run --example ablang2 -- --weights-dir /path/to/ablang2-weights
//! ```
//!
//! The pair is run jointly, so the light chain conditions the heavy-chain
//! prediction and vice versa. The output row is located through the
//! encoding's per-token roles, never by assuming a fixed special-token offset.

use anyhow::{Result, bail};
use candle_core::{D, IndexOp};
use clap::Parser;
use ferritin_plms::ablang2::tokenizer::{AMINO_ACIDS, Chain, TokenRole};
use ferritin_plms::loader::LoadOptions;
use ferritin_plms::{AbLang2Models, AbLang2Runner, device};
use std::path::PathBuf;

/// The VH/VL pair from the AbLang2 README.
const VH: &str = "EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS";
const VL: &str = "DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK";

#[derive(Parser, Debug)]
#[command(about = "Masked-residue ranking with AbLang2")]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
    /// Heavy chain (VH). Empty for a light-only input.
    #[arg(long, default_value = VH)]
    heavy: String,
    /// Light chain (VL). Empty for a heavy-only input.
    #[arg(long, default_value = VL)]
    light: String,
    /// Which chain the masked position is in.
    #[arg(long, value_parser = ["heavy", "light"], default_value = "heavy")]
    chain: String,
    /// 0-based position within that chain to mask and predict.
    #[arg(long, default_value_t = 99)]
    position: usize,
    /// How many candidates to print.
    #[arg(long, default_value_t = 5)]
    top: usize,
    /// Directory holding an unpacked ablang2-weights.tar.gz (model.pt and
    /// hparams.json). Without it the archive is downloaded from Zenodo.
    #[arg(long)]
    weights_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let opts = LoadOptions::new(device(args.cpu)?);
    let runner = match &args.weights_dir {
        Some(dir) => AbLang2Runner::from_dir(dir, &opts)?,
        None => AbLang2Runner::from_pretrained_with(AbLang2Models::Paired, &opts)?,
    };

    let chain = if args.chain == "heavy" {
        Chain::Heavy
    } else {
        Chain::Light
    };
    let target = match chain {
        Chain::Heavy => &args.heavy,
        Chain::Light => &args.light,
    };
    let Some(native) = target.chars().nth(args.position) else {
        bail!(
            "{} chain has {} residues; --position {} is out of range",
            args.chain,
            target.len(),
            args.position
        );
    };

    let mut masked: Vec<char> = target.chars().collect();
    masked[args.position] = '*';
    let masked: String = masked.into_iter().collect();
    let (heavy, light) = match chain {
        Chain::Heavy => (masked.as_str(), args.light.as_str()),
        Chain::Light => (args.heavy.as_str(), masked.as_str()),
    };

    let (logits, enc) = runner.logits_pair(heavy, light)?;
    let row = enc
        .roles
        .iter()
        .position(|r| {
            *r == TokenRole::Residue {
                chain,
                index: args.position,
            }
        })
        .expect("the masked residue has a token row");

    // Softmax over the twenty amino acids only (ids 1..=20).
    let probs = candle_nn::ops::softmax(&logits.i(row)?.narrow(D::Minus1, 1, 20)?, D::Minus1)?
        .to_vec1::<f32>()?;
    let mut ranked: Vec<(char, f32)> = AMINO_ACIDS.iter().copied().zip(probs).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!(
        "{} chain position {} (native {native}), token row {row}:",
        args.chain, args.position
    );
    for (aa, p) in ranked.iter().take(args.top) {
        let mark = if *aa == native { "  <- native" } else { "" };
        println!("  {aa}  {p:.4}{mark}");
    }
    println!(
        "confidence (mean log-prob of the unmasked pair): {:.4}",
        runner.confidence(&args.heavy, &args.light)?
    );
    Ok(())
}
