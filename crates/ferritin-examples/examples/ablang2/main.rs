//! Rank amino acids at a masked antibody position with AbLang2.
//!
//! ```shell
//! cargo run --example ablang2 -- --chain heavy --position 99
//! # TCRLang: the first chain is TRB (beta), the second TRA (alpha)
//! cargo run --example ablang2 -- --model tcrlang --chain heavy --position 95
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
/// The JM22 TCR from upstream's TCRLang notebook.
const TRB: &str = "GGITQSPKYLFRKEGQNVTLSCEQNLNHDAMYWYRQDPGQGLRLIYYSQIVNDFQKGDIAEGYSVSREKKESFPLTVTSAQKNPTAFYLCASSIRSSYEQYFGPGTRLTVTEDLKN";
const TRA: &str = "QLLEQSPQFLSIQEGENLTVYCNSSSVFSSLQWYRQEPGEGPVLLVTVVTGGEVKKLKRLTFQFGDARKDSSLHITAAQPGDTGLYLCAGAGSQGNLIFGKGTKLSVKP";

#[derive(Parser, Debug)]
#[command(about = "Masked-residue ranking with AbLang2")]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
    /// Which checkpoint: AbLang2 (antibodies) or TCRLang (TCRs).
    #[arg(long, value_parser = ["ablang2", "tcrlang"], default_value = "ablang2")]
    model: String,
    /// First chain: VH for AbLang2, TRB (beta) for TCRLang. Defaults to a
    /// demo pair; pass "" for a second-chain-only input.
    #[arg(long)]
    heavy: Option<String>,
    /// Second chain: VL for AbLang2, TRA (alpha) for TCRLang.
    #[arg(long)]
    light: Option<String>,
    /// Which chain the masked position is in ("heavy" = first slot).
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
    let mut args = Args::parse();
    let (model, demo) = match args.model.as_str() {
        "tcrlang" => (AbLang2Models::TcrLangPaired, (TRB, TRA)),
        _ => (AbLang2Models::Paired, (VH, VL)),
    };
    let heavy_in = args.heavy.take().unwrap_or_else(|| demo.0.to_string());
    let light_in = args.light.take().unwrap_or_else(|| demo.1.to_string());
    let opts = LoadOptions::new(device(args.cpu)?);
    let runner = match &args.weights_dir {
        Some(dir) => AbLang2Runner::from_dir(model, dir, &opts)?,
        None => AbLang2Runner::from_pretrained_with(model, &opts)?,
    };

    let chain = if args.chain == "heavy" {
        Chain::Heavy
    } else {
        Chain::Light
    };
    let target = match chain {
        Chain::Heavy => &heavy_in,
        Chain::Light => &light_in,
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
        Chain::Heavy => (masked.as_str(), light_in.as_str()),
        Chain::Light => (heavy_in.as_str(), masked.as_str()),
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
        runner.confidence(&heavy_in, &light_in)?
    );
    Ok(())
}
