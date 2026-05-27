//! ESMC-300M inference example.
//!
//! # Usage
//!
//! **With real weights** (requires HuggingFace token with gated access to
//! `EvolutionaryScale/esmc-300m-2024-12`):
//!
//! ```shell
//! HF_TOKEN=<your_token> cargo run --example esmc
//! HF_TOKEN=<your_token> cargo run --example esmc -- --sequence ACDEFGHIKLMNPQRSTVWY
//! ```
//!
//! **Without weights** (zero-initialised, useful for CI / pipeline smoke-tests):
//!
//! ```shell
//! cargo run --example esmc -- --random-weights
//! cargo run --example esmc -- --random-weights --sequence MKTAYIA
//! ```

use anyhow::Result;
use candle_core::pickle::PthTensors;
use candle_core::{D, DType};
use candle_nn::VarBuilder;
use clap::Parser;
use ferritin_plms::{ESMC, ESMCConfig, LogitsConfig, device};
use hf_hub::{Repo, RepoType, api::sync::Api};

// ---------------------------------------------------------------------------
// ESM-C amino-acid vocabulary (matches SEQUENCE_VOCAB in esm3.rs)
// ---------------------------------------------------------------------------
const SEQUENCE_VOCAB: &[&str] = &[
    "<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "|", "<mask>",
];

/// Default demo sequence: human Ubiquitin (76 residues).
const DEFAULT_SEQUENCE: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
#[derive(Parser, Debug)]
#[command(
    name = "esmc",
    about = "Run ESMC-300M on a protein sequence.\n\n\
             With real weights: set HF_TOKEN and accept the EvolutionaryScale\n\
             gated license on HuggingFace before running.\n\
             Use --random-weights to skip downloading for a quick pipeline test."
)]
struct Args {
    /// Force CPU inference even when a GPU is available.
    #[arg(long)]
    cpu: bool,

    /// Amino-acid sequence to embed/score. Defaults to human Ubiquitin.
    #[arg(long)]
    sequence: Option<String>,

    /// Use zero-initialised weights instead of downloading from HuggingFace.
    /// Output is numerically meaningless but the full pipeline executes.
    #[arg(long)]
    random_weights: bool,
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;
    let config = ESMCConfig::esmc_300m();

    // -----------------------------------------------------------------------
    // 1. Build VarBuilder
    // -----------------------------------------------------------------------
    let vb: VarBuilder = if args.random_weights {
        println!("⚠  Using zero-initialised weights — predictions will be random.");
        VarBuilder::zeros(DType::F32, &device)
    } else {
        println!("Downloading ESMC-300m weights from HuggingFace…");
        println!("  repo: EvolutionaryScale/esmc-300m-2024-12");
        println!("  Requires HF_TOKEN env var with gated-model access.\n");

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            "EvolutionaryScale/esmc-300m-2024-12".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));
        let weights_path = repo.get("data/weights/esmc_300m_2024_12_v0.pth")?;
        println!("Weights cached at: {}", weights_path.display());
        let pth = PthTensors::new(weights_path, None)?;
        VarBuilder::from_backend(Box::new(pth), DType::F32, device.clone())
    };

    // -----------------------------------------------------------------------
    // 2. Instantiate model
    // -----------------------------------------------------------------------
    println!("Building ESMC model…");
    let model = ESMC::load(vb, config)?;
    println!("Model ready.\n");

    // -----------------------------------------------------------------------
    // 3. Encode the input sequence
    // -----------------------------------------------------------------------
    let sequence = args.sequence.unwrap_or_else(|| {
        println!("No --sequence provided; using default (human Ubiquitin).");
        DEFAULT_SEQUENCE.to_string()
    });

    println!("Input sequence ({} residues):", sequence.len());
    println!("  {}\n", sequence);

    // encode() returns a 1-D tensor of shape (L + 2,) with BOS and EOS.
    let tokens = model.encode(&sequence)?;
    println!("Token tensor: shape {:?}", tokens.shape());

    // -----------------------------------------------------------------------
    // 4. Forward pass — collect sequence logits + embeddings
    // -----------------------------------------------------------------------
    println!("Running forward pass…");
    let logits_cfg = LogitsConfig {
        sequence: true,
        return_embeddings: true,
        return_hidden_states: false,
    };
    let output = model.logits(&tokens, logits_cfg)?;
    println!("Forward pass complete.\n");

    // -----------------------------------------------------------------------
    // 5. Decode logits → top-1 amino acid per position
    // -----------------------------------------------------------------------
    if let Some(seq_logits) = &output.sequence_logits {
        // seq_logits: (1, L+2, vocab_size)
        println!("Sequence logits shape: {:?}", seq_logits.shape());

        let logits_2d = seq_logits.squeeze(0)?; // (L+2, vocab_size)
        let pred_ids = logits_2d.argmax(D::Minus1)?; // (L+2,)
        let pred_ids: Vec<u32> = pred_ids.to_vec1()?;

        // Strip BOS (pos 0) and EOS (last position)
        let inner = &pred_ids[1..pred_ids.len().saturating_sub(1)];

        let predicted: String = inner
            .iter()
            .map(|&id| SEQUENCE_VOCAB.get(id as usize).copied().unwrap_or("<?>"))
            .collect::<Vec<_>>()
            .join("");

        println!("Top-1 predicted sequence:");
        println!("  Input    : {}", sequence);
        println!("  Predicted: {}", predicted);

        // Identity / match rate
        let matches = sequence
            .chars()
            .zip(predicted.chars())
            .filter(|(a, b)| a == b)
            .count();
        let identity = if sequence.is_empty() {
            0.0
        } else {
            matches as f64 / sequence.len() as f64 * 100.0
        };
        println!(
            "  Identity : {}/{} ({:.1}%)\n",
            matches,
            sequence.len(),
            identity
        );
    }

    // -----------------------------------------------------------------------
    // 6. Embedding stats
    // -----------------------------------------------------------------------
    if let Some(embeddings) = &output.embeddings {
        // embeddings: (1, L+2, d_model)
        println!("Embedding tensor shape: {:?}", embeddings.shape());
        let emb = embeddings.squeeze(0)?; // (L+2, d_model)
        let mean: f32 = emb.mean_all()?.to_scalar()?;
        let var: f32 = emb
            .broadcast_sub(&emb.mean_all()?)?
            .sqr()?
            .mean_all()?
            .to_scalar()?;
        println!("Embedding mean : {:.6}", mean);
        println!("Embedding std  : {:.6}", var.sqrt());
    }

    println!("\nDone ✓");
    Ok(())
}
