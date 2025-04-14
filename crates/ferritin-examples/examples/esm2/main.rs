use anyhow::{Error as E, Result};
// use candle_core::safetensors::load;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use ferritin_plms::{ESM2, ESM2Config as Config, device};
use hf_hub::{Repo, RepoType, api::sync::Api};
use tokenizers::Tokenizer;
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
    /// Path to a protein FASTA file
    #[arg(long)]
    protein_fasta: Option<std::path::PathBuf>,
}

impl Args {
    fn build_model_and_tokenizer(&self, device: &Device) -> Result<(ESM2, Tokenizer)> {
        let (model_id, revision, config) = match self.model_id.as_str() {
            "8M" => (
                "facebook/esm2_t6_8M_UR50D",
                "main",
                Config::esm2_t6_8M_ur50(),
            ),
            "35M" => (
                "facebook/esm2_t12_35M_UR50D",
                "main",
                Config::esm2_t12_35M_ur50(),
            ),
            "150M" => (
                "facebook/esm2_t30_150M_UR50D",
                "main",
                Config::esm2_t30_150M_ur50(),
            ),
            "650M" => (
                "facebook/esm2_t33_650M_UR50D",
                "main",
                Config::esm2_t33_650M_ur50(),
            ),
            "3B" => (
                "facebook/esm2_t36_3B_UR50D",
                "main",
                Config::esm2_t36_3b_ur50(),
            ),
            "15B" => (
                "facebook/esm2_t48_15B_UR50D",
                "main",
                Config::esm2_t48_15b_ur50(),
            ),
            _ => panic!("Invalid ESM models."),
        };
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let api = Api::new()?;
        let api = api.repo(repo);
        let weights = api.get("model.safetensors")?;
        // let tensors = load(&weights, &device)?;
        // for (name, tensor) in tensors.iter() {
        //     println!("Name: {}, Shape: {:?}", name, tensor.shape());
        // }
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &device)? };
        let tokenizer = ESM2::load_tokenizer()?;
        let model = ESM2::load(vb, config)?;
        println!("Loaded!");
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("Loading the Model and Tokenizer.......");
    let device = device()?;
    let (model, tokenizer) = args.build_model_and_tokenizer(&device)?;
    let protein_sequences = if let Some(seq) = args.protein_string {
        vec![seq]
    } else if let Some(fasta_path) = args.protein_fasta {
        todo!("fasta processing unimplimented")
        // std::fs::read_to_string(fasta_path)?
    } else {
        return Err(E::msg(
            "Either protein_string or protein_fasta must be provided",
        ));
    };
    for prot in protein_sequences.iter() {
        let tokens = tokenizer
            .encode(prot.to_string(), false)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        println!("Encoding.......");
        let encoded = model.forward(&token_ids)?;
        println!("Predicting.......");
        let predictions = encoded.logits.argmax(D::Minus1)?;
        println!("predictions: {:?}", predictions);
        println!("Input string: {}", prot);
        println!("Decoding.......");
        let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        let decoded = tokenizer.decode(indices.as_slice(), true);
        if let Ok(decoded_str) = &decoded {
            println!("Decoded output: {:?}", decoded_str.replace(" ", ""));
        } else {
            println!("Decoding failed!");
        }
        // Calculate similarity between input and output
        if let Ok(decoded_str) = decoded {
            let decoded_str = decoded_str.replace(" ", "");
            let input_chars: Vec<char> = prot.chars().collect();
            let output_chars: Vec<char> = decoded_str.chars().collect();
            let min_len = std::cmp::min(input_chars.len(), output_chars.len());
            let matches = input_chars
                .iter()
                .zip(output_chars.iter())
                .take(min_len)
                .filter(|(a, b)| a == b)
                .count();
            let similarity = if min_len > 0 {
                matches as f32 / min_len as f32
            } else {
                0.0
            };
            println!(
                "Similarity score: {:.2} ({} matching out of {})",
                similarity, matches, min_len
            );
        }
    }
    Ok(())
}
