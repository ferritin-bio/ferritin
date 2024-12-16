use anyhow::{Error as E, Result};
use candle_core::safetensors::load;
use candle_core::{DType, Tensor, D};
use candle_examples::device;
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use clap::Parser;
use ferritin_plms::{ESM2Config as Config, ESM2};
use tokenizers::Tokenizer;

pub const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

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
    fn build_model_and_tokenizer(&self) -> Result<(ESM2, Tokenizer)> {
        // fn build_model_and_tokenizer(&self) -> Result<((), Tokenizer)> {
        let device = device(self.cpu)?;
        let (model_id, revision) = match self.model_id.as_str() {
            "8M" => ("facebook/esm2_t6_8M_UR50D", "main"),
            "35M" => ("facebook/esm2_t12_35M_UR50D", "main"),
            "150M" => ("facebook/esm2_t30_150M_UR50D", "main"),
            "650M" => ("facebook/esm2_t33_650M_UR50D", "main"),
            "3B" => ("facebook/esm2_t36_3B_UR50D", "main"),
            "15B" => ("facebook/esm2_t48_15B_UR50D", "main"),
            _ => panic!("Invalid ESM models."),
        };
        let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
        let (config_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let weights = api.get("model.safetensors")?;
            (config, weights)
        };
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_str = config_str
            .replace("SwiGLU", "swiglu")
            .replace("Swiglu", "swiglu");

        // Now you can iterate through the tensors
        let tensors = load(&weights_filename, &device)?;
        for (name, tensor) in tensors.iter() {
            println!("Name: {}, Shape: {:?}", name, tensor.shape());
        }

        let config: Config = serde_json::from_str(&config_str)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let tokenizer = ESM2::load_tokenizer()?;
        let protein = self.protein_string.as_ref().unwrap().as_str();
        let encoded = tokenizer.encode(protein, false);

        println!("Encoded.... and.....");
        let model = ESM2::load(vb, &config)?;
        println!("Loaded!");

        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading the Model and Tokenizer.......");
    let (model, tokenizer) = args.build_model_and_tokenizer()?;

    // let device = &model.get_device();
    let device = device(false)?;

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
        // let encoded = model.forward(&token_ids, None, false, false)?;

        // println!("Predicting.......");
        // let predictions = encoded.logits.argmax(D::Minus1)?;

        // println!("Decoding.......");
        // let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
        // let decoded = tokenizer.decode(indices.as_slice(), true);

        // println!("Decoded: {:?}, ", decoded);
    }

    Ok(())
}
