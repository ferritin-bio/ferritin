use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_examples::device;
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use clap::Parser;
use ferritin_plms::{AMPLIFYConfig as Config, AmplifyModels, AmplifyRunner, AMPLIFY};
use tokenizers::Tokenizer;

pub const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long, default_value = "false")]
    cpu: bool,

    /// Which AMPLIFY Model to use, either '120M' or '350M'.
    #[arg(long, value_parser = ["120M", "350M"], default_value = "120M")]
    model_id: String,

    /// Protein String
    #[arg(long)]
    protein_string: Option<String>,

    /// Path to a protein FASTA file
    #[arg(long)]
    protein_fasta: Option<std::path::PathBuf>,
}

impl Args {
    // fn build_model_and_tokenizer(&self) -> Result<(AMPLIFY, Tokenizer)> {
    //     let device = device(self.cpu)?;
    //     let (model_id, revision) = match self.model_id.as_str() {
    //         "120M" => ("chandar-lab/AMPLIFY_120M", "main"),
    //         "350M" => ("chandar-lab/AMPLIFY_350M", "main"),
    //         _ => panic!("Amplify models are either `120M` or `350M`"),
    //     };
    //     let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string());
    //     let (config_filename, tokenizer_filename, weights_filename) = {
    //         let api = Api::new()?;
    //         let api = api.repo(repo);
    //         let config = api.get("config.json")?;
    //         let tokenizer = api.get("tokenizer.json")?;
    //         let weights = api.get("model.safetensors")?;
    //         (config, tokenizer, weights)
    //     };
    //     let config_str = std::fs::read_to_string(config_filename)?;
    //     let config_str = config_str
    //         .replace("SwiGLU", "swiglu")
    //         .replace("Swiglu", "swiglu");
    //     let config: Config = serde_json::from_str(&config_str)?;
    //     let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    //     let vb =
    //         unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    //     let model = AMPLIFY::load(vb, &config)?;
    //     Ok((model, tokenizer))
    // }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = device(args.cpu)?;
    let amprunner = AmplifyRunner::load_model(AmplifyModels::AMP120M, device)?;

    let prot_sequence = args.protein_string.unwrap();

    // Runs the model and returns the full, manipulateable result
    let outputs = amprunner.run_forward(&prot_sequence);
    // Runs the model and returns the top hit from each logit
    let top_hit = amprunner.get_best_prediction(&prot_sequence);
    // Runs the model and returns the top probabilities
    let get_probabilities = amprunner.get_pseudo_probabilities(&prot_sequence);
    // Runs the model and returns the contactmap
    let contact_map = amprunner.get_contact_map(&prot_sequence);

    Ok(())
}
