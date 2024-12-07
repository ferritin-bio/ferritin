use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor, D};
use candle_examples::device;
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use clap::Parser;
use ferritin_amplify::{AMPLIFYConfig as Config, AMPLIFY};
use tokenizers::Tokenizer;

pub const DTYPE: DType = DType::F32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,
    // /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    // #[arg(long)]
    // model_id: Option<String>,

    // #[arg(long)]
    // revision: Option<String>,

    // /// When set, compute embeddings for this prompt.
    // #[arg(long)]
    // prompt: Option<String>,

    // /// Use the pytorch weights rather than the safetensors ones
    // #[arg(long)]
    // use_pth: bool,

    // /// The number of times to run the prompt.
    // #[arg(long, default_value = "1")]
    // n: usize,

    // /// L2 normalization for embeddings.
    // #[arg(long, default_value = "true")]
    // normalize_embeddings: bool,

    // /// Use tanh based approximation for Gelu instead of erf implementation.
    // #[arg(long, default_value = "false")]
    // approximate_gelu: bool,
}

impl Args {
    fn build_model_and_tokenizer(&self) -> Result<(AMPLIFY, Tokenizer)> {
        let device = device(self.cpu)?;
        let default_model = "chandar-lab/AMPLIFY_120M".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = (default_model, default_revision);
        // let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
        //     (Some(model_id), Some(revision)) => (model_id, revision),
        //     (Some(model_id), None) => (model_id, "main".to_string()),
        //     (None, Some(revision)) => (default_model, revision),
        //     (None, None) => (default_model, default_revision),
        // };
        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };
        let config_str = std::fs::read_to_string(config_filename)?;
        let config_str = config_str
            .replace("SwiGLU", "swiglu")
            .replace("Swiglu", "swiglu");
        let config: Config = serde_json::from_str(&config_str)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
        let model = AMPLIFY::load(vb, &config)?;
        Ok((model, tokenizer))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (model, tokenizer) = args.build_model_and_tokenizer()?;
    let device = &model.get_device();
    let sprot_01 = "MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL";

    let tokens = tokenizer
        .encode(sprot_01.to_string(), false)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
    println!("Encoding.......");
    let encoded = model.forward(&token_ids, None, false, false)?;

    println!("Predicting.......");
    let predictions = encoded.logits.argmax(D::Minus1)?;

    println!("Decoding.......");
    let indices: Vec<u32> = predictions.to_vec2()?[0].to_vec();
    let decoded = tokenizer.decode(indices.as_slice(), true);

    println!("Decoded: {:?}, ", decoded);
    Ok(())
}
