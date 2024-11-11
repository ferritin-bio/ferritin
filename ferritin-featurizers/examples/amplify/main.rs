use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use ferritin_featurizers::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::SafeTensors;

fn main() -> Result<()> {
    // Setup HF API and model info
    let model_id = "chandar-lab/AMPLIFY_120M";
    let revision = "main";

    // Initialize the Hugging Face API client
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    // Load and analyze the safetensors file
    let weights_path = repo.get("model.safetensors")?;
    let weights = std::fs::read(&weights_path)?;
    let tensors = SafeTensors::deserialize(&weights)?;

    // Print all tensor names and their metadata
    println!("Model tensors:");
    tensors.names().iter().for_each(|tensor_name| {
        if let Ok(tensor_info) = tensors.tensor(tensor_name) {
            println!(
                "Tensor: {:<44}  ||  Shape: {:?}",
                tensor_name,
                tensor_info.shape(),
            );
        }
    });

    // https://github.com/huggingface/candle/blob/main/candle-examples/examples/clip/main.rs#L91C1-L92C101
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &Device::Cpu)?
    };
    let config = AMPLIFYConfig::default().amp_120m();
    let model = AMPLIFY::load(vb, &config)?;
    println!("Successfully created the model!");
    let tokenizer = repo.get("tokenizer.json")?;
    let protein_tokenizer = ProteinTokenizer::new(tokenizer)?;
    println!("Successfully created the tokenizer!");
    let pmatrix = protein_tokenizer.encode(&["METVAL".to_string()], Some(20), true, false)?;
    let pmatrix = pmatrix.unsqueeze(0)?; // [batch, length] <- add batch of 1 in this case
    println!("Successfully encoded the protein!");
    // begin encoding the model....
    println!("Commence Encoding:");
    let encoded = model.forward(&pmatrix, None, false, false)?;
    println!("{:?}", encoded.logits);

    Ok(())
}
