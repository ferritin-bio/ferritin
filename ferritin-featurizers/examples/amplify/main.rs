use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::SafeTensors;
use std::path::PathBuf;

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

    // Download the safetensors file
    let weights_path = repo.get("model.safetensors")?;

    // Load and analyze the safetensors file
    let weights = std::fs::read(&weights_path)?;
    let tensors = SafeTensors::deserialize(&weights)?;

    // Print all tensor names and their metadata
    println!("Model tensors:");
    for tensor_name in tensors.names() {
        if let Ok(tensor_info) = tensors.tensor(tensor_name) {
            println!("Tensor: {}", tensor_name);
            println!("  Shape: {:?}", tensor_info.shape());
            println!("  DType: {:?}", tensor_info.dtype());
            println!("---");
        }
    }

    Ok(())
}
