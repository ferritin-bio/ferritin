use anyhow::Result;
use candle_core::Device;
use candle_nn::VarBuilder;
// use candle_transformers::models::bert::DTYPE;
use ferritin_featurizers::{AMPLIFYConfig, AMPLIFY};
use hf_hub::{api::sync::Api, Repo, RepoType};
use safetensors::{Dtype, SafeTensors};
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
            println!(
                "Tensor: {:<44}  ||  Shape: {:?}",
                tensor_name,
                tensor_info.shape(),
            );
        }
    }
    let weights_path = repo.get("model.safetensors")?;
    let vb = VarBuilder::from_buffered_safetensors(
        weights,
        // candle_core::safetensors::DType::F32,
        candle_core::DType::F32,
        &Device::Cpu,
    )?;

    // Pull a specific Tensor out of the variable builder...
    let Tensor1 = vb.get(&[3424, 640], "transformer_encoder.0.ffn.w12.weight")?;
    println!("Example Tensor Shape: {:?}", Tensor1.shape());
    // let amplify = AMPLIFY::new(config, vb)
    Ok(())
}
