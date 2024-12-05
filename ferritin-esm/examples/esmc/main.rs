use anyhow::Result;
use candle_core::{DType, Device, D};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use candle_nn::VarBuilder;
use ferritin_esm::{AMPLIFYConfig, ProteinTokenizer, AMPLIFY};
use safetensors::SafeTensors;

fn main() -> Result<()> {
    // https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12
    let model_id = "EvolutionaryScale/esmc-300m-2024-12";
    let revision = "main";
    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    let weights_path = repo.get("data/weights/esmc_300m_2024_12_v0.pth")?;

    // // Available for printing Tensor data....
    // let print_tensor_info = false;
    // if print_tensor_info {
    //     println!("Model tensors:");
    //     let weights = std::fs::read(&weights_path)?;
    //     let tensors = SafeTensors::deserialize(&weights)?;
    //     tensors.names().iter().for_each(|tensor_name| {
    //         if let Ok(tensor_info) = tensors.tensor(tensor_name) {
    //             println!(
    //                 "Tensor: {:<44}  ||  Shape: {:?}",
    //                 tensor_name,
    //                 tensor_info.shape(),
    //             );
    //         }
    //     });
    // }

    let pth = PthTensors::new(mpnn_file, Some("model_state_dict"))?;
    let vb = VarBuilder::from_backend(Box::new(pth), default_dtype, self.device.clone());

    println!("VarBuilder for ESMC: {:?}", vb)
}
