use anyhow::Result;
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use ferritin_plms::{ESMCConfig, ESMC};

// pub fn esmc_300m_202412(device: &Device) -> Result<Box<dyn Model>> {
//     let tokenizer = get_model_tokenizers(ESM3_OPEN_SMALL)?.sequence;
//     let model = ESMC::new(960, 15, 30, tokenizer)?;
//     model.eval();
//     let state_dict = Tensor::load(
//         data_root("esmc-300").join("data/weights/esmc_300m_2024_12_v0.safetensors"),
//         device,
//     )?;
//     model.load_state_dict(&state_dict)?;
//     Ok(Box::new(model))
// }

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
    let pth = PthTensors::new(weights_path, None)?;

    // print the names
    for (name, tensor) in pth.tensor_infos() {
        println!("{}: {:?}", name, tensor);
    }

    let vb = VarBuilder::from_backend(Box::new(pth), DType::F32, Device::Cpu);
    let config = ESMCConfig::esmc_300m();
    let esmc = ESMC::load(vb.clone(), config)?;
    // println!("ESMC Loaded: {}", esmc);

    // Error: cannot find tensor transformer.layer.attention.layer_norm.weight
    println!(
        "VB: {}",
        vb.contains_tensor("transformer.blocks.6.attn.layernorm_qkv.1.weight")
    );

    Ok(())
}
