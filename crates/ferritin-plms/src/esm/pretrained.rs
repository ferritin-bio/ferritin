use crate::esm::models::esmc::ESMC;
use crate::esm::tokenization::get_model_tokenizers;
use crate::esm::utils::constants::models::{ESM3_OPEN_SMALL, ESMC_300M};
use candle_core::{Device, Result, Tensor};
use candle_hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use std::path::PathBuf;

// use huggingface_hub::snapshot_download;

pub fn data_root(model: &str) -> PathBuf {
    if std::env::var("INFRA_PROVIDER").is_ok() {
        return PathBuf::new();
    }
    if model.starts_with("esm3") {
        PathBuf::from(snapshot_download("EvolutionaryScale/esm3-sm-open-v1"))
    } else if model.starts_with("esmc-300") {
        PathBuf::from(snapshot_download("EvolutionaryScale/esmc-300m-2024-12"))
    } else if model.starts_with("esmc-600") {
        PathBuf::from(snapshot_download("EvolutionaryScale/esmc-600m-2024-12"))
    } else {
        panic!("{model} is an invalid model name")
    }
}

type ModelBuilder = Box<dyn Fn(&Device) -> Result<Box<dyn Model>>>;

pub fn esmc_300m_202412(device: &Device) -> Result<Box<dyn Model>> {
    let tokenizer = get_model_tokenizers(ESM3_OPEN_SMALL)?.sequence;
    let model = ESMC::new(960, 15, 30, tokenizer)?;
    model.eval();
    let state_dict = Tensor::load(
        data_root("esmc-300").join("data/weights/esmc_300m_2024_12_v0.safetensors"),
        device,
    )?;
    model.load_state_dict(&state_dict)?;
    Ok(Box::new(model))
}

// lazy_static! {
//     static ref LOCAL_MODEL_REGISTRY: HashMap<&'static str, ModelBuilder> = {
//         let mut map = HashMap::new();
//         map.insert(ESM3_OPEN_SMALL, Box::new(esm3_sm_open_v0));
//         map.insert(
//             ESM3_STRUCTURE_ENCODER_V0,
//             Box::new(esm3_structure_encoder_v0),
//         );
//         map.insert(
//             ESM3_STRUCTURE_DECODER_V0,
//             Box::new(esm3_structure_decoder_v0),
//         );
//         map.insert(ESM3_FUNCTION_DECODER_V0, Box::new(esm3_function_decoder_v0));
//         map.insert(ESMC_600M, Box::new(esmc_600m_202412));
//         map.insert(ESMC_300M, Box::new(esmc_300m_202412));
//         map
//     };
// }

pub fn load_local_model(model_name: &str, device: &Device) -> Result<Box<dyn Model>> {
    LOCAL_MODEL_REGISTRY
        .get(model_name)
        .ok_or_else(|| Error::ModelNotFound(model_name.to_string()))?(device)
}

pub fn register_local_model(model_name: &'static str, model_builder: ModelBuilder) {
    LOCAL_MODEL_REGISTRY.insert(model_name, model_builder);
}
