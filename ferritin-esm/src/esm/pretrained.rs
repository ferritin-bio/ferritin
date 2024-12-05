use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::esm::models::{esmc::ESMC, tokenization::get_model_tokenizers};

use crate::esm::utils::constants::models::{ESM3_OPEN_SMALL, ESMC_300M};

// from huggingface_hub import snapshot_download

// @staticmethod
// @cache
// def data_root(model: str):
//     if "INFRA_PROVIDER" in os.environ:
//         return Path("")
//     # Try to download from hugginface if it doesn't exist
//     if model.startswith("esm3"):
//         path = Path(snapshot_download(repo_id="EvolutionaryScale/esm3-sm-open-v1"))
//     elif model.startswith("esmc-300"):
//         path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-300m-2024-12"))
//     elif model.startswith("esmc-600"):
//         path = Path(snapshot_download(repo_id="EvolutionaryScale/esmc-600m-2024-12"))
//     else:
//         raise ValueError(f"{model=} is an invalid model name.")
//     return path

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

lazy_static! {
    static ref LOCAL_MODEL_REGISTRY: HashMap<&'static str, ModelBuilder> = {
        let mut map = HashMap::new();
        map.insert(ESM3_OPEN_SMALL, Box::new(esm3_sm_open_v0));
        map.insert(
            ESM3_STRUCTURE_ENCODER_V0,
            Box::new(esm3_structure_encoder_v0),
        );
        map.insert(
            ESM3_STRUCTURE_DECODER_V0,
            Box::new(esm3_structure_decoder_v0),
        );
        map.insert(ESM3_FUNCTION_DECODER_V0, Box::new(esm3_function_decoder_v0));
        map.insert(ESMC_600M, Box::new(esmc_600m_202412));
        map.insert(ESMC_300M, Box::new(esmc_300m_202412));
        map
    };
}

pub fn load_local_model(model_name: &str, device: &Device) -> Result<Box<dyn Model>> {
    LOCAL_MODEL_REGISTRY
        .get(model_name)
        .ok_or_else(|| Error::ModelNotFound(model_name.to_string()))?(device)
}

pub fn register_local_model(model_name: &'static str, model_builder: ModelBuilder) {
    LOCAL_MODEL_REGISTRY.insert(model_name, model_builder);
}
