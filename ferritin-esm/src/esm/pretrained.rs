use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::models::{
    esm3::ESM3,
    esmc::ESMC,
    function_decoder::FunctionTokenDecoder,
    tokenization::get_model_tokenizers,
    vqvae::{StructureTokenDecoder, StructureTokenEncoder},
};

use crate::constants::{
    ESM3_FUNCTION_DECODER_V0, ESM3_OPEN_SMALL, ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0, ESMC_300M, ESMC_600M,
};

type ModelBuilder = Box<dyn Fn(&Device) -> Result<Box<dyn Model>>>;

pub fn esm3_structure_encoder_v0(device: &Device) -> Result<Box<dyn Model>> {
    let model = StructureTokenEncoder::new(1024, 1, 128, 2, 128, 4096)?;
    model.eval();
    let state_dict = Tensor::load(
        data_root("esm3").join("data/weights/esm3_structure_encoder_v0.safetensors"),
        device,
    )?;
    model.load_state_dict(&state_dict)?;
    Ok(Box::new(model))
}

pub fn esm3_structure_decoder_v0(device: &Device) -> Result<Box<dyn Model>> {
    let model = StructureTokenDecoder::new(1280, 20, 30)?;
    model.eval();
    let state_dict = Tensor::load(
        data_root("esm3").join("data/weights/esm3_structure_decoder_v0.safetensors"),
        device,
    )?;
    model.load_state_dict(&state_dict)?;
    Ok(Box::new(model))
}

pub fn esm3_function_decoder_v0(device: &Device) -> Result<Box<dyn Model>> {
    let model = FunctionTokenDecoder::new()?;
    model.eval();
    let state_dict = Tensor::load(
        data_root("esm3").join("data/weights/esm3_function_decoder_v0.safetensors"),
        device,
    )?;
    model.load_state_dict(&state_dict)?;
    Ok(Box::new(model))
}

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

pub fn esmc_600m_202412(device: &Device) -> Result<Box<dyn Model>> {
    let tokenizer = get_model_tokenizers(ESM3_OPEN_SMALL)?.sequence;
    let model = ESMC::new(1152, 18, 36, tokenizer)?;
    model.eval();
    let state_dict = Tensor::load(
        data_root("esmc-600").join("data/weights/esmc_600m_2024_12_v0.safetensors"),
        device,
    )?;
    model.load_state_dict(&state_dict)?;
    Ok(Box::new(model))
}

pub fn esm3_sm_open_v0(device: &Device) -> Result<Box<dyn Model>> {
    let tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)?;
    let model = ESM3::new(
        1536,
        24,
        256,
        48,
        esm3_structure_encoder_v0,
        esm3_structure_decoder_v0,
        esm3_function_decoder_v0,
        tokenizers,
    )?;
    model.eval();
    let state_dict = Tensor::load(
        data_root("esm3").join("data/weights/esm3_sm_open_v1.safetensors"),
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
