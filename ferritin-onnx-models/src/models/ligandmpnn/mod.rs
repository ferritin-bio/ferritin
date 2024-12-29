//! ESM2 Struct. Loads the hf tokenizer
//!
use crate::{ndarray_to_tensor_f32, tensor_to_ndarray_f32, tensor_to_ndarray_i64};
use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_hf_hub::api::sync::Api;
use ferritin_core::{AtomCollection, StructureFeatures};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
};
use std::path::PathBuf;

pub enum LigandMPNNModels {
    ProteinMPNN,
    LigandMPNN,
}

pub struct LigandMPNN {}

impl LigandMPNN {
    pub fn load_model_path(model: LigandMPNNModels) -> Result<(PathBuf, PathBuf)> {
        let api = Api::new().unwrap();
        let (repo_id, encoder_filename, decoder_filename) = match model {
            LigandMPNNModels::ProteinMPNN => (
                "zcpbx/proteinmpnn-v48-030-onnx".to_string(),
                "protmpnn_encoder.onnx",
                "protmpnn_decoder_step.onnx",
            ),
            LigandMPNNModels::LigandMPNN => (
                "zcpbx/ligandmpnn-v32-030-25-onnx".to_string(),
                "ligand_encoder.onnx",
                "ligand_decoder.onnx",
            ),
        };
        let encoder_path = api.model(repo_id.clone()).get(encoder_filename).unwrap();
        let decoder_path = api.model(repo_id).get(decoder_filename).unwrap();
        Ok((encoder_path, decoder_path))
    }
    /// Ac -> Logit Tensor
    pub fn run_model(ac: AtomCollection, position: i64, temperature: f32) -> Result<Tensor> {
        ort::init()
            .with_name("LigandMPNN")
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .commit()?;

        let session_config = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(1)?;

        let (encoder_path, decoder_path) =
            LigandMPNN::load_model_path(LigandMPNNModels::LigandMPNN)?;
        let encoder_model = session_config.clone().commit_from_file(&encoder_path)?;
        let decoder_model = session_config.clone().commit_from_file(&decoder_path)?;

        // https://github.com/zachcp/ferritin/blob/main/ferritin-plms/src/ligandmpnn/ligandmpnn/configs.rs#L82
        let device = Device::Cpu;
        let x_bb = ac.to_numeric_backbone_atoms(&device)?;
        let (lig_coords_array, lig_elements_array, lig_mask_array) =
            ac.to_numeric_ligand_atoms(&device)?;
        let data_nd = tensor_to_ndarray_f32(x_bb)?;
        let lig_coords_array_nd = tensor_to_ndarray_f32(lig_coords_array)?;
        let lig_elements_array_nd = tensor_to_ndarray_i64(lig_elements_array)?;
        let lig_mask_array_nd = tensor_to_ndarray_f32(lig_mask_array)?;

        let encoder_outputs = encoder_model.run(ort::inputs![
            "coords" => data_nd,
            "ligand_coords" => lig_coords_array_nd,
            "ligand_types" => lig_elements_array_nd,
            "ligand_mask" => lig_mask_array_nd
        ]?)?;
        let h_V = encoder_outputs["h_V"].try_extract_tensor::<f32>()?;
        let h_E = encoder_outputs["h_E"].try_extract_tensor::<f32>()?;
        let E_idx = encoder_outputs["E_idx"].try_extract_tensor::<i64>()?;
        let position_tensor = {
            let data = vec![position];
            let array = ndarray::Array::from_shape_vec([1], data)?;
            ort::value::Tensor::from_array(array)?
        };
        let temp_tensor = {
            let data = vec![temperature];
            let array = ndarray::Array::from_shape_vec([1], data)?;
            ort::value::Tensor::from_array(array)?
        };
        let decoder_outputs = decoder_model.run(ort::inputs![
            "h_V" => h_V,
            "h_E" => h_E,
            "E_idx" => E_idx,
            "position" => position_tensor,
            "temperature" => temp_tensor,
        ]?)?;

        let logits = decoder_outputs["logits"]
            .try_extract_tensor::<f32>()?
            .to_owned();

        let logit_tensor = ndarray_to_tensor_f32(logits)?;

        Ok(logit_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
