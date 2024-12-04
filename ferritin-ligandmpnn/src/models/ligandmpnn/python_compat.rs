//! Serialize python Pytorch models to CANDLE-ready formats
//!
//! This module should be able to consume the JSON/PT data that is created by
//! the scripts in the ligandmpnn repo.

use candle_core::{Device, Shape, Tensor};
use serde::{Deserialize, Deserializer, Serialize};
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum ChainL {
    C,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LigandMPNNData {
    pub output_dict: LigandMPNNDataDict,
    backbone: String,
    other_atoms: String,
    ca_icodes: Vec<String>,
    water_atoms: String,
    // [[0, 1, 14], [10,11,14,15], [20, 21]]
    pub symmetry_residues: Option<Vec<Vec<i64>>>,
    // [[1.0, 1.0, 1.0], [-2.0,1.1,0.2,1.1], [2.3, 1.1]]
    pub symmetry_weights: Option<Vec<Vec<f64>>>,
    homo_oligomer: Option<bool>,
    pub batch_size: Option<i64>,
}

impl LigandMPNNData {
    pub fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let ligandmpnn: LigandMPNNData = serde_json::from_reader(reader)?;
        Ok(ligandmpnn)
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct LigandMPNNDataDict {
    #[serde(rename = "X", deserialize_with = "deserialize_tensor_3d")]
    pub x: Tensor,
    #[serde(deserialize_with = "deserialize_tensor_1d")]
    pub mask: Tensor,
    #[serde(rename = "Y")]
    y: Vec<Vec<f64>>,
    #[serde(rename = "Y_t", deserialize_with = "deserialize_tensor_1d")]
    y_t: Tensor,
    #[serde(rename = "Y_m", deserialize_with = "deserialize_tensor_1d")]
    y_m: Tensor,
    #[serde(rename = "R_idx", deserialize_with = "deserialize_tensor_1d")]
    pub r_idx: Tensor,
    #[serde(deserialize_with = "deserialize_tensor_1d")]
    pub chain_labels: Tensor,
    pub chain_letters: Vec<String>,
    pub mask_c: Vec<Vec<bool>>,
    chain_list: Vec<ChainL>,
    #[serde(rename = "S", deserialize_with = "deserialize_tensor_1d")]
    pub s: Tensor,
    #[serde(deserialize_with = "deserialize_tensor_3d")]
    xyz_37: Tensor,
    #[serde(rename = "xyz_37_m", deserialize_with = "deserialize_tensor_2d")]
    xyz_37__m: Tensor,

    // put these here temporarily
    bias_aa: Option<Tensor>,
    bias_aa_per_residue: Option<Tensor>,
    omit_aa_per_residue_multi: Option<Tensor>,
}

impl LigandMPNNDataDict {
    /// Tensor of chains to keep
    pub fn get_chain_mask(
        &self,
        chains_to_design: Vec<String>,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let chains = self
            .chain_letters
            .iter()
            .map(|item| chains_to_design.contains(item) as u32);
        Tensor::from_iter(chains, &device)
    }
}

fn deserialize_tensor_1d<'de, D>(deserializer: D) -> Result<Tensor, D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<i64> = Vec::deserialize(deserializer)?;
    let shape = Shape::from_dims(&[vec.len()]);

    Tensor::from_vec(vec, shape, &Device::Cpu).map_err(serde::de::Error::custom)
}
fn deserialize_tensor_2d<'de, D>(deserializer: D) -> Result<Tensor, D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<Vec<f64>> = Vec::deserialize(deserializer)?;
    if vec.is_empty() {
        return Err(serde::de::Error::custom("Empty 2D vector"));
    }
    let shape = Shape::from_dims(&[vec.len(), vec[0].len()]);
    let flattened: Vec<f64> = vec.into_iter().flatten().collect();
    Tensor::from_vec(flattened, shape, &Device::Cpu).map_err(serde::de::Error::custom)
}
fn deserialize_tensor_3d<'de, D>(deserializer: D) -> Result<Tensor, D::Error>
where
    D: Deserializer<'de>,
{
    let vec: Vec<Vec<Vec<f64>>> = Vec::deserialize(deserializer)?;
    let shape = Shape::from_dims(&[
        vec.len(),
        vec.first().map_or(0, |v| v.len()),
        vec.first().and_then(|v| v.first()).map_or(0, |v| v.len()),
    ]);
    let flattened: Vec<f64> = vec
        .into_iter()
        .flat_map(|v| v.into_iter().flat_map(|w| w.into_iter()))
        .collect();
    Tensor::from_vec(flattened, shape, &Device::Cpu).map_err(serde::de::Error::custom)
}
