//! Protein Featurizer for ProteinMPNN/LignadMPNN
//!
//! Extract protein features for ligandmpnn
//!
//! Returns a set of features calculated from protein structure
//! including:
//! - Residue-level features like amino acid type, secondary structure
//! - Geometric features like distances, angles
//! - Chemical features like hydrophobicity, charge
//! - Evolutionary features from MSA profiles
use crate::featurize::utilities::aa1to_int;
use candle_core::{Device, Result, Tensor};
use std::collections::{HashMap, HashSet};

#[allow(dead_code)]
pub struct ProteinFeatures {
    /// protein amino acids sequences as 1D Tensor of u32
    pub(crate) s: Tensor,
    /// protein co-oords by residue [batch, seqlength, 37, 3]
    pub(crate) x: Tensor,
    /// protein mask by residue
    pub(crate) x_mask: Option<Tensor>,
    /// ligand coords
    pub(crate) y: Tensor,
    /// encoded ligand atom names
    pub(crate) y_t: Tensor,
    /// ligand mask
    pub(crate) y_m: Option<Tensor>,
    /// R_idx:         Tensor dimensions: torch.Size([93])          # protein residue indices shape=[length]
    pub(crate) r_idx: Tensor,
    /// chain_labels:  Tensor dimensions: torch.Size([93])          # protein chain letters shape=[length]
    pub(crate) chain_labels: Option<Vec<f64>>,
    /// chain_letters: NumPy array dimensions: (93,)
    pub(crate) chain_letters: Vec<String>,
    /// mask_c:        Tensor dimensions: torch.Size([93])
    pub(crate) mask_c: Option<Tensor>,
    pub(crate) chain_list: Vec<String>,
}
impl ProteinFeatures {
    pub fn get_coords(&self) -> &Tensor {
        &self.x
    }
    pub fn get_sequence(&self) -> &Tensor {
        &self.s
    }
    pub fn get_sequence_mask(&self) -> Option<&Tensor> {
        self.x_mask.as_ref()
    }
    pub fn get_residue_index(&self) -> &Tensor {
        &self.r_idx
    }
    pub fn get_encoded(
        &self,
    ) -> Result<(Vec<String>, HashMap<String, usize>, HashMap<usize, String>)> {
        let r_idx_list = &self.r_idx.flatten_all()?.to_vec1::<u32>()?;
        let chain_letters_list = &self.chain_letters;
        let encoded_residues: Vec<String> = r_idx_list
            .iter()
            .enumerate()
            .map(|(i, r_idx)| format!("{}{}", chain_letters_list[i], r_idx))
            .collect();
        let encoded_residue_dict: HashMap<String, usize> = encoded_residues
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        let encoded_residue_dict_rev: HashMap<usize, String> = encoded_residues
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.clone()))
            .collect();
        Ok((
            encoded_residues,
            encoded_residue_dict,
            encoded_residue_dict_rev,
        ))
    }
    // Fixed Residue List --> Tensor of 1/0
    // Inputs: `"C1 C2 C3 C4 C5 C6 C7 C8 C9 C10`
    pub fn get_encoded_tensor(&self, fixed_residues: String, device: &Device) -> Result<Tensor> {
        let res_set: HashSet<String> = fixed_residues.split(' ').map(String::from).collect();
        let (encoded_res, _, _) = &self.get_encoded()?;
        Tensor::from_iter(
            encoded_res
                .iter()
                .map(|item| u32::from(!res_set.contains(item))),
            device,
        )
    }
    pub fn get_chain_mask_tensor(
        &self,
        chains_to_design: Vec<String>,
        device: &Device,
    ) -> Result<Tensor> {
        let mask_values: Vec<u32> = self
            .chain_letters
            .iter()
            .map(|chain| u32::from(chains_to_design.contains(chain)))
            .collect();
        Tensor::from_iter(mask_values, device)
    }
    pub fn update_mask(&mut self, tensor: Tensor) -> Result<()> {
        self.x_mask = match self.x_mask.as_ref() {
            Some(mask) => Some(mask.mul(&tensor)?),
            None => Some(tensor),
        };
        Ok(())
    }
    // Fixed Residue List --> Tensor of length 21
    // Inputs: `A:10.0"`
    pub fn create_bias_tensor(&self, bias_aa: Option<String>) -> Result<Tensor> {
        let device = self.s.device();
        let dtype = self.s.dtype();
        match bias_aa {
            None => Tensor::zeros(21, dtype, device),
            Some(bias_aa) => {
                let mut bias_values = vec![0.0f32; 21];
                for pair in bias_aa.split(',') {
                    if let Some((aa, value_str)) = pair.split_once(':') {
                        if let Ok(value) = value_str.parse::<f32>() {
                            if let Some(aa_char) = aa.chars().next() {
                                let idx = aa1to_int(aa_char) as usize;
                                bias_values[idx] = value;
                            }
                        }
                    }
                }
                Tensor::from_slice(&bias_values, 21, device)
            }
        }
    }
    pub fn save_to_safetensor(&self, path: &str) -> Result<()> {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        // this is only one field. need to do the rest of the fields
        tensors.insert("protein_atom_sequence".to_string(), self.s.clone());
        tensors.insert("protein_atom_positions".to_string(), self.x.clone());
        tensors.insert("ligand_atom_positions".to_string(), self.y.clone());
        tensors.insert("ligand_atom_name".to_string(), self.y_t.clone());
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }
}
