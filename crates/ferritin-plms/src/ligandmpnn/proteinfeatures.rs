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
use super::utilities::{AAAtom, aa1to_int, aa3to1};
use crate::ligandmpnn::utilities::calculate_cb;
use candle_core::{DType, Device, Result, Tensor};
use ferritin_core::AtomCollection;
use ferritin_core::info::elements::Element;
use itertools::MultiUnzip;
use std::collections::{HashMap, HashSet};
use strum::IntoEnumIterator;

// Helper Fns --------------------------------------
fn is_heavy_atom(element: &Element) -> bool {
    !matches!(element, Element::H | Element::He)
}

/// Convert the AtomCollection into a struct that can be passed to a model.
pub trait LMPNNFeatures {
    fn encode_amino_acids(&self, device: &Device) -> Result<Tensor>; // ( residue types )
    fn featurize(&self, device: &Device) -> Result<ProteinFeatures>; // need more control over this featurization process
    fn get_res_index(&self) -> Vec<u32>;
    fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor>; // [residues, N/CA/C/O, xyz]
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor>; // [residues, N/CA/C/O....37, xyz]
    fn to_numeric_ligand_atoms(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)>; // ( positions , elements, mask )
    fn to_pdb(&self); //
}

/// Methods for Convering an AtomCollection into a LigandMPNN-ready
/// datasets
impl LMPNNFeatures for AtomCollection {
    /// Return a 2D tensor of [1, seqlength]
    fn encode_amino_acids(&self, device: &Device) -> Result<Tensor> {
        let n = self.iter_residues_aminoacid().count();
        let s = self
            .iter_residues_aminoacid()
            .map(|res| res.residue_name().to_string())
            .map(|res| aa3to1(&res))
            .map(aa1to_int);
        Tensor::from_iter(s, device)?.reshape((1, n))
    }
    // equivalent to protien MPNN's parse_PDB
    fn featurize(&self, device: &Device) -> Result<ProteinFeatures> {
        let x_37 = self.to_numeric_atom37(device)?;
        let x_37_m = Tensor::zeros((x_37.dim(0)?, x_37.dim(1)?), DType::F32, device)?;
        let (y, y_t, y_m) = self.to_numeric_ligand_atoms(device)?;
        let cb = calculate_cb(&x_37);
        let chain_labels = self.get_resids(); //  <-- need to double-check shape. I think this is all-atom
        let residue_ids = self.get_res_index();
        let residue_length = residue_ids.len();
        let r_idx = Tensor::from_iter(residue_ids, device)?.reshape((1, residue_length))?;
        let chain_letters: Vec<String> = self
            .iter_residues_aminoacid()
            .map(|res| res.chain_id().to_string())
            .collect();
        let chain_list: Vec<String> = chain_letters
            .clone()
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        // Numeric chain labels (optional)
        let chain_labels: Option<Vec<f64>> = None; // Could populate if needed
        let s = self.encode_amino_acids(device)?;
        // coordinates of the backbone atoms
        let indices = Tensor::from_slice(
            &[0i64, 1i64, 2i64, 4i64], // index of N/CA/C/O as integers
            (4,),
            &device,
        )?;
        let x = x_37.index_select(&indices, 2)?;
        Ok(ProteinFeatures {
            s,
            x,
            x_mask: Some(x_37_m),
            y,
            y_t,
            y_m: Some(y_m),
            r_idx,
            chain_labels,
            chain_letters,
            mask_c: None,
            chain_list,
        })
    }
    fn get_res_index(&self) -> Vec<u32> {
        self.iter_residues_aminoacid()
            .map(|res| res.residue_id() as u32)
            .collect()
    }
    /// create numeric Tensor of shape [<batch>, <sequence-length>, <N/CA/C/O = 4>, <xyz=3>]
    fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor> {
        let res_count = self.iter_residues_aminoacid().count();
        let mut backbone_data = vec![0f32; res_count * 4 * 3];
        for residue in self.iter_residues_aminoacid() {
            let resid = residue.residue_id() as usize;
            let backbone_atoms = [
                residue.find_atom_by_name("N"),
                residue.find_atom_by_name("CA"),
                residue.find_atom_by_name("C"),
                residue.find_atom_by_name("O"),
            ];
            for (atom_idx, maybe_atom) in backbone_atoms.iter().enumerate() {
                if let Some(atom) = maybe_atom {
                    let [x, y, z] = atom.coords();
                    let base_idx = (resid * 4 + atom_idx) * 3;
                    backbone_data[base_idx] = *x;
                    backbone_data[base_idx + 1] = *y;
                    backbone_data[base_idx + 2] = *z;
                }
            }
        }
        Tensor::from_vec(backbone_data, (1, res_count, 4, 3), device)
    }

    /// create numeric Tensor of shape [batch <sequence-length>, 37, 3]
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor> {
        let res_count = self.iter_residues_aminoacid().count();
        let mut atom37_data = vec![0f32; res_count * 37 * 3];
        for (idx, residue) in self.iter_residues_aminoacid().enumerate() {
            for atom_type in AAAtom::iter().filter(|&a| a != AAAtom::Unknown) {
                if let Some(atom) = residue.find_atom_by_name(&atom_type.to_string()) {
                    let [x, y, z] = atom.coords();
                    let base_idx = (idx * 37 + atom_type as usize) * 3;
                    atom37_data[base_idx] = *x;
                    atom37_data[base_idx + 1] = *y;
                    atom37_data[base_idx + 2] = *z;
                }
            }
        }
        Tensor::from_vec(atom37_data, (1, res_count, 37, 3), device)
    }

    // create numeric tensor for ligands.
    //
    // 1. Filter non-protein and water
    // 2. Filter out H, and HE
    // 3. convert to 3 tensors:
    //           y = coords
    //           y_t = elements
    //           y_m = mask
    fn to_numeric_ligand_atoms(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)> {
        let (coords, elements): (Vec<[f32; 3]>, Vec<Element>) = self
            .iter_residues()
            .filter(|residue| {
                let res_name = &residue.residue_name();
                !residue.is_amino_acid() && *res_name != "HOH" && *res_name != "WAT"
            })
            .flat_map(|residue| {
                residue
                    .iter_atoms()
                    .filter(|atom| is_heavy_atom(atom.element()))
                    .map(|atom| (*atom.coords(), *atom.element()))
                    .collect::<Vec<_>>()
            })
            .multiunzip();
        let y = Tensor::from_slice(&coords.concat(), (coords.len(), 3), device)?;
        let y_t = Tensor::from_slice(
            &elements
                .iter()
                .map(|e| e.atomic_number() as f32)
                .collect::<Vec<_>>(),
            (elements.len(),),
            device,
        )?;
        let y_m = Tensor::ones_like(&y)?;
        Ok((y, y_t, y_m))
    }
    fn to_pdb(&self) {
        unimplemented!()
    }
}

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
