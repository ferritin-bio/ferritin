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

use super::utilities::{aa1to_int, aa3to1, AAAtom};
use candle_core::{Device, Result, Tensor};
use ferritin_core::AtomCollection;
use itertools::MultiUnzip;
use pdbtbx::Element;
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
            .map(|res| res.res_name)
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res));

        Ok(Tensor::from_iter(s, device)?.reshape((1, n))?)
    }
    // equivalent to protien MPNN's parse_PDB
    fn featurize(&self, device: &Device) -> Result<ProteinFeatures> {
        todo!();
        // let x_37 = self.to_numeric_atom37(device)?;
        // let x_37_m = Tensor::zeros((x_37.dim(0)?, x_37.dim(1)?), DType::F64, device)?;
        // let (y, y_t, y_m) = self.to_numeric_ligand_atoms(device)?;

        // // get CB locations...
        // // although we have these already for our full set...
        // let cb = calculate_cb(&x_37);

        // // chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
        // let chain_labels = self.get_resids(); //  <-- need to double-check shape. I think this is all-atom

        // // R_idx = np.array(CA_resnums, dtype=np.int32)
        // // let _r_idx = self.get_resids(); // todo()!

        // // amino acid names as int....
        // let s = self.encode_amino_acids(device)?;

        // // coordinates of the backbone atoms
        // let indices = Tensor::from_slice(
        //     &[0i64, 1i64, 2i64, 4i64], // index of N/CA/C/O as integers
        //     (4,),
        //     &device,
        // )?;

        // let x = x_37.index_select(&indices, 1)?;

        // Ok(ProteinFeatures {
        //     s,
        //     x,
        //     x_mask: Some(x_37_m),
        //     y,
        //     y_t,
        //     y_m: Some(y_m),
        //     r_idx: None,
        //     chain_labels: None,
        //     chain_letters: None,
        //     mask_c: None,
        //     chain_list: None,
        // })
    }
    fn get_res_index(&self) -> Vec<u32> {
        self.iter_residues_aminoacid()
            .map(|res| res.res_id as u32)
            .collect()
    }
    /// create numeric Tensor of shape [1, <sequence-length>, 4, 3] where the 4 is N/CA/C/O
    fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor> {
        let res_count = self.iter_residues_aminoacid().count();
        let mut backbone_data = vec![0f32; res_count * 4 * 3];
        for residue in self.iter_residues_aminoacid() {
            let resid = residue.res_id as usize;
            let backbone_atoms = [
                residue.find_atom_by_name("N"),
                residue.find_atom_by_name("CA"),
                residue.find_atom_by_name("C"),
                residue.find_atom_by_name("O"),
            ];
            for (atom_idx, maybe_atom) in backbone_atoms.iter().enumerate() {
                if let Some(atom) = maybe_atom {
                    let [x, y, z] = atom.coords;
                    let base_idx = (resid * 4 + atom_idx) * 3;
                    backbone_data[base_idx] = *x;
                    backbone_data[base_idx + 1] = *y;
                    backbone_data[base_idx + 2] = *z;
                }
            }
        }
        // Create tensor with shape [1,residues, 4, 3]
        Tensor::from_vec(backbone_data, (1, res_count, 4, 3), &device)
    }

    /// create numeric Tensor of shape [<sequence-length>, 37, 3]
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor> {
        let res_count = self.iter_residues_aminoacid().count();
        let mut atom37_data = vec![0f32; res_count * 37 * 3];

        for (idx, residue) in self.iter_residues_aminoacid().enumerate() {
            for atom_type in AAAtom::iter().filter(|&a| a != AAAtom::Unknown) {
                if let Some(atom) = residue.find_atom_by_name(&atom_type.to_string()) {
                    let [x, y, z] = atom.coords;
                    let base_idx = (idx * 37 + atom_type as usize) * 3;
                    atom37_data[base_idx] = *x;
                    atom37_data[base_idx + 1] = *y;
                    atom37_data[base_idx + 2] = *z;
                }
            }
        }
        // Create tensor with shape [residues, 37, 3]
        Tensor::from_vec(atom37_data, (1, res_count, 37, 3), &device)
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
            .iter_residues_all()
            // keep only the non-protein, non-water residues
            .filter(|residue| {
                let res_name = &residue.res_name;
                !residue.is_amino_acid() && res_name != "HOH" && res_name != "WAT"
            })
            // keep only the heavy atoms
            .flat_map(|residue| {
                residue
                    .iter_atoms()
                    .filter(|atom| is_heavy_atom(&atom.element))
                    .map(|atom| (*atom.coords, atom.element.clone()))
                    .collect::<Vec<_>>()
            })
            .multiunzip();
        // Create coordinates tensor
        let y = Tensor::from_slice(&coords.concat(), (coords.len(), 3), device)?;

        // Create elements tensor
        let y_t = Tensor::from_slice(
            &elements
                .iter()
                .map(|e| e.atomic_number() as f32)
                .collect::<Vec<_>>(),
            (elements.len(),),
            device,
        )?;

        // Create mask tensor (all ones in this case since we've already filtered)
        let y_m = Tensor::ones_like(&y)?;

        Ok((y, y_t, y_m))
    }

    fn to_pdb(&self) {
        // Todo: finish this. will require somethign like prody....
        // pub fn write_full_pdb(
        //     save_path: &str,
        //     x: &Tensor,
        //     x_m: &Tensor,
        //     b_factors: &Tensor,
        //     r_idx: &Tensor,
        //     chain_letters: &Tensor,
        //     s: &Tensor,
        //     other_atoms: Option<&Tensor>,
        //     icodes: Option<&Tensor>,
        //     force_hetatm: bool,
        // ) -> Result<()> {
        //     //     save_path : path where the PDB will be written to
        //     //     X : protein atom xyz coordinates shape=[length, 14, 3]
        //     //     X_m : protein atom mask shape=[length, 14]
        //     //     b_factors: shape=[length, 14]
        //     //     R_idx: protein residue indices shape=[length]
        //     //     chain_letters: protein chain letters shape=[length]
        //     //     S : protein amino acid sequence shape=[length]
        //     //     other_atoms: other atoms parsed by prody
        //     //     icodes: a list of insertion codes for the PDB; e.g. antibody loops
        //     //     """

        //     let s_str: Vec<&str> = s
        //         .iter()
        //         .map(|&aa| restype_int_to_str(aa))
        //         .map(restype_1to3)
        //         .collect();
        //     let mut x_list = Vec::new();
        //     let mut b_factor_list = Vec::new();
        //     let mut atom_name_list = Vec::new();
        //     let mut element_name_list = Vec::new();
        //     let mut residue_name_list = Vec::new();
        //     let mut residue_number_list = Vec::new();
        //     let mut chain_id_list = Vec::new();
        //     let mut icodes_list = Vec::new();

        //     for (i, aa) in s_str.iter().enumerate() {
        //         let sel = x_m.get(i)?.to_dtype(DType::I32)?.eq(&1)?;
        //         let total = sel.sum_all()?.to_scalar::<i32>()?;
        //         let tmp = Tensor::from_slice(&restype_name_to_atom14_names(aa))?.masked_select(&sel)?;
        //         x_list.push(x.get(i)?.masked_select(&sel)?);
        //         b_factor_list.push(b_factors.get(i)?.masked_select(&sel)?);
        //         atom_name_list.push(tmp.clone());
        //         element_name_list.extend(tmp.iter().map(|&atom| &atom[..1]));
        //         residue_name_list.extend(std::iter::repeat(aa).take(total as usize));
        //         residue_number_list.extend(std::iter::repeat(r_idx.get(i)?).take(total as usize));
        //         chain_id_list.extend(std::iter::repeat(chain_letters.get(i)?).take(total as usize));
        //         icodes_list.extend(std::iter::repeat(icodes.get(i)?).take(total as usize));
        //     }

        //     let x_stack = Tensor::cat(&x_list, 0)?;
        //     let b_factor_stack = Tensor::cat(&b_factor_list, 0)?;
        //     let atom_name_stack = Tensor::cat(&atom_name_list, 0)?;

        //     let mut protein = prody::AtomGroup::new();
        //     protein.set_coords(&x_stack)?;
        //     protein.set_betas(&b_factor_stack)?;
        //     protein.set_names(&atom_name_stack)?;
        //     protein.set_resnames(&residue_name_list)?;
        //     protein.set_elements(&element_name_list)?;
        //     protein.set_occupancies(&Tensor::ones(x_stack.shape()[0])?)?;
        //     protein.set_resnums(&residue_number_list)?;
        //     protein.set_chids(&chain_id_list)?;
        //     protein.set_icodes(&icodes_list)?;

        //     if let Some(other_atoms) = other_atoms {
        //         let mut other_atoms_g = prody::AtomGroup::new();
        //         other_atoms_g.set_coords(&other_atoms.get_coords()?)?;
        //         other_atoms_g.set_names(&other_atoms.get_names()?)?;
        //         other_atoms_g.set_resnames(&other_atoms.get_resnames()?)?;
        //         other_atoms_g.set_elements(&other_atoms.get_elements()?)?;
        //         other_atoms_g.set_occupancies(&other_atoms.get_occupancies()?)?;
        //         other_atoms_g.set_resnums(&other_atoms.get_resnums()?)?;
        //         other_atoms_g.set_chids(&other_atoms.get_chids()?)?;
        //         if force_hetatm {
        //             other_atoms_g.set_flags("hetatm", &other_atoms.get_flags("hetatm")?)?;
        //         }
        //         prody::write_pdb(save_path, &(protein + other_atoms_g))?;
        //     } else {
        //         prody::write_pdb(save_path, &protein)?;
        //     }
        // }
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
    // CA_icodes:     NumPy array dimensions: (93)
    // put these here temporarily
    // bias_AA: Option<Tensor>,
    // bias_AA_per_residue: Option<Tensor>,
    // omit_AA_per_residue_multi: Option<Tensor>,
    // backbone: String,
    // other_atoms: String,
    // ca_icodes: Vec<String>,
    // water_atoms: String,
    // // [[0, 1, 14], [10,11,14,15], [20, 21]]
    // pub symmetry_residues: Option<Vec<Vec<i64>>>,
    // // [[1.0, 1.0, 1.0], [-2.0,1.1,0.2,1.1], [2.3, 1.1]]
    // pub symmetry_weights: Option<Vec<Vec<f64>>>,
    // homo_oligomer: Option<bool>,
    // pub batch_size: Option<i64>,
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
    pub fn get_encoded(
        &self,
    ) -> Result<(Vec<String>, HashMap<String, usize>, HashMap<usize, String>)> {
        // Creates a set of mappings from

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
        if let Some(ref mask) = self.x_mask {
            self.x_mask = Some(mask.mul(&tensor)?);
        } else {
            self.x_mask = Some(tensor);
        }
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
                            // Get first char from aa str and convert u32 to usize for indexing
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
}
