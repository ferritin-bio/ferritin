use super::utilities::{aa1to_int, aa3to1, AAAtom};
use crate::AtomCollection;
use itertools::MultiUnzip;
use ndarray::{Array, Array1, Array2, Array4};
use pdbtbx::Element;
use std::collections::{HashMap, HashSet};
use strum::IntoEnumIterator;

fn is_heavy_atom(element: &Element) -> bool {
    !matches!(element, Element::H | Element::He)
}

pub trait StructureFeatures {
    type Error;

    /// Convert amino acid sequence to numeric representation
    fn encode_amino_acids(&self) -> Result<Array2<f32>, Self::Error>;

    /// Convert structure into complete feature set
    fn featurize(&self) -> Result<ProteinFeatures, Self::Error>;

    /// Get residue indices
    fn get_res_index(&self) -> Array1<u32>;

    /// Extract backbone atom coordinates (N, CA, C, O)
    fn to_numeric_backbone_atoms(&self) -> Result<Array4<f32>, Self::Error>;

    /// Extract all atom coordinates in standard ordering
    fn to_numeric_atom37(&self) -> Result<Array4<f32>, Self::Error>;

    /// Extract ligand atom coordinates and properties
    fn to_numeric_ligand_atoms(
        &self,
    ) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>), Self::Error>;

    /// Convert to PDB format
    fn to_pdb(&self);
}

impl StructureFeatures for AtomCollection {
    type Error = ndarray::ShapeError;

    fn to_pdb(&self) {
        todo!()
    }
    fn featurize(&self) -> Result<ProteinFeatures, Self::Error> {
        todo!()
    }
    fn encode_amino_acids(&self) -> Result<Array2<f32>, Self::Error> {
        let n = self.iter_residues_aminoacid().count();
        let sequence: Vec<f32> = self
            .iter_residues_aminoacid()
            .map(|res| res.res_name)
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res) as f32)
            .collect();

        Array::from_shape_vec((1, n), sequence)
    }

    fn to_numeric_backbone_atoms(&self) -> Result<Array4<f32>, Self::Error> {
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

        Array::from_shape_vec((1, res_count, 4, 3), backbone_data)
    }

    fn to_numeric_atom37(&self) -> Result<Array4<f32>, Self::Error> {
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

        Array::from_shape_vec((1, res_count, 37, 3), atom37_data)
    }

    fn to_numeric_ligand_atoms(
        &self,
    ) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>), Self::Error> {
        let (coords, elements): (Vec<[f32; 3]>, Vec<Element>) = self
            .iter_residues_all()
            .filter(|residue| {
                let res_name = &residue.res_name;
                !residue.is_amino_acid() && res_name != "HOH" && res_name != "WAT"
            })
            .flat_map(|residue| {
                residue
                    .iter_atoms()
                    .filter(|atom| is_heavy_atom(&atom.element))
                    .map(|atom| (*atom.coords, atom.element.clone()))
                    .collect::<Vec<_>>()
            })
            .multiunzip();

        let n_atoms = coords.len();
        let coords_flat: Vec<f32> = coords.into_iter().flat_map(|[x, y, z]| [x, y, z]).collect();
        let coords_array = Array::from_shape_vec((n_atoms, 3), coords_flat)?;

        let elements_array =
            Array1::from_vec(elements.iter().map(|e| e.atomic_number() as f32).collect());

        let mask_array = Array::ones((n_atoms, 3));

        Ok((coords_array, elements_array, mask_array))
    }

    fn get_res_index(&self) -> Array1<u32> {
        self.iter_residues_aminoacid()
            .map(|res| res.res_id as u32)
            .collect()
    }
}

pub struct ProteinFeatures {
    /// protein amino acids sequences as 1D Array of u32
    pub sequence: Array2<f32>,
    /// protein coords by residue [batch, seqlength, 37, 3]
    pub coordinates: Array4<f32>,
    /// protein mask by residue
    pub mask: Option<Array4<f32>>,
    /// ligand coords
    pub ligand_coords: Array2<f32>,
    /// encoded ligand atom names
    pub ligand_types: Array1<f32>,
    /// ligand mask
    pub ligand_mask: Option<Array2<f32>>,
    /// residue indices
    pub residue_index: Array1<f32>,
    /// chain labels
    pub chain_labels: Option<Vec<f32>>,
    /// chain letters
    pub chain_letters: Vec<String>,
    /// chain mask
    pub chain_mask: Option<Array1<f32>>,
    /// list of chains
    pub chain_list: Vec<String>,
}

impl ProteinFeatures {
    pub fn get_coords(&self) -> &Array4<f32> {
        &self.coordinates
    }

    pub fn get_sequence(&self) -> &Array2<f32> {
        &self.sequence
    }

    pub fn get_sequence_mask(&self) -> Option<&Array4<f32>> {
        self.mask.as_ref()
    }

    pub fn get_residue_index(&self) -> &Array1<f32> {
        &self.residue_index
    }

    pub fn get_encoded(
        &self,
    ) -> Result<(Vec<String>, HashMap<String, usize>, HashMap<usize, String>), ndarray::ShapeError>
    {
        let r_idx_list = self
            .residue_index
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<_>>();
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

    pub fn get_encoded_array(
        &self,
        fixed_residues: &str,
    ) -> Result<Array1<f32>, ndarray::ShapeError> {
        let res_set: HashSet<String> = fixed_residues.split(' ').map(String::from).collect();
        let (encoded_res, _, _) = self.get_encoded()?;

        Ok(Array1::from_vec(
            encoded_res
                .iter()
                .map(|item| if res_set.contains(item) { 0.0 } else { 1.0 })
                .collect(),
        ))
    }

    pub fn get_chain_mask_array(
        &self,
        chains_to_design: &[String],
    ) -> Result<Array1<f32>, ndarray::ShapeError> {
        Ok(Array1::from_vec(
            self.chain_letters
                .iter()
                .map(|chain| {
                    if chains_to_design.contains(chain) {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        ))
    }

    pub fn update_mask(&mut self, array: Array4<f32>) -> Result<(), ndarray::ShapeError> {
        if let Some(ref mask) = self.mask {
            self.mask = Some(mask * &array);
        } else {
            self.mask = Some(array);
        }
        Ok(())
    }

    pub fn create_bias_array(
        &self,
        bias_aa: Option<&str>,
    ) -> Result<Array1<f32>, ndarray::ShapeError> {
        let mut bias_values = vec![0.0f32; 21];

        if let Some(bias_str) = bias_aa {
            for pair in bias_str.split(',') {
                if let Some((aa, value_str)) = pair.split_once(':') {
                    if let Ok(value) = value_str.parse::<f32>() {
                        if let Some(aa_char) = aa.chars().next() {
                            let idx = aa1to_int(aa_char) as usize;
                            bias_values[idx] = value;
                        }
                    }
                }
            }
        }

        Ok(Array1::from_vec(bias_values))
    }
}

// pub struct ProteinFeatures {
//     /// Protein amino acid sequence encoding
//     pub sequence: Array2<f32>,
//     /// Protein atom coordinates
//     pub coordinates: Array3<f32>,
//     /// Protein atom mask
//     pub atom_mask: Option<Array2<bool>>,
//     /// Ligand coordinates
//     pub ligand_coords: Array2<f32>,
//     /// Ligand atom types
//     pub ligand_types: Array2<f32>,
//     /// Ligand mask
//     pub ligand_mask: Option<Array2<bool>>,
//     /// Residue indices
//     pub residue_indices: Array1<u32>,
//     /// Chain labels
//     pub chain_labels: Option<Vec<f64>>,
//     /// Chain letters
//     pub chain_letters: Vec<String>,
//     /// Chain mask
//     pub chain_mask: Option<Array2<bool>>,
//     /// List of chains
//     pub chain_list: Vec<String>,
// }
// impl ProteinFeatures {
//     pub fn get_coords(&self) -> &Array3<f32> {
//         &self.coordinates
//     }

//     pub fn get_sequence(&self) -> &Array2<f32> {
//         &self.sequence
//     }

//     pub fn get_sequence_mask(&self) -> Option<&Array2<bool>> {
//         self.atom_mask.as_ref()
//     }

//     pub fn get_residue_index(&self) -> &Array1<u32> {
//         &self.residue_indices
//     }

//     pub fn get_encoded(
//         &self,
//     ) -> Result<
//         (Vec<String>, HashMap<String, usize>, HashMap<usize, String>),
//         Box<dyn std::error::Error>,
//     > {
//         // Implementation
//         todo!()
//     }
// }
//

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_test_data::TestFile;
    use ndarray::{s, Array4};

    #[test]
    fn test_atom_backbone_tensor() {
        let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
        let ac = AtomCollection::from(&pdb);
        let ac_backbone_tensor: Array4<f32> = ac.to_numeric_backbone_atoms().expect("REASON");
        // Check my residue coords in the Tensor
        // ATOM   1    N  N   . MET A 1 1   ? 24.277 8.374   -9.854  1.00 38.41  ? 0   MET A N   1
        // ATOM   2    C  CA  . MET A 1 1   ? 24.404 9.859   -9.939  1.00 37.90  ? 0   MET A CA  1
        // ATOM   3    C  C   . MET A 1 1   ? 25.814 10.249  -10.359 1.00 36.65  ? 0   MET A C   1
        // ATOM   4    O  O   . MET A 1 1   ? 26.748 9.469   -10.197 1.00 37.13  ? 0   MET A O   1
        let backbone_coords = [
            // Methionine - AA00
            ("N", (0, 0, 0, ..), vec![24.277, 8.374, -9.854]),
            ("CA", (0, 0, 1, ..), vec![24.404, 9.859, -9.939]),
            ("C", (0, 0, 2, ..), vec![25.814, 10.249, -10.359]),
            ("O", (0, 0, 3, ..), vec![26.748, 9.469, -10.197]),
            // Valine - AA01
            ("N", (0, 1, 0, ..), vec![25.964, 11.453, -10.903]),
            ("CA", (0, 1, 1, ..), vec![27.263, 11.924, -11.359]),
            ("C", (0, 1, 2, ..), vec![27.392, 13.428, -11.115]),
            ("O", (0, 1, 3, ..), vec![26.443, 14.184, -11.327]),
            // Glycing - AAlast
            ("N", (0, 153, 0, ..), vec![23.474, -3.227, 5.994]),
            ("CA", (0, 153, 1, ..), vec![22.818, -2.798, 7.211]),
            ("C", (0, 153, 2, ..), vec![22.695, -1.282, 7.219]),
            ("O", (0, 153, 3, ..), vec![21.870, -0.745, 7.992]),
        ];

        for (atom_name, (b, i, j, k), expected) in backbone_coords {
            let actual: Vec<f32> = ac_backbone_tensor.slice(s![b, i, j, k]).to_vec();
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }
    }

    // #[test]
    // fn test_all_atom37_tensor() {
    //     let device = Device::Cpu;
    //     let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
    //     let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
    //     let ac = AtomCollection::from(&pdb);
    //     let ac_backbone_tensor: Tensor = ac.to_numeric_atom37(&device).expect("REASON");
    //     // batch size of 1154 residues; all atoms; positions
    //     assert_eq!(ac_backbone_tensor.dims(), &[1, 154, 37, 3]);

    //     // Check my residue coords in the Tensor
    //     // ATOM   1    N  N   . MET A 1 1   ? 24.277 8.374   -9.854  1.00 38.41  ? 0   MET A N   1
    //     // ATOM   2    C  CA  . MET A 1 1   ? 24.404 9.859   -9.939  1.00 37.90  ? 0   MET A CA  1
    //     // ATOM   3    C  C   . MET A 1 1   ? 25.814 10.249  -10.359 1.00 36.65  ? 0   MET A C   1
    //     // ATOM   4    O  O   . MET A 1 1   ? 26.748 9.469   -10.197 1.00 37.13  ? 0   MET A O   1
    //     // ATOM   5    C  CB  . MET A 1 1   ? 24.070 10.495  -8.596  1.00 39.58  ? 0   MET A CB  1
    //     // ATOM   6    C  CG  . MET A 1 1   ? 24.880 9.939   -7.442  1.00 41.49  ? 0   MET A CG  1
    //     // ATOM   7    S  SD  . MET A 1 1   ? 24.262 10.555  -5.873  1.00 44.70  ? 0   MET A SD  1
    //     // ATOM   8    C  CE  . MET A 1 1   ? 24.822 12.266  -5.967  1.00 41.59  ? 0   MET A CE  1
    //     //
    //     // pub enum AAAtom {
    //     //     N = 0,    CA = 1,   C = 2,    CB = 3,   O = 4,
    //     //     CG = 5,   CG1 = 6,  CG2 = 7,  OG = 8,   OG1 = 9,
    //     //     SG = 10,  CD = 11,  CD1 = 12, CD2 = 13, ND1 = 14,
    //     //     ND2 = 15, OD1 = 16, OD2 = 17, SD = 18,  CE = 19,
    //     //     CE1 = 20, CE2 = 21, CE3 = 22, NE = 23,  NE1 = 24,
    //     //     NE2 = 25, OE1 = 26, OE2 = 27, CH2 = 28, NH1 = 29,
    //     //     NH2 = 30, OH = 31,  CZ = 32,  CZ2 = 33, CZ3 = 34,
    //     //     NZ = 35,  OXT = 36,
    //     //     Unknown = -1,
    //     // }
    //     let allatom_coords = [
    //         // Methionine - AA00
    //         // We iterate through these positions. Not all AA's have each
    //         ("N", (0, 0, 0, ..), vec![24.277, 8.374, -9.854]),
    //         ("CA", (0, 0, 1, ..), vec![24.404, 9.859, -9.939]),
    //         ("C", (0, 0, 2, ..), vec![25.814, 10.249, -10.359]),
    //         ("CB", (0, 0, 3, ..), vec![24.070, 10.495, -8.596]),
    //         ("O", (0, 0, 4, ..), vec![26.748, 9.469, -10.197]),
    //         ("CG", (0, 0, 5, ..), vec![24.880, 9.939, -7.442]),
    //         ("CG1", (0, 0, 6, ..), vec![0.0, 0.0, 0.0]),
    //         ("CG2", (0, 0, 7, ..), vec![0.0, 0.0, 0.0]),
    //         ("OG", (0, 0, 8, ..), vec![0.0, 0.0, 0.0]),
    //         ("OG1", (0, 0, 9, ..), vec![0.0, 0.0, 0.0]),
    //         ("SG", (0, 0, 10, ..), vec![0.0, 0.0, 0.0]),
    //         ("CD", (0, 0, 11, ..), vec![0.0, 0.0, 0.0]),
    //         ("CD1", (0, 0, 12, ..), vec![0.0, 0.0, 0.0]),
    //         ("CD2", (0, 0, 13, ..), vec![0.0, 0.0, 0.0]),
    //         ("ND1", (0, 0, 14, ..), vec![0.0, 0.0, 0.0]),
    //         ("ND2", (0, 0, 15, ..), vec![0.0, 0.0, 0.0]),
    //         ("OD1", (0, 0, 16, ..), vec![0.0, 0.0, 0.0]),
    //         ("OD2", (0, 0, 17, ..), vec![0.0, 0.0, 0.0]),
    //         ("SD", (0, 0, 18, ..), vec![24.262, 10.555, -5.873]),
    //         ("CE", (0, 0, 19, ..), vec![24.822, 12.266, -5.967]),
    //         ("CE1", (0, 0, 20, ..), vec![0.0, 0.0, 0.0]),
    //         ("CE2", (0, 0, 21, ..), vec![0.0, 0.0, 0.0]),
    //         ("CE3", (0, 0, 22, ..), vec![0.0, 0.0, 0.0]),
    //         ("NE", (0, 0, 23, ..), vec![0.0, 0.0, 0.0]),
    //         ("NE1", (0, 0, 24, ..), vec![0.0, 0.0, 0.0]),
    //         ("NE2", (0, 0, 25, ..), vec![0.0, 0.0, 0.0]),
    //         ("OE1", (0, 0, 26, ..), vec![0.0, 0.0, 0.0]),
    //         ("OE2", (0, 0, 27, ..), vec![0.0, 0.0, 0.0]),
    //         ("CH2", (0, 0, 28, ..), vec![0.0, 0.0, 0.0]),
    //         ("NH1", (0, 0, 29, ..), vec![0.0, 0.0, 0.0]),
    //         ("NH2", (0, 0, 30, ..), vec![0.0, 0.0, 0.0]),
    //         ("OH", (0, 0, 31, ..), vec![0.0, 0.0, 0.0]),
    //         ("CZ", (0, 0, 32, ..), vec![0.0, 0.0, 0.0]),
    //         ("CZ2", (0, 0, 33, ..), vec![0.0, 0.0, 0.0]),
    //         ("CZ3", (0, 0, 34, ..), vec![0.0, 0.0, 0.0]),
    //         ("NZ", (0, 0, 35, ..), vec![0.0, 0.0, 0.0]),
    //         ("OXT", (0, 0, 36, ..), vec![0.0, 0.0, 0.0]),
    //     ];
    //     for (atom_name, (b, i, j, k), expected) in allatom_coords {
    //         let actual: Vec<f32> = ac_backbone_tensor
    //             .i((b, i, j, k))
    //             .unwrap()
    //             .to_vec1()
    //             .unwrap();
    //         assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
    //     }
    // }

    // #[test]
    // fn test_ligand_tensor() {
    //     let device = Device::Cpu;
    //     let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
    //     let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
    //     let ac = AtomCollection::from(&pdb);
    //     let (ligand_coords, ligand_elements, _) =
    //         ac.to_numeric_ligand_atoms(&device).expect("REASON");

    //     // 54 residues; N/CA/C/O; positions
    //     assert_eq!(ligand_coords.dims(), &[54, 3]);

    //     // Check my residue coords in the Tensor
    //     //
    //     // HETATM 1222 S  S   . SO4 B 2 .   ? 30.746 18.706  28.896  1.00 47.98  ? 157 SO4 A S   1
    //     // HETATM 1223 O  O1  . SO4 B 2 .   ? 30.697 20.077  28.620  1.00 48.06  ? 157 SO4 A O1  1
    //     // HETATM 1224 O  O2  . SO4 B 2 .   ? 31.104 18.021  27.725  1.00 47.52  ? 157 SO4 A O2  1
    //     // HETATM 1225 O  O3  . SO4 B 2 .   ? 29.468 18.179  29.331  1.00 47.79  ? 157 SO4 A O3  1
    //     // HETATM 1226 O  O4  . SO4 B 2 .   ? 31.722 18.578  29.881  1.00 47.85  ? 157 SO4 A O4  1
    //     let allatom_coords = [
    //         ("S", (0, ..), vec![30.746, 18.706, 28.896]),
    //         ("O1", (1, ..), vec![30.697, 20.077, 28.620]),
    //         ("O2", (2, ..), vec![31.104, 18.021, 27.725]),
    //         ("O3", (3, ..), vec![29.468, 18.179, 29.331]),
    //         ("O4", (4, ..), vec![31.722, 18.578, 29.881]),
    //     ];

    //     for (atom_name, (i, j), expected) in allatom_coords {
    //         let actual: Vec<f32> = ligand_coords.i((i, j)).unwrap().to_vec1().unwrap();
    //         assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
    //     }

    //     // Now check the elements
    //     let elements: Vec<&str> = ligand_elements
    //         .to_vec1::<f32>()
    //         .unwrap()
    //         .into_iter()
    //         .map(|elem| Element::new(elem as usize).unwrap().symbol())
    //         .collect();

    //     assert_eq!(elements[0], "S");
    //     assert_eq!(elements[1], "O");
    //     assert_eq!(elements[2], "O");
    //     assert_eq!(elements[3], "O");
    // }

    // #[test]
    // fn test_backbone_tensor() {
    //     let device = Device::Cpu;
    //     let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
    //     let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
    //     let ac = AtomCollection::from(&pdb);
    //     let xyz_37 = ac
    //         .to_numeric_atom37(&device)
    //         .expect("XYZ creation for all-atoms");
    //     assert_eq!(xyz_37.dims(), [1, 154, 37, 3]);

    //     // # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
    //     let xyz_m = create_backbone_mask_37(&xyz_37).expect("masking procedure should work");
    //     assert_eq!(xyz_m.dims(), &[1, 154, 37]);
    // }

    // #[test]
    // fn test_compute_nearest_neighbors() {
    //     // let device = Device::Cpu;
    //     let test_dtype = DType::F32;

    //     // Create a simple 2x3x3 tensor representing 2 sequences of 3 points in 3D space
    //     let coords = Tensor::new(
    //         &[
    //             [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], // First sequence
    //             [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], // Second sequence
    //         ],
    //         &device,
    //     )
    //     .unwrap()
    //     .to_dtype(test_dtype)
    //     .unwrap();

    //     // Create mask indicating all points are valid
    //     let mask = Tensor::ones((2, 3), test_dtype, &device).unwrap();

    //     // Get 2 nearest neighbors for each point
    //     let (distances, indices) = compute_nearest_neighbors(&coords, &mask, 2, 1e-6).unwrap();

    //     // Check shapes
    //     assert_eq!(distances.dims(), &[2, 3, 2]); // [batch, seq_len, k]
    //     assert_eq!(indices.dims(), &[2, 3, 2]); // [batch, seq_len, k]

    //     // For first sequence, point [1,0,0] should have [0,0,0] and [2,0,0] as nearest neighbors
    //     let point_neighbors: Vec<u32> = indices.i((0, 1, ..)).unwrap().to_vec1().unwrap();
    //     assert_eq!(point_neighbors, vec![0, 2]);

    //     // Check distances are correct
    //     let point_distances: Vec<f32> = distances.i((0, 1, ..)).unwrap().to_vec1().unwrap();
    //     assert!((point_distances[0] - 1.0).abs() < 1e-5);
    //     assert!((point_distances[1] - 1.0).abs() < 1e-5);
    // }
}
