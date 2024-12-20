// ferritin-core/src/features/ndarray_impl.rs
use super::{ProteinFeatures, StructureFeatures};
use crate::AtomCollection;
use ndarray::{Array, Array2};

impl StructureFeatures<Array2<f32>> for AtomCollection {
    type Error = ndarray::ShapeError;

    fn encode_amino_acids(&self) -> Result<Array2<f32>, Self::Error> {
        let n = self.iter_residues_aminoacid().count();
        let s: Vec<f32> = self
            .iter_residues_aminoacid()
            .map(|res| res.res_name)
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res) as f32)
            .collect();
        Array2::from_shape_vec((1, n), s)
    }

    fn featurize(&self, device: &Device) -> Result<ProteinFeatures> {
        todo!();
    }
    // equivalent to protien MPNN's parse_PDB
    /// create numeric Array of shape [1, <sequence-length>, 4, 3] where the 4 is N/CA/C/O
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
        // Create array with shape [1,residues, 4, 3]
        Array4::from_shape_vec((1, res_count, 4, 3), backbone_data)
    }
    /// create numeric Array of shape [<sequence-length>, 37, 3]
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
        // Create array with shape [1, residues, 37, 3]
        Array4::from_shape_vec((1, res_count, 37, 3), atom37_data)
    }
    // create numeric tensor for ligands.
    //
    // 1. Filter non-protein and water
    // 2. Filter out H, and HE
    // 3. convert to 3 tensors:
    //           y = coords
    //           y_t = elements
    //           y_m = mask
    fn to_numeric_ligand_atoms(
        &self,
    ) -> Result<(Array2<f32>, Array1<f32>, Array2<f32>), Self::Error> {
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
        let y = Array2::from_shape_vec((coords.len(), 3), coords.into_iter().flatten().collect())?;
        let y_t = Array1::from_vec(elements.iter().map(|e| e.atomic_number() as f32).collect());
        let y_m = Array2::ones((coords.len(), 3));
        Ok((y, y_t, y_m))
    }

    fn to_pdb(&self) {
        unimplemented!()
    }

    fn get_res_index(&self) -> Vec<u32> {
        self.iter_residues_aminoacid()
            .map(|res| res.res_id as u32)
            .collect()
    }
}

impl ProteinFeatures<Array2<f32>> {
    pub fn save_to_npy(&self, path: &str) -> Result<(), std::io::Error> {
        // Implementation for saving to .npy format
        todo!()
    }
}
