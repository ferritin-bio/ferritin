//!  Protein->Tensor utiilities useful for Machine Learning
use super::utilities::{aa1to_int, aa3to1, int_to_aa1, AAAtom};
use crate::AtomCollection;
use candle_core::{DType, Device, Error as CandleError, Result, Tensor};
use pdbtbx::Element;
use strum::IntoEnumIterator;

// Helper Fns --------------------------------------
fn is_heavy_atom(element: &Element) -> bool {
    !matches!(element, Element::H | Element::He)
}

///. Trait defining Protein->Tensor utiilities useful for Machine Learning
pub trait StructureFeatures {
    /// Convert amino acid sequence to numeric representation
    fn decode_amino_acids(&self, device: &Device) -> Result<Tensor>;

    /// Convert amino acid sequence to numeric representation
    fn encode_amino_acids(&self, device: &Device) -> Result<Tensor>;

    /// Get residue indices
    fn get_res_index(&self) -> Result<Tensor>;

    /// Extract backbone atom coordinates (N, CA, C, O)
    fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor>;

    /// Extract all atom coordinates in standard ordering
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor>;

    /// Extract ligand atom coordinates and properties
    fn to_numeric_ligand_atoms(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)>;
}

impl StructureFeatures for AtomCollection {
    /// Convert amino acid sequence to numeric representation
    fn decode_amino_acids(&self, device: &Device) -> Result<Tensor> {
        todo!()
    }

    /// Convert amino acid sequence to numeric representation
    fn encode_amino_acids(&self, device: &Device) -> Result<Tensor> {
        let n = self.iter_residues_aminoacid().count();
        let s = self
            .iter_residues_aminoacid()
            .map(|res| res.res_name)
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res));

        Ok(Tensor::from_iter(s, device)?.reshape((1, n))?)
    }

    /// Get residue indices
    fn get_res_index(&self) -> Result<Tensor> {
        todo!()
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

    /// create numeric Tensor of shape [1, <sequence-length>, 37, 3]
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

    /// Extract ligand atom coordinates and properties
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
}
