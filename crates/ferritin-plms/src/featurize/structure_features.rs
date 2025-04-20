//!  Protein->Tensor utilities useful for Machine Learning
use super::utilities::{AAAtom, aa1to_int, aa3to1, get_nearest_neighbours};
use crate::ligandmpnn::proteinfeatures::ProteinFeatures;
use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use ferritin_core::AtomCollection;
use ferritin_core::info::elements::Element;
use itertools::MultiUnzip;
use std::collections::HashSet;
use strum::IntoEnumIterator;

// Helper Fns --------------------------------------
fn is_heavy_atom(element: &Element) -> bool {
    !matches!(element, Element::H | Element::He)
}

///. Trait defining Protein->Tensor utilities useful for Machine Learning
pub trait StructureFeatures {
    /// Convert amino acid sequence to numeric representation
    fn decode_amino_acids(&self, device: &Device) -> Result<Tensor>;

    /// Convert amino acid sequence to numeric representation
    fn encode_amino_acids(&self, device: &Device) -> Result<Tensor>;

    /// Convert amino acid sequence to numeric representation
    fn create_cb(&self, device: &Device) -> Result<Tensor>;

    /// Prepare for ProteinMPNN
    fn featurize_lmpnn(&self, device: &Device) -> Result<ProteinFeatures>; // need more control over this featurization process

    /// Get residue indices
    fn get_res_index(&self) -> Vec<u32>;

    /// Extract backbone atom coordinates (N, CA, C, O)
    fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor>;

    /// Extract all atom coordinates in standard ordering
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor>;

    /// Extract ligand atom coordinates and properties
    fn to_numeric_ligand_atoms(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)>;
}

impl StructureFeatures for AtomCollection {
    /// Convert amino acid sequence to numeric representation
    fn decode_amino_acids(&self, _device: &Device) -> Result<Tensor> {
        todo!()
    }

    /// Convert amino acid sequence to numeric representation
    fn encode_amino_acids(&self, device: &Device) -> Result<Tensor> {
        let n = self.iter_residues_aminoacid().count();
        let s = self
            .iter_residues_aminoacid()
            .map(|res| res.residue_name().to_string())
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res));

        Ok(Tensor::from_iter(s, device)?.reshape((1, n))?)
    }

    /// Calculate CB for each residue
    fn create_cb(&self, device: &Device) -> Result<Tensor> {
        // N = input_dict["X"][:, 0, :]
        //         CA = input_dict["X"][:, 1, :]
        //         C = input_dict["X"][:, 2, :]
        //         b = CA - N
        //         c = C - CA
        //         a = torch.cross(b, c, axis=-1)
        //         CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        //
        let backbone = self.to_numeric_backbone_atoms(device)?;
        let backbone = backbone.squeeze(0)?; // remove batch dim for calc

        // Extract N, CA, C coordinates
        let n = backbone.i((.., 0, ..))?;
        let ca = backbone.i((.., 1, ..))?;
        let c = backbone.i((.., 2, ..))?;

        // Constants for CB calculation
        let a_coeff = -0.58273431_f64;
        let b_coeff = 0.56802827_f64;
        let c_coeff = -0.54067466_f64;

        // Calculate vectors
        let b = (&ca - &n)?; // CA - N
        let c = (&c - &ca)?; // C - CA

        // Manual cross product components
        // a_x = b_y * c_z - b_z * c_y
        // a_y = b_z * c_x - b_x * c_z
        // a_z = b_x * c_y - b_y * c_x
        let b_x = b.i((.., 0))?;
        let b_y = b.i((.., 1))?;
        let b_z = b.i((.., 2))?;
        let c_x = c.i((.., 0))?;
        let c_y = c.i((.., 1))?;
        let c_z = c.i((.., 2))?;

        let a_x = ((&b_y * &c_z)? - (&b_z * &c_y)?)?;
        let a_y = ((&b_z * &c_x)? - (&b_x * &c_z)?)?;
        let a_z = ((&b_x * &c_y)? - (&b_y * &c_x)?)?;
        let a = Tensor::stack(&[&a_x, &a_y, &a_z], D::Minus1)?;

        // Final CB calculation: -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        let cb = ((&a * a_coeff)? + (&b * b_coeff)? + (&c * c_coeff)? + &ca)?;
        let cb = cb.unsqueeze(0)?;
        Ok(cb)
    }

    // Convert AtomCollection to ProteinFeatures
    fn featurize_lmpnn(&self, device: &Device) -> Result<ProteinFeatures> {
        let x_37 = self.to_numeric_atom37(device)?;
        let x_37_m = Tensor::zeros((x_37.dim(0)?, x_37.dim(1)?), DType::F32, device)?;
        let (y, y_t, y_m) = self.to_numeric_ligand_atoms(device)?;
        let cb = self.create_cb(device);
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
    /// Get residue indices
    fn get_res_index(&self) -> Vec<u32> {
        self.iter_residues_aminoacid()
            .map(|res| res.residue_id() as u32)
            .collect()
    }

    /// create numeric Tensor of shape [1, <sequence-length>, 4, 3] where the 4 is N/CA/C/O
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
                    let [x, y, z] = atom.coords();
                    let base_idx = (idx * 37 + atom_type as usize) * 3;
                    atom37_data[base_idx] = *x;
                    atom37_data[base_idx + 1] = *y;
                    atom37_data[base_idx + 2] = *z;
                }
            }
        }
        // Create tensor with shape [batch, residues, 37, 3]
        Tensor::from_vec(atom37_data, (1, res_count, 37, 3), &device)
    }

    // The purpose of this function it to create 3 output tensors that relate
    // key information about a protein sequence and ligands it interacts with.
    //
    // The outputs are:
    //  - y: 4D tensor of dimensions (<batch=1>, <num_residues>, <number_of_ligand_atoms>, <coords=3>)
    //  - y_t: 1D tensor of dimension = <num_residues>
    //  - y_m: 3D tensor of dimensions: (<batch=1>, <num_residues>, <number_of_ligand_atoms>))
    //
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
                .map(|e| e.atomic_number() as i64)
                .collect::<Vec<_>>(),
            (elements.len(),),
            device,
        )?;
        let y_m = Tensor::ones_like(&y)?;
        Ok((y, y_t, y_m))
    }
}
