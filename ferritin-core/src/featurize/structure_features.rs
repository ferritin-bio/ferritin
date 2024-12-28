//!  Protein->Tensor utiilities useful for Machine Learning
use super::utilities::{aa1to_int, aa3to1, cross_product, int_to_aa1, AAAtom};
use crate::AtomCollection;
use candle_core::{DType, Device, Error as CandleError, IndexOp, Result, Tensor};
use itertools::MultiUnzip;
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

    /// Convert amino acid sequence to numeric representation
    fn create_CB(&self, device: &Device) -> Result<Tensor>;

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

    /// Convert amino acid sequence to numeric representation
    fn create_CB(&self, device: &Device) -> Result<Tensor> {
        // N = input_dict["X"][:, 0, :]
        //         CA = input_dict["X"][:, 1, :]
        //         C = input_dict["X"][:, 2, :]
        //         b = CA - N
        //         c = C - CA
        //         a = torch.cross(b, c, axis=-1)
        //         CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        //
        let backbone = self.to_numeric_backbone_atoms(device)?;
        // Extract N, CA, C coordinates
        let n = backbone.i((.., 0, ..))?; // First atom (N)
        let ca = backbone.i((.., 1, ..))?; // Second atom (CA)
        let c = backbone.i((.., 2, ..))?; // Third atom (C)

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

        // Stack the cross product components back together
        let a = Tensor::stack(&[&a_x, &a_y, &a_z], 1)?;

        // Final CB calculation: -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        let cb = (&a * a_coeff)? + (&b * b_coeff)? + (&c * c_coeff)? + &ca;

        Ok(cb?)
    }

    /// Get residue indices
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

    // Extract ligand atom coordinates and properties
    // So the final 4D shape is:
    // - (1, num_residues, number_of_ligand_atoms, 3)
    //   - 1: batch size
    //   - num_residues: number of protein residues (from CB.shape[0])
    //   - number_of_ligand_atoms: fixed number (16 in your case)
    //   - 3: x,y,z coordinates
    //
    // 1. Input Y starts as (N_atoms, 3)
    // 2. Key transformation steps:
    // ```python
    // # Creates (num_residues, N_atoms, 3) by repeating Y for each residue
    // Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
    // # Gathers nearest neighbors based on distances, shape becomes (num_residues, num_ligand_atoms, 3)
    // Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    // # Creates final Y with shape (num_residues, number_of_ligand_atoms, 3)
    // Y = torch.zeros([CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device)
    // Y[:, :num_nn_update] = Y_tmp
    // ```
    // 3. Finally in the featurize function:
    // ```python
    // output_dict["Y"] = Y[None,]  # Adds batch dimension
    // ```
    // So the final 4D shape is:
    // - (1, num_residues, number_of_ligand_atoms, 3)
    //   - 1: batch size
    //   - num_residues: number of protein residues (from CB.shape[0])
    //   - number_of_ligand_atoms: fixed number (16 in your case)
    //   - 3: x,y,z coordinates
    // The function finds the nearest ligand atoms to each protein residue's CB atom, effectively creating a local chemical environment representation for each residue
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

        // get the C-beta tensor.
        let CB = self.create_CB(device);

        let  (y, y_t, y_m, D_XY) = get_nearest_neighbours(
            CB, mask, y, y_t, y_m, number_of_ligand_atoms
        )

        Ok((y, y_t, y_m))
    }
}
