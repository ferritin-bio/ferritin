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

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use ferritin_core::{is_amino_acid, AtomCollection};
use itertools::MultiUnzip;
use pdbtbx::Element;
use strum::{Display, EnumIter, EnumString, IntoEnumIterator};

/// Convert the AtomCollection into a struct that can be passed to a model.
pub trait LMPNNFeatures {
    fn featurize(&self, device: &Device) -> Result<ProteinFeatures>;
    fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor>; // [residues, N/CA/C/O, xyz]
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor>; // [residues, N/CA/C/O....37, xyz]
    fn to_numeric_ligand_atoms(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)>; // ( positions , elements, mask )
}

fn is_heavy_atom(element: &Element) -> bool {
    !matches!(element, Element::H | Element::He)
}

///
/// Input coords. Output 1 <batch  x 1 > Tensor
/// representing whether each residue has all 4 backbone atoms.
/// note that the internal ordering is different between
/// backbone only [N/CA/C/O] and all-atom [N/CA/C/CB/O]....
///
fn create_backbone_mask_37(xyz_37: &Tensor) -> Result<Tensor> {
    let backbone_indices = Tensor::new(&[0i64, 1, 2, 4], xyz_37.device())?;
    let backbone_selection = xyz_37.index_select(&backbone_indices, 1)?; // [154, 4, 3]

    // Check if coordinates exist (sum over xyz dimensions)
    let exists = backbone_selection.sum(2)?; // [154, 4]

    // All 4 atoms must exist
    let all_exist = exists.sum_keepdim(1)?; // [154, 1]
    Ok(all_exist)
}

fn calculate_cb(xyz_37: &Tensor) -> Result<Tensor> {
    // make sure we are dealing with
    let (_, dim37, dim3) = xyz_37.dims3()?;
    assert_eq!(dim37, 37);
    assert_eq!(dim3, 3);

    // Constants for CB calculation
    let a_coeff = -0.58273431f64;
    let b_coeff = 0.56802827f64;
    let c_coeff = -0.54067466f64;

    // Get N, CA, C coordinates
    let n = xyz_37.i((.., 0, ..))?; // N  at index 0
    let ca = xyz_37.i((.., 1, ..))?; // CA at index 1
    let c = xyz_37.i((.., 2, ..))?; // C  at index 2

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

// Create default
impl LMPNNFeatures for AtomCollection {
    // create numeric Tensor of shape [<length>, 37, 3]
    fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor> {
        let res_count = self.iter_residues_aminoacid().count();
        let mut atom37_data = vec![0f32; res_count * 37 * 3];
        for residue in self.iter_residues_aminoacid() {
            let resid = residue.res_id as usize;
            for atom_type in AAAtom::iter().filter(|&a| a != AAAtom::Unknown) {
                if let Some(atom) = residue.find_atom_by_name(&atom_type.to_string()) {
                    let [x, y, z] = atom.coords;
                    let base_idx = (resid * 37 + atom_type as usize) * 3;
                    atom37_data[base_idx] = *x;
                    atom37_data[base_idx + 1] = *y;
                    atom37_data[base_idx + 2] = *z;
                }
            }
        }
        // Create tensor with shape [residues, 37, 3]
        Tensor::from_vec(atom37_data, (res_count, 37, 3), &device)
    }
    // // create numeric Tensor of shape [<length>, 4, 3] where the 4 is N/CA/C/O
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

        // Create tensor with shape [residues, 4, 3]
        Tensor::from_vec(backbone_data, (res_count, 4, 3), &device)
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
                !is_amino_acid(res_name) && res_name != "HOH" && res_name != "WAT"
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

    // equivalent to protien MPNN's parse_PDB
    fn featurize(&self, device: &Device) -> Result<ProteinFeatures> {
        let x_37 = self.to_numeric_atom37(device)?;
        let x_37_m = Tensor::zeros((x_37.dim(0)?, x_37.dim(1)?), DType::F64, device)?;
        let (y, y_t, y_m) = self.to_numeric_ligand_atoms(device)?;

        // get CB locations...
        // although we have these already for our full set...
        let cb = calculate_cb(&x_37);

        // chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
        let chain_labels = self.get_resids(); //  <- need to double check shape. I think this is all-atom

        // R_idx = np.array(CA_resnums, dtype=np.int32)
        // let _r_idx = self.get_resids(); // todo()!

        // amino acid names as int....
        let s: Vec<i32> = self
            .iter_residues_aminoacid()
            .map(|res| res.res_name)
            .map(|res| aa3to1(&res))
            .map(|res| aa1to_int(res))
            .collect();

        // coordaintes of the backbone atoms

        let indices = Tensor::from_slice(
            &[0., 1., 2., 4.], // index of N/CA/C/O
            (4,),
            &device,
        )?;

        let X = x_37.index_select(&indices, 1)?;

        Ok(ProteinFeatures {
            s,
            x: X,
            x_mask: Some(x_37_m),
            y,
            y_t,
            y_m: Some(y_m),
            r_idx: None,
            chain_labels: None,
            chain_letters: None,
            mask_c: None,
            chain_list: None,
        })
    }
}

pub struct ProteinFeatures {
    // protein amino acids sequences as ints
    s: Vec<i32>,
    // protein co-ords by residue [1, 37, 4]
    x: Tensor,
    // protein mask by residue
    x_mask: Option<Tensor>,
    // ligand coords
    y: Tensor,
    // encoded ligand atom names
    y_t: Tensor,
    // .ignamd mask
    y_m: Option<Tensor>,
    // R_idx:         Tensor dimensions: torch.Size([93])          # protein residue indices shape=[length]
    r_idx: Option<Vec<i32>>,
    // chain_labels:  Tensor dimensions: torch.Size([93])          # protein chain letters shape=[length]
    chain_labels: Option<Vec<f64>>,
    // chain_letters: NumPy array dimensions: (93,)
    chain_letters: Option<Vec<String>>,
    /// mask_c:        Tensor dimensions: torch.Size([93])
    mask_c: Option<Tensor>,
    chain_list: Option<Vec<String>>,
    // CA_icodes:     NumPy array dimensions: (93,)
}

#[rustfmt::skip]
fn aa3to1(aa: &str) -> char {
    match aa {
        "ALA" => 'A', "CYS" => 'C', "ASP" => 'D',
        "GLU" => 'E', "PHE" => 'F', "GLY" => 'G',
        "HIS" => 'H', "ILE" => 'I', "LYS" => 'K',
        "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
        "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R',
        "SER" => 'S', "THR" => 'T', "VAL" => 'V',
        "TRP" => 'W', "TYR" => 'Y', _     => 'X',
    }
}

#[rustfmt::skip]
fn aa1to_int(aa: char) -> i32 {
    match aa {
        'A' => 0, 'C' => 1, 'D' => 2,
        'E' => 3, 'F' => 4, 'G' => 5,
        'H' => 6, 'I' => 7, 'K' => 8,
        'L' => 9, 'M' => 10, 'N' => 11,
        'P' => 12, 'Q' => 13, 'R' => 14,
        'S' => 15, 'T' => 16, 'V' => 17,
        'W' => 18, 'Y' => 19, _   => 20,
    }
}

#[rustfmt::skip]
#[derive(Debug, Clone, Copy, PartialEq, Display, EnumString, EnumIter)]
pub enum AAAtom {
    N = 0,    CA = 1,   C = 2,    CB = 3,   O = 4,
    CG = 5,   CG1 = 6,  CG2 = 7,  OG = 8,   OG1 = 9,
    SG = 10,  CD = 11,  CD1 = 12, CD2 = 13, ND1 = 14,
    ND2 = 15, OD1 = 16, OD2 = 17, SD = 18,  CE = 19,
    CE1 = 20, CE2 = 21, CE3 = 22, NE = 23,  NE1 = 24,
    NE2 = 25, OE1 = 26, OE2 = 27, CH2 = 28, NH1 = 29,
    NH2 = 30, OH = 31,  CZ = 32,  CZ2 = 33, CZ3 = 34,
    NZ = 35,  OXT = 36,
    Unknown = -1,
}
impl AAAtom {
    // Get numeric value (might still be useful in some contexts)
    pub fn to_index(&self) -> usize {
        *self as usize
    }
}

macro_rules! define_residues {
    ($($name:ident: $code3:expr, $code1:expr, $idx:expr, $features:expr, $atoms14:expr),* $(,)?) => {
        #[derive(Debug, Copy, Clone)]
        pub enum Residue {
            $($name),*
        }

        impl Residue {
            pub const fn code3(&self) -> &'static str {
                match self {
                    $(Self::$name => $code3),*
                }
            }
            pub const fn code1(&self) -> char {
                match self {
                    $(Self::$name => $code1),*
                }
            }
            pub const fn atoms14(&self) -> [AAAtom; 14] {
                match self {
                    $(Self::$name => $atoms14),*
                }
            }
            pub fn from_int(value: i32) -> Self {
                match value {
                    $($idx => Self::$name,)*
                    _ => Self::UNK
                }
            }
            pub fn to_int(&self) -> i32 {
                match self {
                    $(Self::$name => $idx),*
                }
            }
        }
    }
}

define_residues! {
    ALA: "ALA", 'A', 0,  [1.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    CYS: "CYS", 'C', 1,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::SG, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    ASP: "ASP", 'D', 2,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::OD1, AAAtom::OD2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    GLU: "GLU", 'E', 3,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD, AAAtom::OE1, AAAtom::OE2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    PHE: "PHE", 'F', 4,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD1, AAAtom::CD2, AAAtom::CE1, AAAtom::CE2, AAAtom::CZ, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    GLY: "GLY", 'G', 5,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    HIS: "HIS", 'H', 6,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::ND1, AAAtom::CD2, AAAtom::CE1, AAAtom::NE2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    ILE: "ILE", 'I', 7,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG1, AAAtom::CG2, AAAtom::CD1, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    LYS: "LYS", 'K', 8,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD, AAAtom::CE, AAAtom::NZ, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    LEU: "LEU", 'L', 9,  [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD1, AAAtom::CD2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    MET: "MET", 'M', 10, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::SD, AAAtom::CE, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    ASN: "ASN", 'N', 11, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::OD1, AAAtom::ND2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    PRO: "PRO", 'P', 12, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    GLN: "GLN", 'Q', 13, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD, AAAtom::OE1, AAAtom::NE2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    ARG: "ARG", 'R', 14, [0.0, 1.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD, AAAtom::NE, AAAtom::CZ, AAAtom::NH1, AAAtom::NH2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    SER: "SER", 'S', 15, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::OG, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    THR: "THR", 'T', 16, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::OG1, AAAtom::CG2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    VAL: "VAL", 'V', 17, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG1, AAAtom::CG2, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
    TRP: "TRP", 'W', 18, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD1, AAAtom::CD2, AAAtom::CE2, AAAtom::CE3, AAAtom::NE1, AAAtom::CZ2, AAAtom::CZ3, AAAtom::CH2],
    TYR: "TYR", 'Y', 19, [0.0, 0.0], [AAAtom::N, AAAtom::CA, AAAtom::C, AAAtom::O, AAAtom::CB, AAAtom::CG, AAAtom::CD1, AAAtom::CD2, AAAtom::CE1, AAAtom::CE2, AAAtom::CZ, AAAtom::OH, AAAtom::Unknown, AAAtom::Unknown],
    UNK: "UNK", 'X', 20, [0.0, 0.0], [AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown, AAAtom::Unknown],
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_test_data::TestFile;
    use pdbtbx;

    #[test]
    fn test_residue_codes() {
        let ala = Residue::ALA;
        assert_eq!(ala.code3(), "ALA");
        assert_eq!(ala.code1(), 'A');
        assert_eq!(ala.to_int(), 0);
    }

    #[test]
    fn test_residue_from_int() {
        assert!(matches!(Residue::from_int(0), Residue::ALA));
        assert!(matches!(Residue::from_int(1), Residue::CYS));
        assert!(matches!(Residue::from_int(999), Residue::UNK));
    }

    #[test]
    fn test_residue_atoms() {
        let trp = Residue::TRP;
        let atoms = trp.atoms14();
        assert_eq!(atoms[0], AAAtom::N);
        assert_eq!(atoms[13], AAAtom::CH2);

        let gly = Residue::GLY;
        let atoms = gly.atoms14();
        assert_eq!(atoms[4], AAAtom::Unknown);
    }

    #[test]
    fn test_atom_backbone_tensor() {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
        let ac = AtomCollection::from(&pdb);
        let ac_backbone_tensor: Tensor = ac.to_numeric_backbone_atoms(&device).expect("REASON");
        // 154 residues; N/CA/C/O; positions
        assert_eq!(ac_backbone_tensor.dims(), &[154, 4, 3]);

        // Check my residue coords in the Tensor
        // ATOM   1    N  N   . MET A 1 1   ? 24.277 8.374   -9.854  1.00 38.41  ? 0   MET A N   1
        // ATOM   2    C  CA  . MET A 1 1   ? 24.404 9.859   -9.939  1.00 37.90  ? 0   MET A CA  1
        // ATOM   3    C  C   . MET A 1 1   ? 25.814 10.249  -10.359 1.00 36.65  ? 0   MET A C   1
        // ATOM   4    O  O   . MET A 1 1   ? 26.748 9.469   -10.197 1.00 37.13  ? 0   MET A O   1
        let backbone_coords = [
            // Methionine - AA00
            ("N", (0, 0, ..), vec![24.277, 8.374, -9.854]),
            ("CA", (0, 1, ..), vec![24.404, 9.859, -9.939]),
            ("C", (0, 2, ..), vec![25.814, 10.249, -10.359]),
            ("O", (0, 3, ..), vec![26.748, 9.469, -10.197]),
            // Valine - AA01
            ("N", (1, 0, ..), vec![25.964, 11.453, -10.903]),
            ("CA", (1, 1, ..), vec![27.263, 11.924, -11.359]),
            ("C", (1, 2, ..), vec![27.392, 13.428, -11.115]),
            ("O", (1, 3, ..), vec![26.443, 14.184, -11.327]),
            // Glycing - AAlast
            ("N", (153, 0, ..), vec![23.474, -3.227, 5.994]),
            ("CA", (153, 1, ..), vec![22.818, -2.798, 7.211]),
            ("C", (153, 2, ..), vec![22.695, -1.282, 7.219]),
            ("O", (153, 3, ..), vec![21.870, -0.745, 7.992]),
        ];

        for (atom_name, (i, j, k), expected) in backbone_coords {
            let actual: Vec<f32> = ac_backbone_tensor.i((i, j, k)).unwrap().to_vec1().unwrap();
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }
    }

    #[test]
    fn test_all_atom37_tensor() {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
        let ac = AtomCollection::from(&pdb);
        let ac_backbone_tensor: Tensor = ac.to_numeric_atom37(&device).expect("REASON");
        // 154 residues; N/CA/C/O; positions
        assert_eq!(ac_backbone_tensor.dims(), &[154, 37, 3]);

        // Check my residue coords in the Tensor
        // ATOM   1    N  N   . MET A 1 1   ? 24.277 8.374   -9.854  1.00 38.41  ? 0   MET A N   1
        // ATOM   2    C  CA  . MET A 1 1   ? 24.404 9.859   -9.939  1.00 37.90  ? 0   MET A CA  1
        // ATOM   3    C  C   . MET A 1 1   ? 25.814 10.249  -10.359 1.00 36.65  ? 0   MET A C   1
        // ATOM   4    O  O   . MET A 1 1   ? 26.748 9.469   -10.197 1.00 37.13  ? 0   MET A O   1
        // ATOM   5    C  CB  . MET A 1 1   ? 24.070 10.495  -8.596  1.00 39.58  ? 0   MET A CB  1
        // ATOM   6    C  CG  . MET A 1 1   ? 24.880 9.939   -7.442  1.00 41.49  ? 0   MET A CG  1
        // ATOM   7    S  SD  . MET A 1 1   ? 24.262 10.555  -5.873  1.00 44.70  ? 0   MET A SD  1
        // ATOM   8    C  CE  . MET A 1 1   ? 24.822 12.266  -5.967  1.00 41.59  ? 0   MET A CE  1
        //
        // pub enum AAAtom {
        //     N = 0,    CA = 1,   C = 2,    CB = 3,   O = 4,
        //     CG = 5,   CG1 = 6,  CG2 = 7,  OG = 8,   OG1 = 9,
        //     SG = 10,  CD = 11,  CD1 = 12, CD2 = 13, ND1 = 14,
        //     ND2 = 15, OD1 = 16, OD2 = 17, SD = 18,  CE = 19,
        //     CE1 = 20, CE2 = 21, CE3 = 22, NE = 23,  NE1 = 24,
        //     NE2 = 25, OE1 = 26, OE2 = 27, CH2 = 28, NH1 = 29,
        //     NH2 = 30, OH = 31,  CZ = 32,  CZ2 = 33, CZ3 = 34,
        //     NZ = 35,  OXT = 36,
        //     Unknown = -1,
        // }
        let allatom_coords = [
            // Methionine - AA00
            // We iterate through these positions. Not all AA's have each
            ("N", (0, 0, ..), vec![24.277, 8.374, -9.854]),
            ("CA", (0, 1, ..), vec![24.404, 9.859, -9.939]),
            ("C", (0, 2, ..), vec![25.814, 10.249, -10.359]),
            ("CB", (0, 3, ..), vec![24.070, 10.495, -8.596]),
            ("O", (0, 4, ..), vec![26.748, 9.469, -10.197]),
            ("CG", (0, 5, ..), vec![24.880, 9.939, -7.442]),
            ("CG1", (0, 6, ..), vec![0.0, 0.0, 0.0]),
            ("CG2", (0, 7, ..), vec![0.0, 0.0, 0.0]),
            ("OG", (0, 8, ..), vec![0.0, 0.0, 0.0]),
            ("OG1", (0, 9, ..), vec![0.0, 0.0, 0.0]),
            ("SG", (0, 10, ..), vec![0.0, 0.0, 0.0]),
            ("CD", (0, 11, ..), vec![0.0, 0.0, 0.0]),
            ("CD1", (0, 12, ..), vec![0.0, 0.0, 0.0]),
            ("CD2", (0, 13, ..), vec![0.0, 0.0, 0.0]),
            ("ND1", (0, 14, ..), vec![0.0, 0.0, 0.0]),
            ("ND2", (0, 15, ..), vec![0.0, 0.0, 0.0]),
            ("OD1", (0, 16, ..), vec![0.0, 0.0, 0.0]),
            ("OD2", (0, 17, ..), vec![0.0, 0.0, 0.0]),
            ("SD", (0, 18, ..), vec![24.262, 10.555, -5.873]),
            ("CE", (0, 19, ..), vec![24.822, 12.266, -5.967]),
            ("CE1", (0, 20, ..), vec![0.0, 0.0, 0.0]),
            ("CE2", (0, 21, ..), vec![0.0, 0.0, 0.0]),
            ("CE3", (0, 22, ..), vec![0.0, 0.0, 0.0]),
            ("NE", (0, 23, ..), vec![0.0, 0.0, 0.0]),
            ("NE1", (0, 24, ..), vec![0.0, 0.0, 0.0]),
            ("NE2", (0, 25, ..), vec![0.0, 0.0, 0.0]),
            ("OE1", (0, 26, ..), vec![0.0, 0.0, 0.0]),
            ("OE2", (0, 27, ..), vec![0.0, 0.0, 0.0]),
            ("CH2", (0, 28, ..), vec![0.0, 0.0, 0.0]),
            ("NH1", (0, 29, ..), vec![0.0, 0.0, 0.0]),
            ("NH2", (0, 30, ..), vec![0.0, 0.0, 0.0]),
            ("OH", (0, 31, ..), vec![0.0, 0.0, 0.0]),
            ("CZ", (0, 32, ..), vec![0.0, 0.0, 0.0]),
            ("CZ2", (0, 33, ..), vec![0.0, 0.0, 0.0]),
            ("CZ3", (0, 34, ..), vec![0.0, 0.0, 0.0]),
            ("NZ", (0, 35, ..), vec![0.0, 0.0, 0.0]),
            ("OXT", (0, 36, ..), vec![0.0, 0.0, 0.0]),
        ];
        for (atom_name, (i, j, k), expected) in allatom_coords {
            let actual: Vec<f32> = ac_backbone_tensor.i((i, j, k)).unwrap().to_vec1().unwrap();
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }
    }

    #[test]
    fn test_ligand_tensor() {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
        let ac = AtomCollection::from(&pdb);
        let (ligand_coords, ligand_elements, _) =
            ac.to_numeric_ligand_atoms(&device).expect("REASON");

        // 54 residues; N/CA/C/O; positions
        assert_eq!(ligand_coords.dims(), &[54, 3]);

        // Check my residue coords in the Tensor
        //
        // HETATM 1222 S  S   . SO4 B 2 .   ? 30.746 18.706  28.896  1.00 47.98  ? 157 SO4 A S   1
        // HETATM 1223 O  O1  . SO4 B 2 .   ? 30.697 20.077  28.620  1.00 48.06  ? 157 SO4 A O1  1
        // HETATM 1224 O  O2  . SO4 B 2 .   ? 31.104 18.021  27.725  1.00 47.52  ? 157 SO4 A O2  1
        // HETATM 1225 O  O3  . SO4 B 2 .   ? 29.468 18.179  29.331  1.00 47.79  ? 157 SO4 A O3  1
        // HETATM 1226 O  O4  . SO4 B 2 .   ? 31.722 18.578  29.881  1.00 47.85  ? 157 SO4 A O4  1
        let allatom_coords = [
            ("S", (0, ..), vec![30.746, 18.706, 28.896]),
            ("O1", (1, ..), vec![30.697, 20.077, 28.620]),
            ("O2", (2, ..), vec![31.104, 18.021, 27.725]),
            ("O3", (3, ..), vec![29.468, 18.179, 29.331]),
            ("O4", (4, ..), vec![31.722, 18.578, 29.881]),
        ];

        for (atom_name, (i, j), expected) in allatom_coords {
            let actual: Vec<f32> = ligand_coords.i((i, j)).unwrap().to_vec1().unwrap();
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }

        // Now check the elements
        let elements: Vec<&str> = ligand_elements
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(|elem| Element::new(elem as usize).unwrap().symbol())
            .collect();

        assert_eq!(elements[0], "S");
        assert_eq!(elements[1], "O");
        assert_eq!(elements[2], "O");
        assert_eq!(elements[3], "O");
    }

    #[test]
    fn test_backbone_tensor() {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(pdb_file).unwrap();
        let ac = AtomCollection::from(&pdb);
        let xyz_37 = ac
            .to_numeric_atom37(&device)
            .expect("XYZ creation for all-atoms");
        assert_eq!(xyz_37.dims(), [154, 37, 3]);

        let xyz_m = create_backbone_mask_37(&xyz_37).expect("masking procedure should work");
        assert_eq!(xyz_m.dims(), &[154, 1]);
    }
}
