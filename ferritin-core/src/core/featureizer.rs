//! Protein Featurizer for ProteinMPNN/LigadMPNN
//!
//! Extract protein features for ligampnn
//!
//! Returns a set of features calculated from protein structure
//! including:
//! - Residue-level features like amino acid type, secondary structure
//! - Geometric features like distances, angles
//! - Chemical features like hydrophobicity, charge
//! - Evolutionary features from MSA profiles

/// Features of rLignamd MPNN
///
///
struct LigandMPNNFeatures {}

macro_rules! define_residues {
    ($($name:ident: $code3:expr, $code1:expr, $idx:expr, $features:expr),* $(,)?) => {
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
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AAAtom {
    N = 0,
    CA = 1,
    C = 2,
    CB = 3,
    O = 4,
    CG = 5,
    CG1 = 6,
    CG2 = 7,
    OG = 8,
    OG1 = 9,
    SG = 10,
    CD = 11,
    CD1 = 12,
    CD2 = 13,
    ND1 = 14,
    ND2 = 15,
    OD1 = 16,
    OD2 = 17,
    SD = 18,
    CE = 19,
    CE1 = 20,
    CE2 = 21,
    CE3 = 22,
    NE = 23,
    NE1 = 24,
    NE2 = 25,
    OE1 = 26,
    OE2 = 27,
    CH2 = 28,
    NH1 = 29,
    NH2 = 30,
    OH = 31,
    CZ = 32,
    CZ2 = 33,
    CZ3 = 34,
    NZ = 35,
    OXT = 36,
    Unknown = -1,
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
// // todo: finish this port
// pub fn parse_pdb(
//     input_path: &str,
//     device: &str,
//     chains: &[String],
//     parse_all_atoms: bool,
//     parse_atoms_with_zero_occupancy: bool,
// ) -> Result<(
//     HashMap<String, Tensor>,
//     AtomCollection,
//     AtomCollection,
//     Vec<char>,
//     AtomCollection,
// )> {
//     let element_list: Vec<String> = ELEMENT_LIST.iter().map(|&s| s.to_uppercase()).collect();
//     let element_dict: HashMap<String, usize> = element_list
//         .iter()
//         .enumerate()
//         .map(|(i, s)| (s.clone(), i + 1))
//         .collect();

//     let atom_types = if !parse_all_atoms {
//         vec!["N", "CA", "C", "O"]
//     } else {
//         vec![
//             "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
//             "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1",
//             "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ",
//         ]
//     };

//     let mut atoms = prody::parse_pdb(input_path)?;

//     if !parse_atoms_with_zero_occupancy {
//         atoms = atoms.select("occupancy > 0")?;
//     }
//     if !chains.is_empty() {
//         let str_out = chains
//             .iter()
//             .map(|item| format!(" chain {} or", item))
//             .collect::<String>();
//         atoms = atoms.select(&str_out[1..str_out.len() - 3])?;
//     }

//     let protein_atoms = atoms.select("protein")?;
//     let backbone = protein_atoms.select("backbone")?;
//     let other_atoms = atoms.select("not protein and not water")?;
//     let water_atoms = atoms.select("water")?;

//     let ca_atoms = protein_atoms.select("name CA")?;
//     let ca_resnums = ca_atoms.get_resnums()?;
//     let ca_chain_ids = ca_atoms.get_chids()?;
//     let ca_icodes = ca_atoms.get_icodes()?;

//     let mut ca_dict = HashMap::new();
//     for i in 0..ca_resnums.len() {
//         let code = format!("{}_{}{}", ca_chain_ids[i], ca_resnums[i], ca_icodes[i]);
//         ca_dict.insert(code, i);
//     }

//     let mut xyz_37 = Array3::<f32>::zeros((ca_dict.len(), 37, 3));
//     let mut xyz_37_m = Array2::<i32>::zeros((ca_dict.len(), 37));
//     for atom_name in &atom_types {
//         let (xyz, xyz_m) = get_aligned_coordinates(&protein_atoms, &ca_dict, atom_name)?;
//         xyz_37
//             .slice_mut(s![.., atom_order(atom_name), ..])
//             .assign(&xyz);
//         xyz_37_m
//             .slice_mut(s![.., atom_order(atom_name)])
//             .assign(&xyz_m);
//     }

//     let n = xyz_37.slice(s![.., atom_order("N"), ..]);
//     let ca = xyz_37.slice(s![.., atom_order("CA"), ..]);
//     let c = xyz_37.slice(s![.., atom_order("C"), ..]);
//     let o = xyz_37.slice(s![.., atom_order("O"), ..]);

//     let n_m = xyz_37_m.slice(s![.., atom_order("N")]);
//     let ca_m = xyz_37_m.slice(s![.., atom_order("CA")]);
//     let c_m = xyz_37_m.slice(s![.., atom_order("C")]);
//     let o_m = xyz_37_m.slice(s![.., atom_order("O")]);

//     let mask = &n_m * &ca_m * &c_m * &o_m;

//     let b = &ca - &n;
//     let c = &c - &ca;
//     let a = b.cross(&c);
//     let cb = -0.58273431 * &a + 0.56802827 * &b - 0.54067466 * &c + &ca;

//     let chain_labels = ca_atoms.get_chindices()?;
//     let r_idx = ca_resnums;
//     let s = ca_atoms.get_resnames()?;
//     let s: Vec<char> = s.iter().map(|aa| restype_3to1(aa).unwrap_or('X')).collect();
//     let s: Vec<i32> = s.iter().map(|&aa| restype_str_to_int(aa)).collect();
//     let x = Array3::from_shape_vec(
//         (n.shape()[0], 4, 3),
//         [n.to_vec(), ca.to_vec(), c.to_vec(), o.to_vec()].concat(),
//     )?;

//     let (y, y_t, y_m) = if let Ok(other_coords) = other_atoms.get_coords() {
//         let y = Array2::from_shape_vec((other_coords.len(), 3), other_coords)?;
//         let y_t = other_atoms.get_elements()?;
//         let y_t: Vec<i32> = y_t
//             .iter()
//             .map(|y_t| *element_dict.get(&y_t.to_uppercase()).unwrap_or(&0) as i32)
//             .collect();
//         let y_m = y_t
//             .iter()
//             .map(|&y| (y != 1) as i32 * (y != 0) as i32)
//             .collect::<Vec<i32>>();

//         let y_mask = Array1::from_vec(y_m.clone());
//         let y = y.select(Axis(0), &y_mask.mapv(|v| v == 1));
//         let y_t = Array1::from_vec(y_t).select(Axis(0), &y_mask.mapv(|v| v == 1));
//         let y_m = Array1::from_vec(y_m).select(Axis(0), &y_mask.mapv(|v| v == 1));

//         (y, y_t, y_m)
//     } else {
//         (Array2::zeros((1, 3)), Array1::zeros(1), Array1::zeros(1))
//     };

//     let mut output_dict = HashMap::new();
//     output_dict.insert("X".to_string(), Tensor::from_array(&x, device)?);
//     output_dict.insert("mask".to_string(), Tensor::from_array(&mask, device)?);
//     output_dict.insert("Y".to_string(), Tensor::from_array(&y, device)?);
//     output_dict.insert("Y_t".to_string(), Tensor::from_array(&y_t, device)?);
//     output_dict.insert("Y_m".to_string(), Tensor::from_array(&y_m, device)?);
//     output_dict.insert("R_idx".to_string(), Tensor::from_slice(&r_idx, device)?);
//     output_dict.insert(
//         "chain_labels".to_string(),
//         Tensor::from_slice(&chain_labels, device)?,
//     );
//     output_dict.insert(
//         "chain_letters".to_string(),
//         Tensor::from_slice(&ca_chain_ids, device)?,
//     );

//     let mut mask_c = Vec::new();
//     let mut chain_list: Vec<char> = ca_chain_ids.iter().cloned().collect();
//     chain_list.sort_unstable();
//     chain_list.dedup();
//     for chain in &chain_list {
//         let mask = ca_chain_ids
//             .iter()
//             .map(|&c| c == *chain)
//             .collect::<Vec<bool>>();
//         mask_c.push(Tensor::from_slice(&mask, device)?);
//     }

//     output_dict.insert("mask_c".to_string(), Tensor::stack(&mask_c, 0)?);
//     output_dict.insert(
//         "chain_list".to_string(),
//         Tensor::from_slice(&chain_list, device)?,
//     );
//     output_dict.insert("S".to_string(), Tensor::from_slice(&s, device)?);
//     output_dict.insert("xyz_37".to_string(), Tensor::from_array(&xyz_37, device)?);
//     output_dict.insert(
//         "xyz_37_m".to_string(),
//         Tensor::from_array(&xyz_37_m, device)?,
//     );

//     Ok((output_dict, backbone, other_atoms, ca_icodes, water_atoms))
// }
