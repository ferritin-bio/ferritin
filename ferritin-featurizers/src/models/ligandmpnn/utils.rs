// Note: this is currently a direct Python --> Rust translation using Torch.
// I'd like to move this to Candle

use candle_core::{Result, Tensor, D};
use candle_nn::encoding::one_hot;
use ferritin_core::AtomCollection;
use std::collections::HashMap;

/// Gather_edges
/// Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
pub fn gather_edges(edges: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    let (d1, d2, d3) = neighbor_idx.dims3()?;
    let neighbors =
        neighbor_idx
            .unsqueeze(D::Minus1)?
            .expand((d1, d2, d3, edges.dim(D::Minus1)?))?;
    edges.gather(&neighbors, 2)
}

pub fn gather_nodes(nodes: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    // Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    // Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    let neighbors_flat =
        neighbor_idx.reshape((neighbor_idx.dim(0)?, neighbor_idx.dim(D::Minus1)?))?;
    let (d1, d2, d3) = neighbor_idx.dims3()?;
    let neighbors_flat = neighbors_flat
        .unsqueeze(D::Minus1)?
        .expand((d1, d2, nodes.dim(2)?))?;
    let neighbor_features = nodes.gather(nodes, 1)?;
    neighbor_features.reshape((d1, d2, d3, neighbor_features.dim(D::Minus1)?))
}

fn gather_nodes_t(nodes: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    // Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    let (d1, d2, d3) = nodes.dims3()?;
    let idx_flat = neighbor_idx.unsqueeze(D::Minus1)?.expand((d1, d2, d3))?;
    nodes.gather(&idx_flat, 1)
}

pub fn cat_neighbors_nodes(
    h_nodes: &Tensor,
    h_neighbors: &Tensor,
    e_idx: &Tensor,
) -> Result<Tensor> {
    let h_nodes_gathered = gather_nodes(h_nodes, e_idx)?;
    Tensor::cat(&[h_neighbors, &h_nodes_gathered], D::Minus1)
}

fn restype_1to3(code: char) -> &'static str {
    match code {
        'A' => "ALA",
        'R' => "ARG",
        'N' => "ASN",
        'D' => "ASP",
        'C' => "CYS",
        'Q' => "GLN",
        'E' => "GLU",
        'G' => "GLY",
        'H' => "HIS",
        'I' => "ILE",
        'L' => "LEU",
        'K' => "LYS",
        'M' => "MET",
        'F' => "PHE",
        'P' => "PRO",
        'S' => "SER",
        'T' => "THR",
        'W' => "TRP",
        'Y' => "TYR",
        'V' => "VAL",
        _ => "UNK",
    }
}

fn restype_3to1(restype: &str) -> char {
    match restype {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        _ => 'X',
    }
}

pub fn restype_str_to_int(code: char) -> i32 {
    match code.to_ascii_uppercase() {
        'A' => 0,
        'C' => 1,
        'D' => 2,
        'E' => 3,
        'F' => 4,
        'G' => 5,
        'H' => 6,
        'I' => 7,
        'K' => 8,
        'L' => 9,
        'M' => 10,
        'N' => 11,
        'P' => 12,
        'Q' => 13,
        'R' => 14,
        'S' => 15,
        'T' => 16,
        'V' => 17,
        'W' => 18,
        'Y' => 19,
        'X' => 20,
        _ => 20, // Default to 'X' (20) for any unrecognized character
    }
}

fn restype_int_to_str(code: i32) -> &'static str {
    match code {
        0 => "A",
        1 => "C",
        2 => "D",
        3 => "E",
        4 => "F",
        5 => "G",
        6 => "H",
        7 => "I",
        8 => "K",
        9 => "L",
        10 => "M",
        11 => "N",
        12 => "P",
        13 => "Q",
        14 => "R",
        15 => "S",
        16 => "T",
        17 => "V",
        18 => "W",
        19 => "Y",
        _ => "X", // Default to 'X' for any unrecognized integer
    }
}

#[rustfmt::skip]
fn restype_name_to_atom14_names(restype: &str) -> [&'static str; 14] {
    match restype {
        "ALA" => ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", "",],
        "ARG" => ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", "",],
        "ASN" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", "",],
        "ASP" => ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", "",],
        "CYS" => ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", "",],
        "GLN" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", "",],
        "GLU" => ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", "",],
        "GLY" => ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", "",],
        "HIS" => ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", "",],
        "ILE" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", "",],
        "LEU" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", "",],
        "LYS" => ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", "",],
        "MET" => ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", "",],
        "PHE" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", "",],
        "PRO" => ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", "",],
        "SER" => ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", "",],
        "THR" => ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", "",],
        "TRP" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2",],
        "TYR" => ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", "",],
        "VAL" => ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", "",],
        _ => ["", "", "", "", "", "", "", "", "", "", "", "", "", "",], // UNK or any other case
    }
}

#[rustfmt::skip]
fn atom_order(atom: &str) -> i32 {
    match atom {
        "N" => 0,
        "CA" => 1,
        "C" => 2,
        "CB" => 3,
        "O" => 4,
        "CG" => 5,
        "CG1" => 6,
        "CG2" => 7,
        "OG" => 8,
        "OG1" => 9,
        "SG" => 10,
        "CD" => 11,
        "CD1" => 12,
        "CD2" => 13,
        "ND1" => 14,
        "ND2" => 15,
        "OD1" => 16,
        "OD2" => 17,
        "SD" => 18,
        "CE" => 19,
        "CE1" => 20,
        "CE2" => 21,
        "CE3" => 22,
        "NE" => 23,
        "NE1" => 24,
        "NE2" => 25,
        "OE1" => 26,
        "OE2" => 27,
        "CH2" => 28,
        "NH1" => 29,
        "NH2" => 30,
        "OH" => 31,
        "CZ" => 32,
        "CZ2" => 33,
        "CZ3" => 34,
        "NZ" => 35,
        "OXT" => 36,
        _ => -1, // Return -1 for unknown atoms
    }
}

const ALPHABET: [char; 21] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y', 'X',
];

const ELEMENT_LIST: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk",
    "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
    "Fl", "Mc", "Lv", "Ts", "Og",
];

// element_list = [item.upper() for item in element_list]
// element_dict = dict(zip(element_list, range(1,len(element_list))))
// element_dict_rev = dict(zip(range(1, len(element_list)), element_list))

fn get_seq_rec(s: &Tensor, s_pred: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // S: true sequence shape=[batch, length]
    // S_pred: predicted sequence shape=[batch, length]
    // mask: mask to compute average over the region shape=[batch, length]
    // Returns: averaged sequence recovery shape=[batch]
    //
    // Compute the match tensor
    let match_tensor = s.eq(s_pred)?;
    let match_f32 = match_tensor.to_dtype(candle_core::DType::F32)?;
    let numerator = (match_f32 * mask)?.sum_keepdim(1)?;
    let denominator = mask.sum_keepdim(1)?;
    let average = numerator.broadcast_div(&denominator)?;
    // Remove the last dimension to get shape=[batch]
    average.squeeze(1)
}

fn get_score(s: &Tensor, log_probs: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor)> {
    //     S : true sequence shape=[batch, length]
    //     log_probs : predicted sequence shape=[batch, length]
    //     mask : mask to compute average over the region shape=[batch, length]
    //     average_loss : averaged categorical cross entropy (CCE) [batch]
    //     loss_per_resdue : per position CCE [batch, length]

    //     """
    //     S_one_hot = torch.nn.functional.one_hot(S, 21)
    //     loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    //     average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
    //         torch.sum(mask, dim=-1) + 1e-8
    //     )
    //     return average_loss, loss_per_residue

    // S: true sequence shape=[batch, length]
    // log_probs: predicted sequence shape=[batch, length, 21]
    // mask: mask to compute average over the region shape=[batch, length]
    // Returns:
    //   - average_loss: averaged categorical cross entropy (CCE) [batch]
    //   - loss_per_residue: per position CCE [batch, length]

    // Create one-hot encoding of S.
    // see https://docs.rs/candle-nn/0.7.2/candle_nn/encoding/fn.one_hot.html
    // this could be wrong...
    let s_one_hot = one_hot(s.clone(), 21, 1., 0.)?;
    let loss_per_residue = s_one_hot.mul(&log_probs.neg()?)?.sum(D::Minus1)?;
    let average_loss = loss_per_residue
        .mul(&mask)?
        .sum_keepdim(D::Minus1)?
        .div(&(mask.sum_keepdim(D::Minus1)? + 1e-8f64)?)?
        .squeeze(D::Minus1)?;

    Ok((average_loss, loss_per_residue))
}

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

// Todo: finish this
// pub fn get_aligned_coordinates(
//     protein_atoms: AtomCollection,
//     ca_dict: &HashMap<String, usize>,
//     atom_name: &str,
// ) -> Result<(Tensor, Tensor)> {
//     let atom_atoms = protein_atoms.select(&format!("name {}", atom_name));

//     let (atom_coords, atom_resnums, atom_chain_ids, atom_icodes) = if let Some(atoms) = atom_atoms {
//         (
//             atoms.get_coords()?,
//             atoms.get_resnums()?,
//             atoms.get_chids()?,
//             atoms.get_icodes()?,
//         )
//     } else {
//         (
//             Tensor::zeros((0, 3), DType::F32, &Device::Cpu)?,
//             Tensor::zeros((0,), DType::I64, &Device::Cpu)?,
//             Tensor::zeros((0,), DType::I64, &Device::Cpu)?,
//             Tensor::zeros((0,), DType::I64, &Device::Cpu)?,
//         )
//     };

//     let mut atom_coords_ = Tensor::zeros((ca_dict.len(), 3), DType::F32, &Device::Cpu)?;
//     let mut atom_coords_m = Tensor::zeros((ca_dict.len(),), DType::I32, &Device::Cpu)?;

//     if let Some(atoms) = atom_atoms {
//         for i in 0..atom_resnums.dim(0)? {
//             let code = format!(
//                 "{}_{}{}",
//                 atom_chain_ids.get(i)?,
//                 atom_resnums.get(i)?,
//                 atom_icodes.get(i)?
//             );
//             if let Some(&idx) = ca_dict.get(&code) {
//                 atom_coords_
//                     .slice_mut(1, idx, idx + 1)?
//                     .copy_(&atom_coords.slice(1, i, i + 1)?)?;
//                 atom_coords_m.set(idx, 1)?;
//             }
//         }
//     }

//     Ok((atom_coords_, atom_coords_m))
// }

// pub fn get_nearest_neighbours(
//     cb: &Tensor,
//     mask: &Tensor,
//     y: &Tensor,
//     y_t: &Tensor,
//     y_m: &Tensor,
//     number_of_ligand_atoms: i64,
// ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
//     let device = cb.device();
//     let mask_cby = mask.unsqueeze(1)?.broadcast_mul(&y_m.unsqueeze(0)?)?; // [A,B]
//     let l2_ab = ((cb.unsqueeze(1)? - y.unsqueeze(0)?).powf(2.0)?)?.sum_keepdim(D::Minus1)?;
//     let l2_ab = l2_ab.broadcast_mul(&mask_cby)? + (mask_cby.neg()?.add(1.0)?)?.mul(1000.0)?;

//     let nn_idx = l2_ab
//         .argsort(D::Minus1, false)?
//         .narrow(D::Minus1, 0, number_of_ligand_atoms)?;
//     let l2_ab_nn = l2_ab.gather(&nn_idx, 1)?;
//     let d_ab_closest = l2_ab_nn.narrow(D::Minus1, 0, 1)?.sqrt()?;

//     let y_r = y.unsqueeze(0)?.expand((cb.dim(0)?, y.dim(0)?, y.dim(1)?))?;
//     let y_t_r = y_t.unsqueeze(0)?.expand((cb.dim(0)?, y_t.dim(0)?))?;
//     let y_m_r = y_m.unsqueeze(0)?.expand((cb.dim(0)?, y_m.dim(0)?))?;

//     let y_tmp = y_r.gather(
//         &nn_idx
//             .unsqueeze(D::Minus1)?
//             .expand((cb.dim(0)?, number_of_ligand_atoms, 3))?,
//         1,
//     )?;
//     let y_t_tmp = y_t_r.gather(&nn_idx, 1)?;
//     let y_m_tmp = y_m_r.gather(&nn_idx, 1)?;

//     let mut y = Tensor::zeros((cb.dim(0)?, number_of_ligand_atoms, 3), DType::F32, device)?;
//     let mut y_t = Tensor::zeros((cb.dim(0)?, number_of_ligand_atoms), DType::I32, device)?;
//     let mut y_m = Tensor::zeros((cb.dim(0)?, number_of_ligand_atoms), DType::I32, device)?;

//     let num_nn_update = y_tmp.dim(1)?;
//     y.narrow(1, 0, num_nn_update)?.copy_(&y_tmp)?;
//     y_t.narrow(1, 0, num_nn_update)?.copy_(&y_t_tmp)?;
//     y_m.narrow(1, 0, num_nn_update)?.copy_(&y_m_tmp)?;

//     Ok((y, y_t, y_m, d_ab_closest))
// }

// pub fn featurize(
//     input_dict: &HashMap<String, Tensor>,
//     cutoff_for_score: f32,
//     use_atom_context: bool,
//     number_of_ligand_atoms: i64,
//     model_type: &str,
// ) -> Result<HashMap<String, Tensor>> {
//     let mut output_dict = HashMap::new();
//     if model_type == "ligand_mpnn" {
//         let mask = input_dict.get("mask").unwrap();
//         let y = input_dict.get("Y").unwrap();
//         let y_t = input_dict.get("Y_t").unwrap();
//         let y_m = input_dict.get("Y_m").unwrap();
//         let n = input_dict.get("X").unwrap().slice(1, 0, 1, 1)?;
//         let ca = input_dict.get("X").unwrap().slice(1, 1, 2, 1)?;
//         let c = input_dict.get("X").unwrap().slice(1, 2, 3, 1)?;
//         let b = &ca - &n;
//         let c = &c - &ca;
//         let a = b.cross(&c)?;
//         let cb = &(-0.58273431 * &a + 0.56802827 * &b - 0.54067466 * &c) + &ca;
//         let (y, y_t, y_m, d_xy) =
//             get_nearest_neighbours(&cb, mask, y, y_t, y_m, number_of_ligand_atoms)?;
//         let mask_xy = (&d_xy.lt(cutoff_for_score)? * mask * &y_m.slice(1, 0, 1, 1)?)?;
//         output_dict.insert("mask_XY".to_string(), mask_xy.unsqueeze(0)?);
//         if input_dict.contains_key("side_chain_mask") {
//             output_dict.insert(
//                 "side_chain_mask".to_string(),
//                 input_dict.get("side_chain_mask").unwrap().unsqueeze(0)?,
//             );
//         }
//         output_dict.insert("Y".to_string(), y.unsqueeze(0)?);
//         output_dict.insert("Y_t".to_string(), y_t.unsqueeze(0)?);
//         output_dict.insert("Y_m".to_string(), y_m.unsqueeze(0)?);
//         if !use_atom_context {
//             output_dict.insert("Y_m".to_string(), (&output_dict["Y_m"] * 0.0)?);
//         }
//     } else if model_type == "per_residue_label_membrane_mpnn"
//         || model_type == "global_label_membrane_mpnn"
//     {
//         output_dict.insert(
//             "membrane_per_residue_labels".to_string(),
//             input_dict
//                 .get("membrane_per_residue_labels")
//                 .unwrap()
//                 .unsqueeze(0)?,
//         );
//     }

//     let r_idx = input_dict.get("R_idx").unwrap();
//     let mut r_idx_list = Vec::new();
//     let mut count = 0;
//     let mut r_idx_prev = -100000;
//     for &r_idx_val in r_idx.iter::<i64>()?.iter() {
//         if r_idx_prev == r_idx_val {
//             count += 1;
//         }
//         r_idx_list.push(r_idx_val + count);
//         r_idx_prev = r_idx_val;
//     }
//     let r_idx_renumbered = Tensor::from_slice(&r_idx_list, r_idx.device())?;
//     output_dict.insert("R_idx".to_string(), r_idx_renumbered.unsqueeze(0)?);
//     output_dict.insert("R_idx_original".to_string(), r_idx.unsqueeze(0)?);
//     output_dict.insert(
//         "chain_labels".to_string(),
//         input_dict.get("chain_labels").unwrap().unsqueeze(0)?,
//     );
//     output_dict.insert("S".to_string(), input_dict.get("S").unwrap().unsqueeze(0)?);
//     output_dict.insert(
//         "chain_mask".to_string(),
//         input_dict.get("chain_mask").unwrap().unsqueeze(0)?,
//     );
//     output_dict.insert(
//         "mask".to_string(),
//         input_dict.get("mask").unwrap().unsqueeze(0)?,
//     );

//     output_dict.insert("X".to_string(), input_dict.get("X").unwrap().unsqueeze(0)?);

//     if input_dict.contains_key("xyz_37") {
//         output_dict.insert(
//             "xyz_37".to_string(),
//             input_dict.get("xyz_37").unwrap().unsqueeze(0)?,
//         );
//         output_dict.insert(
//             "xyz_37_m".to_string(),
//             input_dict.get("xyz_37_m").unwrap().unsqueeze(0)?,
//         );
//     }
//     Ok(output_dict)
// }
