use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::encoding::one_hot;
use strum::{Display, EnumIter, EnumString};

#[rustfmt::skip]
// todo: better utility library
pub fn aa3to1(aa: &str) -> char {
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
// todo: better utility library
pub fn aa1to_int(aa: char) -> u32 {
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

pub fn cat_neighbors_nodes(
    h_nodes: &Tensor,
    h_neighbors: &Tensor,
    e_idx: &Tensor,
) -> Result<Tensor> {
    let h_nodes_gathered = gather_nodes(h_nodes, e_idx)?;
    // todo: fix this hacky Dtype
    Tensor::cat(
        &[h_neighbors, &h_nodes_gathered.to_dtype(DType::F32)?],
        D::Minus1,
    )
}

/// Retrieve the nearest Neighbor of a set of coordinates.
/// Usually used for CA carbon distance.
pub fn compute_nearest_neighbors(
    coords: &Tensor,
    mask: &Tensor,
    k: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (batch_size, seq_len,_) = coords.dims3()?;

    // broadcast_matmul handles broadcasting automatically
    // [2, 3, 1] × [2, 1, 3] -> [2, 3, 3]
    let mask_2d = mask
        .unsqueeze(2)?
        .broadcast_matmul(&mask.unsqueeze(1)?)?
        .to_dtype(DType::F64)?; // Convert to f64 once, at the start

    // Compute pairwise distances with broadcasting
    let distances = (coords
        .unsqueeze(2)?
        .broadcast_sub(&coords.unsqueeze(1)?)?
        .powf(2.)?
        .sum(D::Minus1)?
        + eps as f64)?
        .sqrt()?;

    // Apply mask
    // Get max values for adjustment
    let masked_distances = (&distances * &mask_2d.to_dtype(DType::F32)?)?;
    // println!("after masked_distances");
    let d_max = masked_distances.max_keepdim(D::Minus1)?;
    let mask_term = ((&mask_2d.to_dtype(DType::F32)? * -1.0)? + 1.0)?;
    let d_adjust = (&masked_distances + mask_term.broadcast_mul(&d_max)?)?;
    let d_adjust = d_adjust.to_dtype(DType::F32)?;

    Ok(topk_last_dim(&d_adjust, k.min(seq_len))?)
}

// https://github.com/huggingface/candle/pull/2375/files#diff-e4d52a71060a80ac8c549f2daffcee77f9bf4de8252ad067c47b1c383c3ac828R957
pub fn topk_last_dim(xs: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let sorted_indices = xs.arg_sort_last_dim(false)?.to_dtype(DType::I64)?;
    let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
    Ok((xs.gather(&topk_indices, D::Minus1)?, topk_indices))
}

/// Input coords. Output 1 <batch  x 1 > Tensor
/// representing whether each residue has all 4 backbone atoms.
/// note that the internal ordering is different between
/// backbone only [N/CA/C/O] and all-atom [N/CA/C/CB/O]....
pub fn create_backbone_mask_37(xyz_37: &Tensor) -> Result<Tensor> {
    let backbone_indices = Tensor::new(&[0u32, 1, 2, 4], xyz_37.device())?;
    let backbone_selection = xyz_37.index_select(&backbone_indices, 1)?; // [154, 4, 3]
                                                                         // Check if coordinates exist (sum over xyz dimensions)
    let exists = backbone_selection.sum(2)?; // [154, 4]

    // All 4 atoms must exist
    let all_exist = exists.sum_keepdim(1)?; // [154, 1]
    Ok(all_exist)
}

/// Get Pseudo CB
pub fn calculate_cb(xyz_37: &Tensor) -> Result<Tensor> {
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

/// Custom Cross-Product Fn.
pub fn cross_product(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let last_dim = a.dims().len() - 1;

    // Extract components
    let a0 = a.narrow(last_dim, 0, 1)?;
    let a1 = a.narrow(last_dim, 1, 1)?;
    let a2 = a.narrow(last_dim, 2, 1)?;

    let b0 = b.narrow(last_dim, 0, 1)?;
    let b1 = b.narrow(last_dim, 1, 1)?;
    let b2 = b.narrow(last_dim, 2, 1)?;

    // Compute cross product components
    let c0 = ((&a1 * &b2)? - (&a2 * &b1)?)?;
    let c1 = ((&a2 * &b0)? - (&a0 * &b2)?)?;
    let c2 = ((&a0 * &b1)? - (&a1 * &b0)?)?;

    // Stack the results
    Tensor::cat(&[&c0, &c1, &c2], last_dim)
}

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

/// Gather_edges
/// Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
pub fn gather_edges(edges: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    let (d1, d2, d3) = neighbor_idx.dims3()?;
    let neighbors =
        neighbor_idx
            .unsqueeze(D::Minus1)?
            .expand((d1, d2, d3, edges.dim(D::Minus1)?))?;

    // println!("Neighbors idx: {:?}", neighbors.dims());
    let edge_gather = edges.gather(&neighbors, 2)?;
    // println!("edge_gather idx: {:?}", edge_gather.dims());
    Ok(edge_gather)
}

/// Gather Nodes
///
/// Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
/// Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
pub fn gather_nodes(nodes: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    let (batch_size, n_nodes, n_features) = nodes.dims3()?;
    let (_, _, k_neighbors) = neighbor_idx.dims3()?;

    // Reshape neighbor_idx to [B, N*K]
    let neighbors_flat = neighbor_idx.reshape((batch_size, n_nodes * k_neighbors))?;

    // Add feature dimension and expand
    let neighbors_flat = neighbors_flat
        .unsqueeze(2)? // Add feature dimension [B, N*K, 1]
        .expand((batch_size, n_nodes * k_neighbors, n_features))?; // Expand to [B, N*K, C]

    // make contiguous for the gather.
    let neighbors_flat = neighbors_flat.contiguous()?;
    // Gather features
    let neighbor_features = nodes.gather(&neighbors_flat, 1)?;

    println!(
        "neighbor_features dims before final reshape: {:?}",
        neighbor_features.dims()
    );

    // Reshape back to [B, N, K, C]
    neighbor_features.reshape((batch_size, n_nodes, k_neighbors, n_features))
}

pub fn gather_nodes_t(nodes: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    // Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    let (d1, d2, d3) = nodes.dims3()?;
    let idx_flat = neighbor_idx.unsqueeze(D::Minus1)?.expand((d1, d2, d3))?;
    nodes.gather(&idx_flat, 1)
}

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

fn get_seq_rec(s: &Tensor, s_pred: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // S: true sequence shape=[batch, length]
    // S_pred: predicted sequence shape=[batch, length]
    // mask: mask to compute average over the region shape=[batch, length]
    // Returns: averaged sequence recovery shape=[batch]
    //
    // Compute the match tensor
    let match_tensor = s.eq(s_pred)?;
    let match_f32 = match_tensor.to_dtype(DType::F32)?;
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

pub fn linspace(start: f64, stop: f64, steps: usize, device: &Device) -> Result<Tensor> {
    if steps == 0 {
        Tensor::from_vec(Vec::<f64>::new(), steps, device)
    } else if steps == 1 {
        Tensor::from_vec(vec![start], steps, device)
    } else {
        let delta = (stop - start) / (steps - 1) as f64;
        let vs = (0..steps)
            .map(|step| start + step as f64 * delta)
            .collect::<Vec<_>>();
        Tensor::from_vec(vs, steps, device)
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
    use crate::models::ligandmpnn::featurizer::LMPNNFeatures;
    use ferritin_core::AtomCollection;
    use ferritin_test_data::TestFile;
    use pdbtbx;
    use pdbtbx::Element;

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
    #[test]
    fn test_compute_nearest_neighbors() {
        let device = Device::Cpu;

        // Create a simple 2x3x3 tensor representing 2 sequences of 3 points in 3D space
        let coords = Tensor::new(
            &[
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], // First sequence
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], // Second sequence
            ],
            &device,
        )
        .unwrap();

        // Create mask indicating all points are valid
        let mask = Tensor::ones((2, 3), DType::F32, &device).unwrap();

        // Get 2 nearest neighbors for each point
        let (distances, indices) = compute_nearest_neighbors(&coords, &mask, 2, 1e-6).unwrap();

        // Check shapes
        assert_eq!(distances.dims(), &[2, 3, 2]); // [batch, seq_len, k]
        assert_eq!(indices.dims(), &[2, 3, 2]); // [batch, seq_len, k]

        // For first sequence, point [1,0,0] should have [0,0,0] and [2,0,0] as nearest neighbors
        let point_neighbors: Vec<i64> = indices.i((0, 1, ..)).unwrap().to_vec1().unwrap();
        assert_eq!(point_neighbors, vec![0, 2]);

        // Check distances are correct
        let point_distances: Vec<f32> = distances.i((0, 1, ..)).unwrap().to_vec1().unwrap();
        assert!((point_distances[0] - 1.0).abs() < 1e-5);
        assert!((point_distances[1] - 1.0).abs() < 1e-5);
    }
}
