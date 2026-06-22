use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
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

#[rustfmt::skip]
pub fn int_to_aa1(aa_int: u32) -> char {
    match aa_int {
        0 => 'A', 1 => 'C', 2 => 'D',
        3 => 'E', 4 => 'F', 5 => 'G',
        6 => 'H', 7 => 'I', 8 => 'K',
        9 => 'L', 10 => 'M', 11 => 'N',
        12 => 'P', 13 => 'Q', 14 => 'R',
        15 => 'S', 16 => 'T', 17 => 'V',
        18 => 'W', 19 => 'Y', 20 => 'X',
        _ => 'X'

    }
}

#[allow(dead_code)]
const ALPHABET: [char; 21] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
    'Y', 'X',
];

#[allow(dead_code)]
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
    ($($name:ident: $code3:expr_2021, $code1:expr_2021, $idx:expr_2021, $features:expr_2021, $atoms14:expr_2021),* $(,)?) => {
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

pub fn get_nearest_neighbours(
    cb: &Tensor,
    mask: &Tensor,
    y: &Tensor,
    y_t: &Tensor,
    y_m: &Tensor,
    number_of_ligand_atoms: i64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    // First, remove batch dimension if present using squeeze(0)
    let cb = cb.squeeze(0)?;
    let mask = mask.squeeze(0)?;
    let y_m = if y_m.dims().len() > 1 {
        y_m.sum_keepdim(1)?.squeeze(1)? // or .any(1)? depending on your needs
    } else {
        y_m.clone()
    };
    let num_residues = cb.dim(0)?;
    let mask_cby = mask.unsqueeze(1)?.matmul(&y_m.unsqueeze(0)?)?;
    let cb_flat = cb.reshape((cb.dim(0)?, 1, 3))?; // [154, 1, 3]
    let y_flat = y.reshape((1, y.dim(0)?, 3))?; // [1, 54, 3]
    // Try broadcasting manually if needed
    let cb_broadcast = cb_flat.broadcast_as((cb.dim(0)?, y.dim(0)?, 3))?; // [154, 54, 3]
    let y_broadcast = y_flat.broadcast_as((cb.dim(0)?, y.dim(0)?, 3))?; // [154, 54, 3]
    let diff = cb_broadcast.sub(&y_broadcast)?;
    let l2_ab = diff.powf(2.0)?.sum(D::Minus1)?;
    let complement_mask = (mask_cby.neg()? + 1.0)?;
    let padding_value = Tensor::full(1000.0_f32, mask_cby.dims(), cb.device())?;
    let masked_distances = l2_ab.mul(&mask_cby)?;
    let padding_contribution = complement_mask.mul(&padding_value)?;
    let l2_ab = masked_distances.add(&padding_contribution)?;

    // Get nearest neighbors
    let nn_idx = l2_ab
        .arg_sort_last_dim(false)?
        .narrow(1, 0, number_of_ligand_atoms as usize)?
        .contiguous()?;
    let l2_ab_nn = l2_ab.contiguous()?.gather(&nn_idx, 1)?;
    let d_ab_closest = l2_ab_nn.i((.., 0))?.sqrt()?;
    let y_new = y
        .unsqueeze(0)?
        .expand((num_residues, y.dim(0)?, 3))?
        .contiguous()?
        .gather(
            &nn_idx
                .unsqueeze(2)?
                .expand((num_residues, number_of_ligand_atoms as usize, 3))?
                .contiguous()?,
            1,
        )?;

    let y_t_new = y_t
        .unsqueeze(0)?
        .expand((num_residues, y_t.dim(0)?))?
        .contiguous()?
        .gather(&nn_idx, 1)?;

    let y_m_new = y_m
        .unsqueeze(0)?
        .expand((num_residues, y_m.dim(0)?))?
        .contiguous()?
        .gather(&nn_idx, 1)?;

    Ok((y_new, y_t_new, y_m_new, d_ab_closest))
}

pub fn cat_neighbors_nodes(
    h_nodes: &Tensor,
    h_neighbors: &Tensor,
    e_idx: &Tensor,
) -> Result<Tensor> {
    let h_nodes_gathered = gather_nodes(h_nodes, e_idx)?;
    let h_neighbors = h_neighbors.expand((
        h_neighbors.dim(0)?, // 1
        h_nodes.dim(1)?,     // 93
        h_neighbors.dim(2)?, // 24
        h_neighbors.dim(3)?, // 128
    ))?;

    Tensor::cat(
        &[h_neighbors, h_nodes_gathered.to_dtype(DType::F32)?],
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
    let (_batch_size, seq_len, _) = coords.dims3()?;
    // broadcast_matmul handles broadcasting automatically
    // [2, 3, 1] × [2, 1, 3] -> [2, 3, 3]

    let mask_2d = mask
        .unsqueeze(2)?
        .broadcast_matmul(&mask.unsqueeze(1)?)?
        .to_dtype(DType::F32)?;
    // Compute pairwise distances with broadcasting

    let distances = (coords
        .unsqueeze(2)?
        .broadcast_sub(&coords.unsqueeze(1)?)?
        .powf(2.)?
        .sum(D::Minus1)?
        + eps as f64)? // also  doesn't have add
        .sqrt()?
        .to_dtype(DType::F32)?;

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
    let sorted_indices = xs.arg_sort_last_dim(false)?.to_dtype(DType::U32)?;
    let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
    let gathered = xs.gather(&topk_indices, D::Minus1)?;
    Ok((gathered, topk_indices))
}

/// Input coords. Output 1 <batch  x 1 > Tensor
/// representing whether each residue has all 4 backbone atoms.
/// note that the internal ordering is different between
/// backbone only [N/CA/C/O] and all-atom [N/CA/C/CB/O]....
pub fn create_backbone_mask_37(xyz_37: &Tensor) -> Result<Tensor> {
    let (b, l, rescount, _) = xyz_37.dims4()?;
    // Create a vector with 1s at positions 0,1,2,4 and 0s elsewhere
    let mut values = vec![0f32; rescount];
    for &idx in &[0, 1, 2, 4] {
        if idx < rescount {
            values[idx] = 1.0;
        }
    }

    // Create the base mask for one sequence, explicitly specifying the data type
    let base_mask = Tensor::new(values.as_slice(), xyz_37.device())?.to_dtype(DType::F32)?;

    // Create the full mask by repeating it for each batch and length
    let mask = base_mask.unsqueeze(0)?.unsqueeze(0)?; // Add batch and length dimensions
    let mask = mask.broadcast_as((b, l, rescount))?;

    Ok(mask)
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

    // Compute cross produAAct components
    let c0 = ((&a1 * &b2)? - (&a2 * &b1)?)?;
    let c1 = ((&a2 * &b0)? - (&a0 * &b2)?)?;
    let c2 = ((&a0 * &b1)? - (&a1 * &b0)?)?;

    // Stack the results
    Tensor::cat(&[&c0, &c1, &c2], last_dim)
}

/// Gather_edges
/// Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
pub fn gather_edges(edges: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    let (d1, d2, d3) = neighbor_idx.dims3()?;
    let neighbors =
        neighbor_idx
            .unsqueeze(D::Minus1)?
            .expand((d1, d2, d3, edges.dim(D::Minus1)?))?;
    let edge_gather = edges.gather(&neighbors, 2)?;
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
    // Reshape back to [B, N, K, C]
    neighbor_features.reshape((batch_size, n_nodes, k_neighbors, n_features))
}

pub fn gather_nodes_t(nodes: &Tensor, neighbor_idx: &Tensor) -> Result<Tensor> {
    // Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    let (d1, d2, d3) = nodes.dims3()?;
    let idx_flat = neighbor_idx.unsqueeze(D::Minus1)?.expand((d1, d2, d3))?;
    nodes.gather(&idx_flat, 1)
}

#[allow(dead_code)]
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

#[allow(dead_code)]
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

pub fn linspace(
    start: f64,
    stop: f64,
    steps: usize,
    device: &Device,
    return_type: DType,
) -> Result<Tensor> {
    if steps == 0 {
        Tensor::from_vec(Vec::<f64>::new(), steps, device)
    } else if steps == 1 {
        Tensor::from_vec(vec![start], steps, device)
    } else {
        let delta = (stop - start) / (steps - 1) as f64;
        let vs = (0..steps)
            .map(|step| start + step as f64 * delta)
            .collect::<Vec<_>>();
        Tensor::from_vec(vs, steps, device)?.to_dtype(return_type)
    }
}

pub fn linspace_f32(start: f32, stop: f32, steps: usize, device: &Device) -> Result<Tensor> {
    match steps {
        0 => Tensor::from_vec(Vec::<f32>::new(), steps, device),
        1 => Tensor::from_vec(vec![start], steps, device),
        _ => {
            let delta = (stop - start) / (steps - 1) as f32;
            let vs = (0..steps)
                .map(|step| start + step as f32 * delta)
                .collect::<Vec<_>>();
            Tensor::from_vec(vs, steps, device)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StructureFeatures;
    use anyhow::Result;
    use ferritin_core::info::elements::Element;
    use ferritin_core::load_structure;
    use ferritin_test_data::TestFile;

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
    fn test_atom_backbone_tensor() -> Result<()> {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(pdb_file)?;
        let ac_backbone_tensor: Tensor = ac.to_numeric_backbone_atoms(&device).expect("REASON");
        // batch size of 1;154 residues; N/CA/C/O; positions
        assert_eq!(ac_backbone_tensor.dims(), &[1, 154, 4, 3]);

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
            // assert_eq!(ac_backbone_tensor.dims(), &[1, 154, 4, 3])
            let actual: Vec<f32> = ac_backbone_tensor.i((b, i, j, k))?.to_vec1()?;
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }
        Ok(())
    }

    #[test]
    fn test_all_atom37_tensor() -> Result<()> {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(pdb_file)?;
        let ac_backbone_tensor: Tensor = ac.to_numeric_atom37(&device).expect("REASON");
        assert_eq!(ac_backbone_tensor.dims(), &[1, 154, 37, 3]);

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
            ("N", (0, 0, 0, ..), vec![24.277, 8.374, -9.854]),
            ("CA", (0, 0, 1, ..), vec![24.404, 9.859, -9.939]),
            ("C", (0, 0, 2, ..), vec![25.814, 10.249, -10.359]),
            ("CB", (0, 0, 3, ..), vec![24.070, 10.495, -8.596]),
            ("O", (0, 0, 4, ..), vec![26.748, 9.469, -10.197]),
            ("CG", (0, 0, 5, ..), vec![24.880, 9.939, -7.442]),
            ("CG1", (0, 0, 6, ..), vec![0.0, 0.0, 0.0]),
            ("CG2", (0, 0, 7, ..), vec![0.0, 0.0, 0.0]),
            ("OG", (0, 0, 8, ..), vec![0.0, 0.0, 0.0]),
            ("OG1", (0, 0, 9, ..), vec![0.0, 0.0, 0.0]),
            ("SG", (0, 0, 10, ..), vec![0.0, 0.0, 0.0]),
            ("CD", (0, 0, 11, ..), vec![0.0, 0.0, 0.0]),
            ("CD1", (0, 0, 12, ..), vec![0.0, 0.0, 0.0]),
            ("CD2", (0, 0, 13, ..), vec![0.0, 0.0, 0.0]),
            ("ND1", (0, 0, 14, ..), vec![0.0, 0.0, 0.0]),
            ("ND2", (0, 0, 15, ..), vec![0.0, 0.0, 0.0]),
            ("OD1", (0, 0, 16, ..), vec![0.0, 0.0, 0.0]),
            ("OD2", (0, 0, 17, ..), vec![0.0, 0.0, 0.0]),
            ("SD", (0, 0, 18, ..), vec![24.262, 10.555, -5.873]),
            ("CE", (0, 0, 19, ..), vec![24.822, 12.266, -5.967]),
            ("CE1", (0, 0, 20, ..), vec![0.0, 0.0, 0.0]),
            ("CE2", (0, 0, 21, ..), vec![0.0, 0.0, 0.0]),
            ("CE3", (0, 0, 22, ..), vec![0.0, 0.0, 0.0]),
            ("NE", (0, 0, 23, ..), vec![0.0, 0.0, 0.0]),
            ("NE1", (0, 0, 24, ..), vec![0.0, 0.0, 0.0]),
            ("NE2", (0, 0, 25, ..), vec![0.0, 0.0, 0.0]),
            ("OE1", (0, 0, 26, ..), vec![0.0, 0.0, 0.0]),
            ("OE2", (0, 0, 27, ..), vec![0.0, 0.0, 0.0]),
            ("CH2", (0, 0, 28, ..), vec![0.0, 0.0, 0.0]),
            ("NH1", (0, 0, 29, ..), vec![0.0, 0.0, 0.0]),
            ("NH2", (0, 0, 30, ..), vec![0.0, 0.0, 0.0]),
            ("OH", (0, 0, 31, ..), vec![0.0, 0.0, 0.0]),
            ("CZ", (0, 0, 32, ..), vec![0.0, 0.0, 0.0]),
            ("CZ2", (0, 0, 33, ..), vec![0.0, 0.0, 0.0]),
            ("CZ3", (0, 0, 34, ..), vec![0.0, 0.0, 0.0]),
            ("NZ", (0, 0, 35, ..), vec![0.0, 0.0, 0.0]),
            ("OXT", (0, 0, 36, ..), vec![0.0, 0.0, 0.0]),
        ];
        for (atom_name, (b, i, j, k), expected) in allatom_coords {
            let actual: Vec<f32> = ac_backbone_tensor.i((b, i, j, k))?.to_vec1()?;
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }
        Ok(())
    }

    #[test]
    fn test_ligand_tensor() -> Result<()> {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(pdb_file)?;
        let (ligand_coords, ligand_elements, _) =
            ac.to_numeric_ligand_atoms(&device).expect("REASON");
        // 154 residues; 54 other atoms.
        assert_eq!(ligand_coords.dims(), &[1, 154, 54, 3]);
        // Check my residue coords in the Tensor
        //
        // HETATM 1222 S  S   . SO4 B 2 .   ? 30.746 18.706  28.896  1.00 47.98  ? 157 SO4 A S   1
        // HETATM 1223 O  O1  . SO4 B 2 .   ? 30.697 20.077  28.620  1.00 48.06  ? 157 SO4 A O1  1
        // HETATM 1224 O  O2  . SO4 B 2 .   ? 31.104 18.021  27.725  1.00 47.52  ? 157 SO4 A O2  1
        // HETATM 1225 O  O3  . SO4 B 2 .   ? 29.468 18.179  29.331  1.00 47.79  ? 157 SO4 A O3  1
        // HETATM 1226 O  O4  . SO4 B 2 .   ? 31.722 18.578  29.881  1.00 47.85  ? 157 SO4 A O4  1
        let allatom_coords = [
            ("S", (0, 0, 0, ..), vec![30.746, 18.706, 28.896]),
            ("O1", (0, 0, 1, ..), vec![30.697, 20.077, 28.620]),
            ("O2", (0, 0, 2, ..), vec![31.104, 18.021, 27.725]),
            ("O3", (0, 0, 3, ..), vec![29.468, 18.179, 29.331]),
            ("O4", (0, 0, 4, ..), vec![31.722, 18.578, 29.881]),
        ];
        for (atom_name, (b, l, i, _j), expected) in allatom_coords {
            let actual: Vec<f32> = ligand_coords.i((b, l, i, ..))?.to_vec1()?;
            assert_eq!(actual, expected, "Mismatch for atom {}", atom_name);
        }

        // Now check the elements
        let elements: Vec<&str> = ligand_elements
            .i((0, 0, ..))?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|elem| Element::new(elem as usize).unwrap().symbol())
            .collect();

        assert_eq!(elements[0], "S");
        assert_eq!(elements[1], "O");
        assert_eq!(elements[2], "O");
        assert_eq!(elements[3], "O");

        Ok(())
    }

    #[test]
    fn test_backbone_tensor() {
        let device = Device::Cpu;
        let (pdb_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let ac = load_structure(pdb_file).unwrap();
        let xyz_37 = ac
            .to_numeric_atom37(&device)
            .expect("XYZ creation for all-atoms");
        assert_eq!(xyz_37.dims(), [1, 154, 37, 3]);

        // # xyz_37_m = feature_dict["xyz_37_m"] #[B,L,37] - mask for all coords
        let xyz_m = create_backbone_mask_37(&xyz_37).expect("masking procedure should work");
        assert_eq!(xyz_m.dims(), &[1, 154, 37]);
    }
    #[test]
    fn test_compute_nearest_neighbors() {
        let device = Device::Cpu;
        let test_dtype = DType::F32;

        // Create a simple 2x3x3 tensor representing 2 sequences of 3 points in 3D space
        let coords = Tensor::new(
            &[
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], // First sequence
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], // Second sequence
            ],
            &device,
        )
        .unwrap()
        .to_dtype(test_dtype)
        .unwrap();

        // Create mask indicating all points are valid
        let mask = Tensor::ones((2, 3), test_dtype, &device).unwrap();

        // Get 2 nearest neighbors for each point
        let (distances, indices) = compute_nearest_neighbors(&coords, &mask, 2, 1e-6).unwrap();

        // Check shapes
        assert_eq!(distances.dims(), &[2, 3, 2]); // [batch, seq_len, k]
        assert_eq!(indices.dims(), &[2, 3, 2]); // [batch, seq_len, k]

        // For first sequence, point [1,0,0] should have [0,0,0] and [2,0,0] as nearest neighbors
        let point_neighbors: Vec<u32> = indices.i((0, 1, ..)).unwrap().to_vec1().unwrap();
        assert_eq!(point_neighbors, vec![0, 2]);

        // Check distances are correct
        let point_distances: Vec<f32> = distances.i((0, 1, ..)).unwrap().to_vec1().unwrap();
        assert!((point_distances[0] - 1.0).abs() < 1e-5);
        assert!((point_distances[1] - 1.0).abs() < 1e-5);
    }
}
