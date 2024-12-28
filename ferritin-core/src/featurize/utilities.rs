use candle_core::{DType, Result, Tensor, D};
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

/// Retrieve the nearest Neighbor of a set of coordinates.
/// Usually used for CA carbon distance.
pub fn compute_nearest_neighbors(
    coords: &Tensor,
    mask: &Tensor,
    k: usize,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    // Todo: fix the F32/F64 issue
    let (_batch_size, seq_len, _) = coords.dims3()?;

    // broadcast_matmul handles broadcasting automatically
    // [2, 3, 1] × [2, 1, 3] -> [2, 3, 3]
    let mask_2d = mask
        .unsqueeze(2)?
        .broadcast_matmul(&mask.unsqueeze(1)?)?
        .to_dtype(DType::F32)?; // Convert to f64 once, at the start

    // Compute pairwise distances with broadcasting
    let distances = (coords
        .unsqueeze(2)?
        .broadcast_sub(&coords.unsqueeze(1)?)?
        .powf(2.)?
        .sum(D::Minus1)?
        + eps as f64)?
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

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::ligandmpnn::proteinfeatures::LMPNNFeatures;
    use crate::AtomCollection;
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
}
