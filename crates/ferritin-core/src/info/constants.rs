//! # Constants
//!
//! This module contains core data structures and functions for analyzing molecular structures.
//!
//! ## Residue Types
//! The module provides functions to check residue types:
//!
//! - `is_amino_acid()` - Check if a residue is an amino acid
//! - `is_carbohydrate()` - Check if a residue is a carbohydrate
//! - `is_nucleotide()` - Check if a residue is a nucleotide
//!
//! ## Bond Information
//! Bond data from CCD includes:
//!
//! - Bond lengths with standard deviations
//! - Canonical amino acid connectivity
//! - Bond order information
//!

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[rustfmt::skip]
pub(crate) fn default_distance_range(a: &str, b: &str) -> (f32, f32) {
    match (a, b) {
        // https://github.com/biotite-dev/biotite/blob/main/src/biotite/structure/bonds.pyx#L1341C1-L1389C1
        //               # Taken from Allen et al.
        //                 min   - 2*std     max   + 2*std
        ("B",  "C" ) => (1.556 - 2.0*0.015,  1.556 + 2.0*0.015),
        ("BR", "C" ) => (1.875 - 2.0*0.029,  1.966 + 2.0*0.029),
        ("BR", "O" ) => (1.581 - 2.0*0.007,  1.581 + 2.0*0.007),
        ("C",  "C" ) => (1.174 - 2.0*0.011,  1.588 + 2.0*0.025),
        ("C",  "CL") => (1.713 - 2.0*0.011,  1.849 + 2.0*0.011),
        ("C",  "F" ) => (1.320 - 2.0*0.009,  1.428 + 2.0*0.009),
        ("C",  "H" ) => (1.059 - 2.0*0.030,  1.099 + 2.0*0.007),
        ("C",  "I" ) => (2.095 - 2.0*0.015,  2.162 + 2.0*0.015),
        ("C",  "N" ) => (1.325 - 2.0*0.009,  1.552 + 2.0*0.023),
        ("C",  "O" ) => (1.187 - 2.0*0.011,  1.477 + 2.0*0.008),
        ("C",  "P" ) => (1.791 - 2.0*0.006,  1.855 + 2.0*0.019),
        ("C",  "S" ) => (1.630 - 2.0*0.014,  1.863 + 2.0*0.015),
        ("C",  "SE") => (1.893 - 2.0*0.013,  1.970 + 2.0*0.032),
        ("C",  "SI") => (1.837 - 2.0*0.012,  1.888 + 2.0*0.023),
        ("CL", "O" ) => (1.414 - 2.0*0.026,  1.414 + 2.0*0.026),
        ("CL", "P" ) => (1.997 - 2.0*0.035,  2.008 + 2.0*0.035),
        ("CL", "S" ) => (2.072 - 2.0*0.023,  2.072 + 2.0*0.023),
        ("CL", "SI") => (2.072 - 2.0*0.009,  2.072 + 2.0*0.009),
        ("F",  "N" ) => (1.406 - 2.0*0.016,  1.406 + 2.0*0.016),
        ("F",  "P" ) => (1.495 - 2.0*0.016,  1.579 + 2.0*0.025),
        ("F",  "S" ) => (1.640 - 2.0*0.011,  1.640 + 2.0*0.011),
        ("F",  "SI") => (1.588 - 2.0*0.014,  1.694 + 2.0*0.013),
        ("H",  "N" ) => (1.009 - 2.0*0.022,  1.033 + 2.0*0.022),
        ("H",  "O" ) => (0.967 - 2.0*0.010,  1.015 + 2.0*0.017),
        ("I",  "O" ) => (2.144 - 2.0*0.028,  2.144 + 2.0*0.028),
        ("N",  "N" ) => (1.124 - 2.0*0.015,  1.454 + 2.0*0.021),
        ("N",  "O" ) => (1.210 - 2.0*0.011,  1.463 + 2.0*0.012),
        ("N",  "P" ) => (1.571 - 2.0*0.013,  1.697 + 2.0*0.015),
        ("N",  "S" ) => (1.541 - 2.0*0.022,  1.710 + 2.0*0.019),
        ("N",  "SI") => (1.711 - 2.0*0.019,  1.748 + 2.0*0.022),
        ("O",  "P" ) => (1.449 - 2.0*0.007,  1.689 + 2.0*0.024),
        ("O",  "S" ) => (1.423 - 2.0*0.008,  1.580 + 2.0*0.015),
        ("O",  "SI") => (1.622 - 2.0*0.014,  1.680 + 2.0*0.008),
        ("P",  "P" ) => (2.214 - 2.0*0.022,  2.214 + 2.0*0.022),
        ("P",  "S" ) => (1.913 - 2.0*0.014,  1.954 + 2.0*0.005),
        ("P",  "SE") => (2.093 - 2.0*0.019,  2.093 + 2.0*0.019),
        ("P",  "SI") => (2.264 - 2.0*0.019,  2.264 + 2.0*0.019),
        ("S",  "S" ) => (1.897 - 2.0*0.012,  2.070 + 2.0*0.022),
        ("S",  "SE") => (2.193 - 2.0*0.015,  2.193 + 2.0*0.015),
        ("S",  "SI") => (2.145 - 2.0*0.020,  2.145 + 2.0*0.020),
        ("SE", "SE") => (2.340 - 2.0*0.024,  2.340 + 2.0*0.024),
        ("SI", "SE") => (2.359 - 2.0*0.012,  2.359 + 2.0*0.012),
        _ => panic!("Unknown atom pair: {} and {}", a, b),
    }
}

static AA_BONDS: OnceLock<HashMap<&'static str, Vec<(&'static str, &'static str, i32)>>> =
    OnceLock::new();

#[rustfmt::skip]
/// get_bonds_canonical20
///
/// This is the bond information for the 10 canonical
/// AAs.  Data were obtained from the [CCD](https://www.wwpdb.org/data/ccd).
///
pub(crate) fn get_bonds_canonical20() -> &'static HashMap<&'static str, Vec<(&'static str, &'static str, i32)>> {
    AA_BONDS.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("ALA", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","HB1",1), ("CB","HB2",1), ("CB","HB3",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HXT","OXT",1)
        ]);
        m.insert("ARG", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD","HD2",1), ("CD","HD3",1),
            ("CD","NE",1), ("CD","CG",1), ("CG","HG2",1), ("CG","HG3",1), ("CZ","NH1",1),
            ("CZ","NH2",2), ("CA","N",1), ("H","N",1), ("H2","N",1), ("CZ","NE",1),
            ("HE","NE",1), ("HH11","NH1",1), ("HH12","NH1",1), ("HH21","NH2",1),
            ("HH22","NH2",1), ("HXT","OXT",1)
        ]);
        m.insert("ASN", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CG","ND2",1), ("CG","OD1",2),
            ("CA","N",1), ("H","N",1), ("H2","N",1), ("HD21","ND2",1), ("HD22","ND2",1),
            ("HXT","OXT",1)
        ]);
        m.insert("ASP", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CG","OD1",2), ("CG","OD2",1),
            ("CA","N",1), ("H","N",1), ("H2","N",1), ("HD2","OD2",1), ("HXT","OXT",1)
        ]);
        m.insert("CYS", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","HB2",1), ("CB","HB3",1), ("CB","SG",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HXT","OXT",1), ("HG","SG",1)
        ]);
        m.insert("GLN", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD","NE2",1), ("CD","OE1",2),
            ("CD","CG",1), ("CG","HG2",1), ("CG","HG3",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HE21","NE2",1), ("HE22","NE2",1), ("HXT","OXT",1)
        ]);
        m.insert("GLU", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD","OE1",2), ("CD","OE2",1),
            ("CD","CG",1), ("CG","HG2",1), ("CG","HG3",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HE2","OE2",1), ("HXT","OXT",1)
        ]);
        m.insert("GLY", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","HA2",1), ("CA","HA3",1),
            ("CA","N",1), ("H","N",1), ("H2","N",1), ("HXT","OXT",1)
        ]);
        m.insert("HIS", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD2","HD2",1), ("CD2","NE2",5),
            ("CE1","HE1",1), ("CE1","NE2",5), ("CD2","CG",6), ("CG","ND1",5), ("CA","N",1),
            ("H","N",1), ("H2","N",1), ("CE1","ND1",6), ("HD1","ND1",1), ("HE2","NE2",1),
            ("HXT","OXT",1)
        ]);
        m.insert("ILE", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG1",1), ("CB","CG2",1), ("CB","HB",1), ("CD1","HD11",1), ("CD1","HD12",1),
            ("CD1","HD13",1), ("CD1","CG1",1), ("CG1","HG12",1), ("CG1","HG13",1),
            ("CG2","HG21",1), ("CG2","HG22",1), ("CG2","HG23",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HXT","OXT",1)
        ]);
        m.insert("LEU", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD1","HD11",1), ("CD1","HD12",1),
            ("CD1","HD13",1), ("CD2","HD21",1), ("CD2","HD22",1), ("CD2","HD23",1),
            ("CD1","CG",1), ("CD2","CG",1), ("CG","HG",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HXT","OXT",1)
        ]);
        m.insert("LYS", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD","CE",1), ("CD","HD2",1),
            ("CD","HD3",1), ("CE","HE2",1), ("CE","HE3",1), ("CE","NZ",1), ("CD","CG",1),
            ("CG","HG2",1), ("CG","HG3",1), ("CA","N",1), ("H","N",1), ("H2","N",1),
            ("HZ1","NZ",1), ("HZ2","NZ",1), ("HZ3","NZ",1), ("HXT","OXT",1)
        ]);
        m.insert("MET", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CE","HE1",1), ("CE","HE2",1),
            ("CE","HE3",1), ("CG","HG2",1), ("CG","HG3",1), ("CG","SD",1), ("CA","N",1),
            ("H","N",1), ("H2","N",1), ("HXT","OXT",1), ("CE","SD",1)
        ]);
        m.insert("PHE", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD1","CE1",5), ("CD1","HD1",1),
            ("CD2","CE2",6), ("CD2","HD2",1), ("CE1","CZ",6), ("CE1","HE1",1), ("CE2","CZ",5),
            ("CE2","HE2",1), ("CD1","CG",6), ("CD2","CG",5), ("CZ","HZ",1), ("CA","N",1),
            ("H","N",1), ("H2","N",1), ("HXT","OXT",1)
        ]);
        m.insert("PRO", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD","HD2",1), ("CD","HD3",1),
            ("CD","CG",1), ("CG","HG2",1), ("CG","HG3",1), ("CA","N",1), ("CD","N",1),
            ("H","N",1), ("HXT","OXT",1)
        ]);
        m.insert("SER", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","HB2",1), ("CB","HB3",1), ("CB","OG",1), ("CA","N",1), ("H","N",1),
            ("H2","N",1), ("HG","OG",1), ("HXT","OXT",1)
        ]);
        m.insert("THR", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG2",1), ("CB","HB",1), ("CB","OG1",1), ("CG2","HG21",1), ("CG2","HG22",1),
            ("CG2","HG23",1), ("CA","N",1), ("H","N",1), ("H2","N",1), ("HG1","OG1",1),
            ("HXT","OXT",1)
        ]);
        m.insert("TRP", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD1","HD1",1), ("CD1","NE1",5),
            ("CD2","CE2",6), ("CD2","CE3",5), ("CE2","CZ2",5), ("CE3","CZ3",6), ("CE3","HE3",1),
            ("CD1","CG",6), ("CD2","CG",5), ("CH2","HH2",1), ("CH2","CZ2",6), ("CZ2","HZ2",1),
            ("CH2","CZ3",5), ("CZ3","HZ3",1), ("CA","N",1), ("H","N",1), ("H2","N",1),
            ("CE2","NE1",5), ("HE1","NE1",1), ("HXT","OXT",1)
        ]);
        m.insert("TYR", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG",1), ("CB","HB2",1), ("CB","HB3",1), ("CD1","CE1",5), ("CD1","HD1",1),
            ("CD2","CE2",6), ("CD2","HD2",1), ("CE1","CZ",6), ("CE1","HE1",1), ("CE2","CZ",5),
            ("CE2","HE2",1), ("CD1","CG",6), ("CD2","CG",5), ("CZ","OH",1), ("CA","N",1),
            ("H","N",1), ("H2","N",1), ("HH","OH",1), ("HXT","OXT",1)
        ]);
        m.insert("VAL", vec![
            ("C","O",2), ("C","OXT",1), ("C","CA",1), ("CA","CB",1), ("CA","HA",1),
            ("CB","CG1",1), ("CB","CG2",1), ("CB","HB",1), ("CG1","HG11",1), ("CG1","HG12",1),
            ("CG1","HG13",1), ("CG2","HG21",1), ("CG2","HG22",1), ("CG2","HG23",1),
            ("CA","N",1), ("H","N",1), ("H2","N",1), ("HXT","OXT",1)
        ]);
        m
    })
}

static AMINO_ACIDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
static CARBOHYDRATES: OnceLock<HashSet<&'static str>> = OnceLock::new();
static NUCLEOTIDES: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn get_amino_acids() -> &'static HashSet<&'static str> {
    AMINO_ACIDS.get_or_init(|| include_str!("ccddata/amino_acids.txt").lines().collect())
}

fn get_carbohydrates() -> &'static HashSet<&'static str> {
    CARBOHYDRATES.get_or_init(|| include_str!("ccddata/carbohydrates.txt").lines().collect())
}

fn get_nucleotides() -> &'static HashSet<&'static str> {
    NUCLEOTIDES.get_or_init(|| include_str!("ccddata/nucleotides.txt").lines().collect())
}

pub(crate) fn is_amino_acid(symbol: &str) -> bool {
    get_amino_acids().contains(symbol)
}

pub(crate) fn is_carbohydrate(symbol: &str) -> bool {
    get_carbohydrates().contains(symbol)
}

pub(crate) fn is_nucleotide(symbol: &str) -> bool {
    get_nucleotides().contains(symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_checking() {
        assert!(is_amino_acid("ALA"));
        assert!(is_amino_acid("ARG"));
        assert!(!is_amino_acid("ZZZ"));

        assert!(is_carbohydrate("045"));
        assert!(is_carbohydrate("05L"));
        assert!(!is_carbohydrate("ZZZ"));

        assert!(is_nucleotide("02I"));
        assert!(is_nucleotide("05A"));
        assert!(!is_nucleotide("ZZZ"));
    }
}
