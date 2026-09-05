//! Canonical 37-slot heavy-atom representation and atom-name lookups.
//!
//! The atom37 scheme is the standard representation used by ESMFold, AlphaFold2,
//! and related models: every amino acid is represented with up to 37 possible
//! heavy-atom positions, each at a fixed slot index defined by [`ATOM37_NAMES`].
//!
//! # Key items
//! - [`NUM_ATOM37`] / [`NUM_RESIDUES`] — dimension constants
//! - [`ATOM37_NAMES`] — canonical name→slot mapping (array)
//! - [`atom_order`] — `HashMap` version of the same mapping
//! - [`atom37_index`] — slot lookup by atom name
//! - [`AAAtom`] — typed enum mirroring the slot indices

use std::collections::HashMap;
use std::sync::OnceLock;
use strum::{Display, EnumIter, EnumString};

// ── Dimension constants ───────────────────────────────────────────────────────

pub const NUM_ATOM37: usize = 37;
pub const NUM_RESIDUES: usize = 21; // 20 standard AAs + UNK

// ── Atom37 slot names ─────────────────────────────────────────────────────────

/// Canonical atom37 names in slot order.
///
/// The index of each name is its atom37 slot number, matching [`AAAtom`].
#[rustfmt::skip]
pub const ATOM37_NAMES: [&str; NUM_ATOM37] = [
    "N",   "CA",  "C",   "CB",  "O",
    "CG",  "CG1", "CG2", "OG",  "OG1",
    "SG",  "CD",  "CD1", "CD2", "ND1",
    "ND2", "OD1", "OD2", "SD",  "CE",
    "CE1", "CE2", "CE3", "NE",  "NE1",
    "NE2", "OE1", "OE2", "CH2", "NH1",
    "NH2", "OH",  "CZ",  "CZ2", "CZ3",
    "NZ",  "OXT",
];

static ATOM_ORDER: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();

/// Map from atom name (e.g. `"CA"`) to its atom37 slot index.
pub fn atom_order() -> &'static HashMap<&'static str, usize> {
    ATOM_ORDER.get_or_init(|| {
        ATOM37_NAMES
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect()
    })
}

/// Returns the atom37 slot index for a named atom, or `None` if unknown.
pub fn atom37_index(name: &str) -> Option<usize> {
    ATOM37_NAMES.iter().position(|&n| n == name)
}

// ── AAAtom enum ───────────────────────────────────────────────────────────────

/// Typed atom37 slots; discriminants match [`ATOM37_NAMES`] indices.
///
/// `Unknown = -1` is used as a sentinel for "atom not present in this residue"
/// in atom14 tables. Callers working with `usize` slot indices should use
/// [`atom37_index`] instead.
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
    pub fn to_index(&self) -> usize {
        *self as usize
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_order_completeness() {
        let order = atom_order();
        assert_eq!(order.len(), NUM_ATOM37);
        assert_eq!(order["N"], 0);
        assert_eq!(order["CA"], 1);
        assert_eq!(order["C"], 2);
        assert_eq!(order["CB"], 3);
        assert_eq!(order["O"], 4);
    }

    #[test]
    fn test_atom37_index_roundtrip() {
        for (i, &name) in ATOM37_NAMES.iter().enumerate() {
            assert_eq!(atom37_index(name), Some(i));
        }
        assert_eq!(atom37_index("ZZZ"), None);
    }

    #[test]
    fn test_aaatom_discriminants_match_names() {
        assert_eq!(AAAtom::N as i32, 0);
        assert_eq!(AAAtom::CA as i32, 1);
        assert_eq!(AAAtom::OXT as i32, 36);
        assert_eq!(AAAtom::Unknown as i32, -1);
    }
}
