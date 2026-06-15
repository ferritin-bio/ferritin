//! Residue chemistry constants for structure prediction.
//!
//! Centralises the biochemical lookup tables needed by the ESMFold2 forward
//! pass and ESMC structure utilities:
//!
//! - [`ATOM37_NAMES`] / [`atom_order`] — canonical 37-slot heavy-atom ordering
//! - [`chi_angles_atoms`] — 4-atom tuples defining each sidechain dihedral
//! - [`atom14_to_atom37_for_residue`] — per-residue atom14 → atom37 slot mapping
//! - [`vdw_radius`] — van der Waals radii for common protein elements
//!
//! The atom37 ordering matches the [`crate::featurize::utilities::AAAtom`] enum.

use std::collections::HashMap;

// ── Atom37 ordering ────────────────────────────────────────────────────────

pub const NUM_ATOM37: usize = 37;
pub const NUM_RESIDUES: usize = 21; // 20 standard AAs + UNK

/// Canonical atom37 names in slot order.
///
/// The index of each name is its atom37 slot number, matching the
/// `AAAtom` enum in `featurize::utilities`.
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

/// Map from atom name (e.g. `"CA"`) to its atom37 slot index.
pub fn atom_order() -> HashMap<String, usize> {
    ATOM37_NAMES
        .iter()
        .enumerate()
        .map(|(i, &name)| (name.to_string(), i))
        .collect()
}

/// Returns the atom37 slot index for a named atom, or `None` if unknown.
pub fn atom37_index(name: &str) -> Option<usize> {
    ATOM37_NAMES.iter().position(|&n| n == name)
}

// ── Chi angle definitions ──────────────────────────────────────────────────

/// Sidechain chi-angle atom tuples for a residue given by 3-letter code.
///
/// Returns a slice of `(a1, a2, a3, a4)` tuples, one per chi angle (chi1
/// first). An empty slice means the residue has no rotatable sidechains.
///
/// Source: OpenFold / AlphaFold2 `residue_constants.py`.
#[rustfmt::skip]
pub fn chi_angles_atoms(res3: &str) -> &'static [(&'static str, &'static str, &'static str, &'static str)] {
    match res3 {
        "ALA" | "GLY" | "UNK" => &[],
        "ARG" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
            ("CB","CG","CD","NE"), ("CG","CD","NE","CZ"),
        ],
        "ASN" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","OD1"),
        ],
        "ASP" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","OD1"),
        ],
        "CYS" => &[
            ("N","CA","CB","SG"),
        ],
        "GLN" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1"),
        ],
        "GLU" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD"), ("CB","CG","CD","OE1"),
        ],
        "HIS" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","ND1"),
        ],
        "ILE" => &[
            ("N","CA","CB","CG1"), ("CA","CB","CG1","CD1"),
        ],
        "LEU" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD1"),
        ],
        "LYS" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
            ("CB","CG","CD","CE"), ("CG","CD","CE","NZ"),
        ],
        "MET" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","SD"), ("CB","CG","SD","CE"),
        ],
        "PHE" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD1"),
        ],
        "PRO" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD"),
        ],
        "SER" => &[
            ("N","CA","CB","OG"),
        ],
        "THR" => &[
            ("N","CA","CB","OG1"),
        ],
        "TRP" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD1"),
        ],
        "TYR" => &[
            ("N","CA","CB","CG"), ("CA","CB","CG","CD1"),
        ],
        "VAL" => &[
            ("N","CA","CB","CG1"),
        ],
        _ => &[],
    }
}

// ── atom14 → atom37 mapping ────────────────────────────────────────────────

/// For a residue given by its 3-letter code, return the atom37 slot index for
/// each of the 14 atom14 positions, or `None` if that slot is unoccupied.
///
/// The atom14 slot ordering matches the `Residue::atoms14()` method in
/// `featurize::utilities`: [N, CA, C, O, CB, sidechain…].
#[rustfmt::skip]
pub fn atom14_to_atom37_for_residue(res3: &str) -> [Option<usize>; 14] {
    let names: [Option<&str>; 14] = match res3 {
        "ALA" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),None,None,None,None,None,None,None,None,None],
        "CYS" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("SG"),None,None,None,None,None,None,None,None],
        "ASP" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("OD1"),Some("OD2"),None,None,None,None,None,None],
        "GLU" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD"),Some("OE1"),Some("OE2"),None,None,None,None,None],
        "PHE" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD1"),Some("CD2"),Some("CE1"),Some("CE2"),Some("CZ"),None,None,None],
        "GLY" => [Some("N"),Some("CA"),Some("C"),Some("O"),None,None,None,None,None,None,None,None,None,None],
        "HIS" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("ND1"),Some("CD2"),Some("CE1"),Some("NE2"),None,None,None,None],
        "ILE" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG1"),Some("CG2"),Some("CD1"),None,None,None,None,None,None],
        "LYS" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD"),Some("CE"),Some("NZ"),None,None,None,None,None],
        "LEU" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD1"),Some("CD2"),None,None,None,None,None,None],
        "MET" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("SD"),Some("CE"),None,None,None,None,None,None],
        "ASN" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("OD1"),Some("ND2"),None,None,None,None,None,None],
        "PRO" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD"),None,None,None,None,None,None,None],
        "GLN" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD"),Some("OE1"),Some("NE2"),None,None,None,None,None],
        "ARG" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD"),Some("NE"),Some("CZ"),Some("NH1"),Some("NH2"),None,None,None],
        "SER" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("OG"),None,None,None,None,None,None,None,None],
        "THR" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("OG1"),Some("CG2"),None,None,None,None,None,None,None],
        "VAL" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG1"),Some("CG2"),None,None,None,None,None,None,None],
        "TRP" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD1"),Some("CD2"),Some("CE2"),Some("CE3"),Some("NE1"),Some("CZ2"),Some("CZ3"),Some("CH2")],
        "TYR" => [Some("N"),Some("CA"),Some("C"),Some("O"),Some("CB"),Some("CG"),Some("CD1"),Some("CD2"),Some("CE1"),Some("CE2"),Some("CZ"),Some("OH"),None,None],
        _     => [None; 14],
    };

    let mut out = [None; 14];
    for (i, name_opt) in names.iter().enumerate() {
        out[i] = name_opt.and_then(atom37_index);
    }
    out
}

// ── Van der Waals radii ────────────────────────────────────────────────────

/// Van der Waals radius (Å) for elements commonly found in proteins.
///
/// Source: Bondi (1964) / standard crystallographic values.
pub fn vdw_radius(element: &str) -> f32 {
    match element {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "F" => 1.47,
        "CL" | "Cl" => 1.75,
        "BR" | "Br" => 1.85,
        "I" => 1.98,
        "SE" | "Se" => 1.90,
        _ => 1.70,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_order_completeness() {
        let order = atom_order();
        assert_eq!(order.len(), NUM_ATOM37);
        // Backbone atoms must be present at well-known indices
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
    fn test_chi_gly_ala_empty() {
        assert!(chi_angles_atoms("GLY").is_empty());
        assert!(chi_angles_atoms("ALA").is_empty());
    }

    #[test]
    fn test_chi_arg_four_angles() {
        assert_eq!(chi_angles_atoms("ARG").len(), 4);
        assert_eq!(chi_angles_atoms("ARG")[0], ("N", "CA", "CB", "CG"));
    }

    #[test]
    fn test_chi_val_one_angle() {
        let chis = chi_angles_atoms("VAL");
        assert_eq!(chis.len(), 1);
        assert_eq!(chis[0], ("N", "CA", "CB", "CG1"));
    }

    #[test]
    fn test_atom14_to_atom37_gly() {
        let mapping = atom14_to_atom37_for_residue("GLY");
        // GLY: N=0, CA=1, C=2, O=4, rest None
        assert_eq!(mapping[0], Some(0)); // N
        assert_eq!(mapping[1], Some(1)); // CA
        assert_eq!(mapping[2], Some(2)); // C
        assert_eq!(mapping[3], Some(4)); // O
        assert_eq!(mapping[4], None); // no CB for GLY
        for i in 4..14 {
            assert_eq!(mapping[i], None, "GLY slot {i} should be None");
        }
    }

    #[test]
    fn test_atom14_to_atom37_ala() {
        let mapping = atom14_to_atom37_for_residue("ALA");
        assert_eq!(mapping[4], Some(3)); // CB → atom37 slot 3
        assert_eq!(mapping[5], None);
    }

    #[test]
    fn test_atom14_to_atom37_trp_full() {
        // TRP fills all 14 slots
        let mapping = atom14_to_atom37_for_residue("TRP");
        assert!(
            mapping.iter().all(|m| m.is_some()),
            "TRP should fill all 14 slots"
        );
    }

    #[test]
    fn test_atom14_to_atom37_unk() {
        let mapping = atom14_to_atom37_for_residue("UNK");
        assert!(mapping.iter().all(|m| m.is_none()));
    }

    #[test]
    fn test_vdw_radius_known_elements() {
        assert!((vdw_radius("C") - 1.70).abs() < 1e-6);
        assert!((vdw_radius("N") - 1.55).abs() < 1e-6);
        assert!((vdw_radius("O") - 1.52).abs() < 1e-6);
        assert!((vdw_radius("S") - 1.80).abs() < 1e-6);
    }

    #[test]
    fn test_vdw_radius_unknown_defaults_to_carbon() {
        assert!((vdw_radius("X") - 1.70).abs() < 1e-6);
    }
}
