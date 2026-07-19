//! Table structs for per-atom, per-residue, and per-chain data.
//!
//! These tables hold topology data that is constant across trajectory frames.
//! All tables use struct-of-arrays (SoA) layout for cache-friendly iteration.

use super::ModelError;
use crate::info::elements::Element;

/// Sentinel used when an mmCIF sequence identifier is explicitly missing (`.` or `?`).
pub const MISSING_SEQ_ID: i32 = i32::MIN;

fn require_len(table: &str, field: &str, actual: usize, expected: usize) -> Result<(), ModelError> {
    if actual != expected {
        return Err(ModelError::new(format!(
            "{table}.{field} has length {actual}, expected {expected}"
        )));
    }
    Ok(())
}

/// Residue classification: polymer (protein/nucleic acid) or non-polymer (ligand/solvent).
#[derive(Clone, Debug, PartialEq)]
pub enum ResidueGroup {
    /// Polymer residue: part of a protein, DNA, or RNA chain.
    Polymer,
    /// Non-polymer residue: ligand, solvent, ion, etc.
    NonPolymer,
}

/// Per-atom topology data (not coordinates — those live in [`AtomicConformation`]).
///
/// Each field is a parallel array indexed by atom index.
#[derive(Clone, Debug)]
pub struct AtomsTable {
    /// Canonical atom name (`label_atom_id` in mmCIF), e.g. "CA", "N", "CB".
    pub atom_name: Vec<String>,
    /// Author-assigned atom name (`auth_atom_id` in mmCIF).
    pub auth_atom_name: Vec<String>,
    /// Element type for each atom.
    pub element: Vec<Element>,
    /// Alternative location indicator (e.g. `Some('A')`), or `None`.
    pub alt_loc: Vec<Option<char>>,
    /// Formal charge, or `None` if unspecified.
    pub formal_charge: Vec<Option<i8>>,
}

impl AtomsTable {
    /// Validate that every atom column has the same length.
    pub fn validate(&self) -> Result<(), ModelError> {
        let n = self.atom_name.len();
        require_len("atoms", "auth_atom_name", self.auth_atom_name.len(), n)?;
        require_len("atoms", "element", self.element.len(), n)?;
        require_len("atoms", "alt_loc", self.alt_loc.len(), n)?;
        require_len("atoms", "formal_charge", self.formal_charge.len(), n)
    }
    /// Number of atoms.
    pub fn len(&self) -> usize {
        self.atom_name.len()
    }

    /// Returns `true` if the table contains no atoms.
    pub fn is_empty(&self) -> bool {
        self.atom_name.is_empty()
    }
}

/// Per-residue topology data.
///
/// Each field is a parallel array indexed by residue index.
#[derive(Clone, Debug)]
pub struct ResiduesTable {
    /// Canonical residue name (`label_comp_id`), e.g. "ALA", "HOH", "ATP".
    pub comp_id: Vec<String>,
    /// Author-assigned residue name (`auth_comp_id`).
    pub auth_comp_id: Vec<String>,
    /// Internal sequential residue index (label_seq_id in mmCIF).
    /// [`MISSING_SEQ_ID`] represents an explicitly absent value.
    pub label_seq_id: Vec<i32>,
    /// Author-assigned sequence number (auth_seq_id in mmCIF).
    /// Used as the primary sort key for canonical iteration order within a chain.
    pub auth_seq_id: Vec<i32>,
    /// Insertion code (e.g. `Some('A')`), or `None`. Secondary sort key for canonical order.
    pub ins_code: Vec<Option<char>>,
    /// Whether this residue belongs to a polymer or non-polymer group.
    pub group: Vec<ResidueGroup>,
}

impl ResiduesTable {
    /// Validate that every residue column has the same length.
    pub fn validate(&self) -> Result<(), ModelError> {
        let n = self.comp_id.len();
        require_len("residues", "auth_comp_id", self.auth_comp_id.len(), n)?;
        require_len("residues", "label_seq_id", self.label_seq_id.len(), n)?;
        require_len("residues", "auth_seq_id", self.auth_seq_id.len(), n)?;
        require_len("residues", "ins_code", self.ins_code.len(), n)?;
        require_len("residues", "group", self.group.len(), n)
    }
    /// Number of residues.
    pub fn len(&self) -> usize {
        self.comp_id.len()
    }

    /// Returns `true` if the table contains no residues.
    pub fn is_empty(&self) -> bool {
        self.comp_id.is_empty()
    }
}

/// Per-chain topology data.
///
/// Each field is a parallel array indexed by chain index.
#[derive(Clone, Debug)]
pub struct ChainsTable {
    /// Internal chain identifier (label_asym_id in mmCIF).
    pub label_asym_id: Vec<String>,
    /// Author-assigned chain identifier (auth_asym_id in mmCIF).
    pub auth_asym_id: Vec<String>,
    /// Entity identifier this chain belongs to.
    pub entity_id: Vec<String>,
}

impl ChainsTable {
    /// Validate that every chain column has the same length.
    pub fn validate(&self) -> Result<(), ModelError> {
        let n = self.label_asym_id.len();
        require_len("chains", "auth_asym_id", self.auth_asym_id.len(), n)?;
        require_len("chains", "entity_id", self.entity_id.len(), n)
    }
    /// Number of chains.
    pub fn len(&self) -> usize {
        self.label_asym_id.len()
    }

    /// Returns `true` if the table contains no chains.
    pub fn is_empty(&self) -> bool {
        self.label_asym_id.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atoms_table_len() {
        let table = AtomsTable {
            atom_name: vec!["N".into(), "CA".into(), "C".into()],
            auth_atom_name: vec!["N".into(), "CA".into(), "C".into()],
            element: vec![Element::N, Element::C, Element::C],
            alt_loc: vec![None, None, None],
            formal_charge: vec![None, None, None],
        };
        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());

        let empty = AtomsTable {
            atom_name: vec![],
            auth_atom_name: vec![],
            element: vec![],
            alt_loc: vec![],
            formal_charge: vec![],
        };
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_residue_group_variants() {
        let polymer = ResidueGroup::Polymer;
        let non_polymer = ResidueGroup::NonPolymer;

        assert_eq!(polymer, ResidueGroup::Polymer);
        assert_eq!(non_polymer, ResidueGroup::NonPolymer);
        assert_ne!(polymer, non_polymer);

        // Verify Clone + Debug work
        let cloned = polymer.clone();
        assert_eq!(cloned, ResidueGroup::Polymer);
        assert!(!format!("{:?}", non_polymer).is_empty());
    }

    #[test]
    fn test_atoms_table_validation_rejects_mismatched_columns() {
        let table = AtomsTable {
            atom_name: vec!["CA".into()],
            auth_atom_name: vec![],
            element: vec![Element::C],
            alt_loc: vec![None],
            formal_charge: vec![None],
        };
        assert_eq!(
            table.validate().unwrap_err().to_string(),
            "atoms.auth_atom_name has length 0, expected 1"
        );
    }
}
