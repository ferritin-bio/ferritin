//! Table structs for per-atom, per-residue, and per-chain data.
//!
//! These tables hold topology data that is constant across trajectory frames.
//! All tables use struct-of-arrays (SoA) layout for cache-friendly iteration.

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
    /// Atom name, e.g. "CA", "N", "CB".
    pub atom_name: Vec<String>,
    /// Element symbol, e.g. "C", "N", "O", "S".
    pub element: Vec<String>,
    /// Alternative location indicator (e.g. `Some('A')`), or `None`.
    pub alt_loc: Vec<Option<char>>,
    /// Formal charge, or `None` if unspecified.
    pub formal_charge: Vec<Option<i8>>,
}

impl AtomsTable {
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
    /// Residue name (CCD component ID), e.g. "ALA", "HOH", "ATP".
    pub comp_id: Vec<String>,
    /// Internal sequential residue index (label_seq_id in mmCIF).
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
            element: vec!["N".into(), "C".into(), "C".into()],
            alt_loc: vec![None, None, None],
            formal_charge: vec![None, None, None],
        };
        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());

        let empty = AtomsTable {
            atom_name: vec![],
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
}
