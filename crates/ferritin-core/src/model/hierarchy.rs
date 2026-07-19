//! Atomic hierarchy: topology layer shared across trajectory frames.
//!
//! [`AtomicHierarchy`] holds all structural data that does not change between
//! trajectory frames (connectivity, sequence, chain/residue assignments).
//! It is wrapped in [`std::sync::Arc`] so multiple [`super::model::Model`]s
//! (frames) can share a single topology without cloning.

use super::bonds::Bonds;
use super::error::ModelError;
use super::tables::{AtomsTable, ChainsTable, ResiduesTable};
use crate::data::Segmentation;
use std::ops::Range;

/// Topology layer: all structural data that is constant across trajectory frames.
///
/// Wrap in `Arc<AtomicHierarchy>` so multiple `Model`s (frames) share one topology
/// without duplication.
///
/// # Hierarchy
///
/// ```text
/// Chain  ──┐
///           ├── Residue ──┐
///                          └── Atom
/// ```
///
/// The `atom_to_residue` and `residue_to_chain` segmentations provide O(1)
/// range queries and O(log n) reverse lookups.
#[derive(Clone, Debug)]
pub struct AtomicHierarchy {
    /// Per-atom topology data.
    pub atoms: AtomsTable,
    /// Per-residue topology data.
    pub residues: ResiduesTable,
    /// Per-chain topology data.
    pub chains: ChainsTable,
    /// Segmentation mapping atom index → residue index.
    ///
    /// `atom_to_residue.segment(res_idx)` gives the range of atom indices
    /// belonging to residue `res_idx`.
    pub atom_to_residue: Segmentation,
    /// Segmentation mapping residue index → chain index.
    ///
    /// `residue_to_chain.segment(chain_idx)` gives the range of residue indices
    /// belonging to chain `chain_idx`.
    pub residue_to_chain: Segmentation,
    /// Bond connectivity.
    pub bonds: Bonds,
}

impl AtomicHierarchy {
    /// Validate table lengths and hierarchy segmentations.
    pub fn validate(&self) -> Result<(), ModelError> {
        self.atoms.validate()?;
        self.residues.validate()?;
        self.chains.validate()?;
        if self.atom_to_residue.count() != self.residues.len()
            || self.atom_to_residue.element_count() != self.atoms.len()
        {
            return Err(ModelError::new(format!(
                "atom_to_residue describes {} residues and {} atoms, expected {} residues and {} atoms",
                self.atom_to_residue.count(),
                self.atom_to_residue.element_count(),
                self.residues.len(),
                self.atoms.len()
            )));
        }
        if self.residue_to_chain.count() != self.chains.len()
            || self.residue_to_chain.element_count() != self.residues.len()
        {
            return Err(ModelError::new(format!(
                "residue_to_chain describes {} chains and {} residues, expected {} chains and {} residues",
                self.residue_to_chain.count(),
                self.residue_to_chain.element_count(),
                self.chains.len(),
                self.residues.len()
            )));
        }
        if self.bonds.atom_bond_starts.len() != self.atoms.len() + 1 {
            return Err(ModelError::new(format!(
                "bond CSR index has length {}, expected {}",
                self.bonds.atom_bond_starts.len(),
                self.atoms.len() + 1
            )));
        }
        if self.bonds.atom_a.len() != self.bonds.atom_b.len()
            || self.bonds.atom_a.len() != self.bonds.order.len()
        {
            return Err(ModelError::new("bond columns have inconsistent lengths"));
        }
        if self
            .bonds
            .atom_a
            .iter()
            .chain(&self.bonds.atom_b)
            .any(|&atom| atom as usize >= self.atoms.len())
        {
            return Err(ModelError::new("bond endpoint is out of atom range"));
        }
        if self.bonds.atom_bond_starts.windows(2).any(|w| w[0] > w[1])
            || self.bonds.atom_bond_starts.last().copied().unwrap_or(0) as usize
                != self.bonds.atom_a.len()
        {
            return Err(ModelError::new(
                "bond CSR index is inconsistent with bond columns",
            ));
        }
        Ok(())
    }
    /// Total number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    /// Total number of residues.
    pub fn n_residues(&self) -> usize {
        self.residues.len()
    }

    /// Total number of chains.
    pub fn n_chains(&self) -> usize {
        self.chains.len()
    }

    /// Returns the range of atom indices belonging to residue `res_idx`.
    ///
    /// # Panics
    /// Panics if `res_idx >= n_residues()`.
    pub fn atoms_in_residue(&self, res_idx: usize) -> Range<usize> {
        self.atom_to_residue.segment(res_idx)
    }

    /// Returns the range of residue indices belonging to chain `chain_idx`.
    ///
    /// # Panics
    /// Panics if `chain_idx >= n_chains()`.
    pub fn residues_in_chain(&self, chain_idx: usize) -> Range<usize> {
        self.residue_to_chain.segment(chain_idx)
    }

    /// Returns the residue index that contains atom `atom_idx`.
    ///
    /// O(log n) binary search over residue boundaries.
    ///
    /// # Panics
    /// Panics if `atom_idx >= n_atoms()`.
    pub fn residue_of_atom(&self, atom_idx: usize) -> usize {
        self.atom_to_residue.segment_of(atom_idx)
    }

    /// Returns the chain index that contains residue `res_idx`.
    ///
    /// O(log n) binary search over chain boundaries.
    ///
    /// # Panics
    /// Panics if `res_idx >= n_residues()`.
    pub fn chain_of_residue(&self, res_idx: usize) -> usize {
        self.residue_to_chain.segment_of(res_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info::elements::Element;
    use crate::model::tables::ResidueGroup;

    /// Build a test hierarchy: 3 chains, 10 residues (4+3+3), 30 atoms (3 per residue).
    fn make_test_hierarchy() -> AtomicHierarchy {
        // Chain 0: residues 0..4, Chain 1: residues 4..7, Chain 2: residues 7..10
        // Each residue has 3 atoms.
        let n_residues = 10;
        let n_atoms = 30; // 3 per residue
        let _n_chains = 3;

        let atoms = AtomsTable {
            atom_name: (0..n_atoms)
                .map(|i| {
                    match i % 3 {
                        0 => "N",
                        1 => "CA",
                        _ => "C",
                    }
                    .to_string()
                })
                .collect(),
            auth_atom_name: (0..n_atoms)
                .map(|i| {
                    match i % 3 {
                        0 => "N",
                        1 => "CA",
                        _ => "C",
                    }
                    .to_string()
                })
                .collect(),
            element: (0..n_atoms)
                .map(|i| if i % 3 == 0 { Element::N } else { Element::C })
                .collect(),
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };

        let residues = ResiduesTable {
            comp_id: (0..n_residues).map(|i| format!("RES{}", i)).collect(),
            auth_comp_id: (0..n_residues).map(|i| format!("RES{}", i)).collect(),
            label_seq_id: (0..n_residues as i32).collect(),
            auth_seq_id: (1..=n_residues as i32).collect(),
            ins_code: vec![None; n_residues],
            group: vec![ResidueGroup::Polymer; n_residues],
        };

        let chains = ChainsTable {
            label_asym_id: vec!["A".into(), "B".into(), "C".into()],
            auth_asym_id: vec!["A".into(), "B".into(), "C".into()],
            entity_id: vec!["1".into(), "2".into(), "3".into()],
        };

        // atom_to_residue: each residue has 3 atoms => offsets [0,3,6,...,30]
        let atom_offsets: Vec<u32> = (0..=n_residues as u32).map(|i| i * 3).collect();
        let atom_to_residue = Segmentation::from_offsets(atom_offsets);

        // residue_to_chain: chain 0 has 4 residues, chain 1 has 3, chain 2 has 3
        let residue_offsets: Vec<u32> = vec![0, 4, 7, 10];
        let residue_to_chain = Segmentation::from_offsets(residue_offsets);

        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);

        AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        }
    }

    #[test]
    fn test_hierarchy_basic() {
        let h = make_test_hierarchy();

        assert_eq!(h.n_atoms(), 30);
        assert_eq!(h.n_residues(), 10);
        assert_eq!(h.n_chains(), 3);

        // residue 0 -> atoms 0..3
        assert_eq!(h.atoms_in_residue(0), 0..3);
        // residue 4 -> atoms 12..15
        assert_eq!(h.atoms_in_residue(4), 12..15);
        // residue 9 -> atoms 27..30
        assert_eq!(h.atoms_in_residue(9), 27..30);

        // chain 0 -> residues 0..4
        assert_eq!(h.residues_in_chain(0), 0..4);
        // chain 1 -> residues 4..7
        assert_eq!(h.residues_in_chain(1), 4..7);
        // chain 2 -> residues 7..10
        assert_eq!(h.residues_in_chain(2), 7..10);
    }

    #[test]
    fn test_hierarchy_residue_of_atom() {
        let h = make_test_hierarchy();

        // atoms 0,1,2 -> residue 0
        assert_eq!(h.residue_of_atom(0), 0);
        assert_eq!(h.residue_of_atom(1), 0);
        assert_eq!(h.residue_of_atom(2), 0);
        // atoms 3,4,5 -> residue 1
        assert_eq!(h.residue_of_atom(3), 1);
        assert_eq!(h.residue_of_atom(5), 1);
        // atom 29 -> residue 9
        assert_eq!(h.residue_of_atom(29), 9);
    }

    #[test]
    fn test_hierarchy_chain_of_residue() {
        let h = make_test_hierarchy();

        // residues 0..4 -> chain 0
        for r in 0..4 {
            assert_eq!(
                h.chain_of_residue(r),
                0,
                "residue {} should be in chain 0",
                r
            );
        }
        // residues 4..7 -> chain 1
        for r in 4..7 {
            assert_eq!(
                h.chain_of_residue(r),
                1,
                "residue {} should be in chain 1",
                r
            );
        }
        // residues 7..10 -> chain 2
        for r in 7..10 {
            assert_eq!(
                h.chain_of_residue(r),
                2,
                "residue {} should be in chain 2",
                r
            );
        }
    }
}
