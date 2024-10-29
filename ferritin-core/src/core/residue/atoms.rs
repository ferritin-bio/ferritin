//! Provides functionality for working with residue atoms in molecular structures.
//!
//! This module contains the `ResidueAtoms` struct and its implementation, which provides methods for
//! accessing and manipulating atoms within a single residue of a molecular structure. A residue is
//! a structural unit in a protein or other macromolecule consisting of multiple atoms.
//!
use super::info::AtomInfo;
use crate::core::selection::AtomView;
use crate::core::{AtomCollection, Selection};
use pdbtbx::Element;

pub struct ResidueAtoms<'a> {
    pub start_idx: usize,
    pub end_idx: usize,
    pub res_id: i32,
    pub res_name: String,
    pub chain_id: String,
    pub atoms: Selection,
    pub parent: &'a AtomCollection,
}

impl<'a> ResidueAtoms<'a> {
    // Get all atom coordinates for this residue
    pub fn coords(&self) -> Vec<&[f32; 3]> {
        (self.start_idx..self.end_idx)
            .map(|i| self.parent.get_coord(i))
            .collect()
    }

    // Get all atom names for this residue
    pub fn atom_names(&self) -> Vec<&String> {
        (self.start_idx..self.end_idx)
            .map(|i| self.parent.get_atom_name(i))
            .collect()
    }

    // Get all elements for this residue
    pub fn elements(&self) -> Vec<&Element> {
        (self.start_idx..self.end_idx)
            .map(|i| self.parent.get_element(i))
            .collect()
    }

    // Get atom view for this residue
    pub fn view(&self) -> AtomView {
        self.parent.view(self.atoms.clone())
    }

    // Get number of atoms in this residue
    pub fn atom_count(&self) -> usize {
        self.end_idx - self.start_idx
    }

    // Iterator over atoms in this residue
    pub fn iter_atoms(&self) -> impl Iterator<Item = AtomInfo> + '_ {
        (self.start_idx..self.end_idx).map(|i| AtomInfo {
            index: i,
            coords: self.parent.get_coord(i),
            element: self.parent.get_element(i),
            atom_name: self.parent.get_atom_name(i),
            is_hetero: self.parent.get_is_hetero(i),
        })
    }

    // Get specific atom by index within residue
    pub fn get_atom(&self, residue_atom_idx: usize) -> Option<AtomInfo> {
        let abs_idx = self.start_idx + residue_atom_idx;
        if abs_idx < self.end_idx {
            Some(AtomInfo {
                index: abs_idx,
                coords: &self.parent.get_coord(abs_idx),
                element: &self.parent.get_element(abs_idx),
                atom_name: &self.parent.get_atom_name(abs_idx),
                is_hetero: self.parent.get_is_hetero(abs_idx),
            })
        } else {
            None
        }
    }

    // Find atom by name within this residue
    pub fn find_atom_by_name(&self, name: &str) -> Option<AtomInfo> {
        (self.start_idx..self.end_idx)
            .find(|&i| self.parent.get_atom_name(i) == name)
            .map(|i| AtomInfo {
                index: i,
                coords: &self.parent.get_coord(i),
                element: &self.parent.get_element(i),
                atom_name: &self.parent.get_atom_name(i),
                is_hetero: self.parent.get_is_hetero(i),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::test_utilities::get_atom_container;
    use crate::core::AtomCollection;

    #[test]
    fn test_iteration() {
        let ac: AtomCollection = get_atom_container();
        assert_eq!(ac.iter_residues_aminoacid().count(), 154);

        let first_residue: ResidueAtoms = ac
            .iter_residues_aminoacid()
            .take(1)
            .next()
            .expect("Should have at least one amino acid residue");

        assert_eq!(first_residue.start_idx, 0);
        assert_eq!(first_residue.end_idx, 8); // Met has 9 Atoms
        assert_eq!(first_residue.res_id, 0);
        assert_eq!(first_residue.res_name, "MET");
        assert_eq!(first_residue.chain_id, "A");
        // assert_eq!(first_residue.atoms, "A"); // <- should test Selection
        assert_eq!(first_residue.parent.get_size(), 1413);
    }
}
