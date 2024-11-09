//! Core module for iterating over residues in a molecular structure
//!
//! This module provides functionality to iterate over residues in an [AtomCollection],
//! including utilities to extract and collect residues into new collections.
//!
//! ```

use super::atoms::ResidueAtoms;
use crate::AtomCollection;
use crate::Selection;
use std::iter::FromIterator;

// Rest of the iterator implementation remains the same
pub struct ResidueIter<'a> {
    atom_collection: &'a AtomCollection,
    residue_starts: Vec<i64>,
    current_idx: usize,
}

impl<'a> ResidueIter<'a> {
    pub fn new(atom_collection: &'a AtomCollection, residue_starts: Vec<i64>) -> Self {
        ResidueIter {
            atom_collection,
            residue_starts,
            current_idx: 0,
        }
    }
}
impl<'a> Iterator for ResidueIter<'a> {
    type Item = ResidueAtoms<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.residue_starts.len() - 1 {
            return None;
        }

        let start_idx = self.residue_starts[self.current_idx] as usize;
        let end_idx = self.residue_starts[self.current_idx + 1] as usize;

        let atoms = Selection::new((start_idx..end_idx).collect());

        let residue = ResidueAtoms {
            start_idx,
            end_idx,
            res_id: self.atom_collection.get_res_id(start_idx).clone(),
            res_name: self.atom_collection.get_res_name(start_idx).clone(),
            chain_id: self.atom_collection.get_chain_id(start_idx).clone(),
            atoms,
            parent: self.atom_collection,
        };

        self.current_idx += 1;
        Some(residue)
    }
}

impl<'a> FromIterator<ResidueAtoms<'a>> for AtomCollection {
    fn from_iter<T: IntoIterator<Item = ResidueAtoms<'a>>>(iter: T) -> Self {
        let mut coords = Vec::new();
        let mut res_ids = Vec::new();
        let mut res_names = Vec::new();
        let mut is_hetero = Vec::new();
        let mut elements = Vec::new();
        let mut atom_names = Vec::new();
        let mut chain_ids = Vec::new();
        let mut size = 0;

        // Collect all atoms from the residues
        for residue in iter {
            for i in residue.start_idx..residue.end_idx {
                coords.push(*residue.parent.get_coord(i));
                res_ids.push(*residue.parent.get_res_id(i));
                res_names.push(residue.parent.get_res_name(i).clone());
                is_hetero.push(residue.parent.get_is_hetero(i));
                elements.push(residue.parent.get_element(i).clone());
                atom_names.push(residue.parent.get_atom_name(i).clone());
                chain_ids.push(residue.parent.get_chain_id(i).clone());
                size += 1;
            }
        }

        AtomCollection::new(
            size, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, None,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::info::constants::is_amino_acid;
    use crate::AtomCollection;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_collect_amino_acids() {
        // the collect creates a new AC. Improtant if we want to make new copies.
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        let ac = AtomCollection::from(&pdb);

        let amino_acids: AtomCollection = ac.iter_residues_aminoacid().collect();
        assert!(amino_acids.get_size() < ac.get_size());

        for residue in amino_acids.iter_residues_all() {
            assert!(is_amino_acid(&residue.res_name));
        }
    }
}
