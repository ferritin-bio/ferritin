use super::atom::AtomView;
use crate::AtomCollection;
use crate::info::constants::{is_amino_acid, is_carbohydrate, is_nucleotide};

/// View representing a residue (amino acid, nucleotide, etc.) in the molecule.
pub struct ResidueView<'a> {
    pub(crate) data: &'a AtomCollection,
    pub(crate) start_atom_idx: usize,
    pub(crate) end_atom_idx: usize,
}

impl<'a> ResidueView<'a> {
    pub fn new(data: &'a AtomCollection, start_atom_idx: usize, end_atom_idx: usize) -> Self {
        ResidueView {
            data,
            start_atom_idx,
            end_atom_idx,
        }
    }

    pub fn atom_count(&self) -> usize {
        self.end_atom_idx - self.start_atom_idx
    }

    pub fn chain_id(&self) -> &str {
        self.data.get_chain_id(self.start_atom_idx)
    }

    pub fn is_amino_acid(&self) -> bool {
        is_amino_acid(self.residue_name())
    }

    // Add methods for other residue types
    pub fn is_nucleotide(&self) -> bool {
        is_nucleotide(self.residue_name())
    }

    pub fn is_carbohydrate(&self) -> bool {
        is_carbohydrate(self.residue_name())
    }

    pub fn iter_atoms(&self) -> impl Iterator<Item = AtomView<'_>> + '_ {
        (self.start_atom_idx..self.end_atom_idx).map(move |idx| AtomView::new(self.data, idx))
    }

    pub fn residue_id(&self) -> i32 {
        *self.data.get_res_id(self.start_atom_idx)
    }

    pub fn residue_name(&self) -> &str {
        &self.data.get_res_name(self.start_atom_idx)
    }

    // pub fn is_hetero(&self) -> bool {
    //     self.data.get_is_hetero(self.start_atom_idx)
    // }

    // Get atom by name within this residue
    pub fn find_atom_by_name(&self, name: &str) -> Option<AtomView<'a>> {
        (self.start_atom_idx..self.end_atom_idx)
            .find(|&i| self.data.get_atom_name(i) == name)
            .map(|idx| AtomView::new(self.data, idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::load_structure;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_residue_view_properties() {
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let ac = load_structure(prot_file).unwrap();

        // Create a simple residue view for testing
        let residue = ResidueView::new(&ac, 0, 10); // First 10 atoms as a test

        // Test basic properties
        assert!(!residue.residue_name().is_empty());
        assert!(residue.residue_id() >= 0);
        assert!(!residue.chain_id().is_empty());

        // Test atom count
        assert_eq!(residue.atom_count(), 10);
    }
}
