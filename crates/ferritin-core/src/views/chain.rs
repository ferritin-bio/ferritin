use super::residue::ResidueView;
use crate::AtomCollection;

/// View representing a molecular chain.
pub struct ChainView<'a> {
    pub(crate) data: &'a AtomCollection,
    pub(crate) start_residue_idx: usize,
    pub(crate) end_residue_idx: usize,
}

impl<'a> ChainView<'a> {
    pub fn chain_id(&self) -> &str {
        // All atoms in this chain should have the same chain ID
        // We use the first residue's first atom to identify the chain
        let residue_starts = self.data.get_residue_start_indices().unwrap();
        let first_atom_idx = residue_starts[self.start_residue_idx] as usize;
        self.data.get_chain_id(first_atom_idx)
    }
    pub fn iter_residues(&self) -> Box<dyn Iterator<Item = ResidueView<'_>> + '_> {
        let residue_indices = self.data.get_residue_start_indices().unwrap();
        // Convert relevant section of residue indices to atom indices
        let atom_indices: Vec<usize> = (self.start_residue_idx..=self.end_residue_idx)
            .filter(|&idx| idx < residue_indices.len())
            .map(|idx| residue_indices[idx] as usize)
            .collect();
        // If the list is empty, return an empty iterator
        if atom_indices.is_empty() {
            return Box::new(std::iter::empty());
        }
        // Make a copy of the last atom index for use after we move atom_indices
        let last_atom_idx = *atom_indices.last().unwrap();

        // If we have indices but not reaching the end
        if self.end_residue_idx < residue_indices.len() {
            // Create pairs of (start,end) for each residue
            let iter = (0..atom_indices.len() - 1)
                .map(move |i| ResidueView::new(self.data, atom_indices[i], atom_indices[i + 1]));
            Box::new(iter)
        } else {
            let main_residues = (0..atom_indices.len() - 1)
                .map(move |i| ResidueView::new(self.data, atom_indices[i], atom_indices[i + 1]));

            // For the last residue
            let last_residue = std::iter::once(ResidueView::new(
                self.data,
                last_atom_idx,
                self.data.get_size(),
            ));
            Box::new(main_residues.chain(last_residue))
        }
    }
    // Get the number of residues in this chain
    pub fn residue_count(&self) -> usize {
        self.end_residue_idx - self.start_residue_idx
    }
}

#[cfg(test)]
mod tests {
    use crate::load_structure;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_chain_view() {
        let (prot_file, _temp) = TestFile::protein_04().create_temp().unwrap();
        let mut ac = load_structure(prot_file).unwrap();
        // Calculate indices first
        ac.calculate_chain_indices();

        let chains: Vec<_> = ac.iter_chains().collect();
        assert!(chains.len() > 0);

        // Test first chain
        let first_chain = &chains[0];
        assert!(!first_chain.chain_id().is_empty());
        assert!(first_chain.residue_count() > 0);

        // Test residue iteration
        let residues: Vec<_> = first_chain.iter_residues().collect();
        assert_eq!(residues.len(), first_chain.residue_count());

        // Test chain ID consistency
        let chain_id = first_chain.chain_id();
        for residue in residues {
            assert_eq!(residue.chain_id(), chain_id);
        }
    }
}
