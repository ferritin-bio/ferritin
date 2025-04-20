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
    pub fn iter_residues(&self) -> impl Iterator<Item = ResidueView<'_>> {
        let residue_indices = self.data.get_residue_start_indices().unwrap();
        let atom_indices: Vec<usize> = (self.start_residue_idx
            ..=self
                .end_residue_idx
                .min(residue_indices.len().saturating_sub(1)))
            .map(|idx| residue_indices[idx] as usize)
            .collect();
        // Get the last atom index if it exists
        let last_atom_idx = atom_indices.last().copied();
        // Get the end atom index for the last residue
        let last_residue_end_idx = if self.end_residue_idx < residue_indices.len().saturating_sub(1)
        {
            // If not the final residue in the structure, use the next residue's start
            residue_indices[self.end_residue_idx + 1] as usize
        } else {
            // If it's the final residue, use the structure's end
            self.data.get_size()
        };
        // Generate normal residue views
        (0..atom_indices.len().saturating_sub(1))
            .map(move |i| ResidueView::new(self.data, atom_indices[i], atom_indices[i + 1]))
            // Add the last residue only if there are any indices
            .chain(
                last_atom_idx
                    .map(|idx| ResidueView::new(self.data, idx, last_residue_end_idx))
                    .into_iter(),
            )
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
