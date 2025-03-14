// use super::atom::AtomView;
// use super::residue::ResidueView;
use crate::AtomCollection;
// use crate::selection::Selection;

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

    // // Iterate through residues in this chain
    // pub fn iter_residues(&self) -> impl Iterator<Item = ResidueView<'a>> {
    //     let residue_starts = self.data.residue_start_indices.as_ref().unwrap();
    //     let start_indices: Vec<i64> = residue_starts[self.start_residue_idx..self.end_residue_idx]
    //         .iter()
    //         .map(|&i| i as i64)
    //         .collect();

    //     ResidueView::new(self.data, start_indices)
    // }

    // Get the number of residues in this chain
    pub fn residue_count(&self) -> usize {
        self.end_residue_idx - self.start_residue_idx
    }

    // // Get a view of all atoms in this chain
    // pub fn atom_view(&self) -> AtomView<'a> {
    //     // Create a selection of all atoms in this chain
    //     let residue_starts = self.data.residue_start_indices.as_ref().unwrap();
    //     let first_atom = residue_starts[self.start_residue_idx] as usize;
    //     let last_atom = if self.end_residue_idx < residue_starts.len() {
    //         residue_starts[self.end_residue_idx] as usize - 1
    //     } else {
    //         self.data.size - 1 // Last atom in structure
    //     };

    //     let indices: Vec<usize> = (first_atom..=last_atom).collect();
    //     let selection = Selection::new(indices);
    //     AtomView::new(self.data, selection)
    // }
}
