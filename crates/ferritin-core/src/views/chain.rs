// use super::atom::AtomView;
use super::residue::ResidueView;
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

    // Add to ChainView implementation
    pub fn iter_residues(&self) -> impl Iterator<Item = ResidueView<'_>> {
        let residue_indices = self.data.get_residue_start_indices().unwrap();

        // Convert relevant section of residue indices to atom indices
        let mut atom_indices: Vec<usize> = Vec::new();
        for residue_idx in self.start_residue_idx..=self.end_residue_idx {
            if residue_idx < residue_indices.len() {
                atom_indices.push(residue_indices[residue_idx] as usize);
            }
        }

        // If we didn't reach the end of the structure, the end index is the next residue start
        // Otherwise it's the end of the structure
        if self.end_residue_idx < residue_indices.len() {
            (0..atom_indices.len() - 1)
                .map(move |i| ResidueView::new(self.data, atom_indices[i], atom_indices[i + 1]))
        } else {
            // Handle case where this chain extends to the end of the structure
            (0..atom_indices.len() - 1)
                .map(move |i| ResidueView::new(self.data, atom_indices[i], atom_indices[i + 1]))
                .chain(if !atom_indices.is_empty() {
                    Some(ResidueView::new(
                        self.data,
                        *atom_indices.last().unwrap(),
                        self.data.get_size(),
                    ))
                } else {
                    None
                })
        }
    }

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
