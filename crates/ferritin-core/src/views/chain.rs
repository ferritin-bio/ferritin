use super::atom::AtomView;
use super::residue::ResidueView;
use crate::AtomCollection;

/// View representing a molecular chain.
pub struct ChainView<'a> {
    data: &'a AtomCollection,
    start_residue_idx: usize,
    end_residue_idx: usize,
}

impl<'a> ChainView<'a> {
    // Method to iterate through residues in this chain
    pub fn residues(&self) -> impl Iterator<Item = ResidueView<'a>> + '_ {
        (self.start_residue_idx..self.end_residue_idx).map(move |residue_idx| {
            let start_atom = self.data.residue_start_indices[residue_idx];
            let end_atom = self
                .data
                .residue_start_indices
                .get(residue_idx + 1)
                .copied()
                .unwrap_or(self.data.atom_positions.len());

            ResidueView {
                data: self.data,
                start_atom_idx: start_atom,
                end_atom_idx: end_atom,
                residue_idx,
            }
        })
    }

    // Convenience method to get all atoms in this chain
    pub fn atoms(&self) -> impl Iterator<Item = AtomView<'a>> + '_ {
        self.residues().flat_map(|residue| residue.atoms())
    }
}
