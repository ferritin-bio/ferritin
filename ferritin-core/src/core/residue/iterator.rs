use super::atoms::ResidueAtoms;
use crate::core::AtomCollection;

// Rest of the iterator implementation remains the same
pub struct ResidueIter<'a> {
    atom_collection: &'a AtomCollection,
    residue_starts: Vec<i64>,
    current_idx: usize,
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
            res_id: self.atom_collection.res_ids[start_idx],
            res_name: self.atom_collection.res_names[start_idx].clone(),
            chain_id: self.atom_collection.chain_ids[start_idx].clone(),
            atoms,
            parent: self.atom_collection,
        };

        self.current_idx += 1;
        Some(residue)
    }
}
