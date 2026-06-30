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

    /// Returns the half-open range of atom indices `[start, end)` for this residue.
    pub fn atom_range(&self) -> std::ops::Range<usize> {
        self.start_atom_idx..self.end_atom_idx
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

    pub fn is_hetero(&self) -> bool {
        self.data.get_is_hetero(self.start_atom_idx)
    }

    pub fn is_water(&self) -> bool {
        matches!(self.residue_name(), "HOH" | "WAT" | "H2O")
    }

    /// Single-atom hetero residue that is not water (e.g. Na+, Mg2+, Zn2+).
    pub fn is_ion(&self) -> bool {
        self.atom_count() == 1 && self.is_hetero() && !self.is_water()
    }

    // Get atom by name within this residue
    pub fn find_atom_by_name(&self, name: &str) -> Option<AtomView<'a>> {
        (self.start_atom_idx..self.end_atom_idx)
            .find(|&i| self.data.get_atom_name(i) == name)
            .map(|idx| AtomView::new(self.data, idx))
    }
}
