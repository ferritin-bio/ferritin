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
            is_hetero: self.parent.is_hetero(i),
        })
    }

    // Get specific atom by index within residue
    pub fn get_atom(&self, residue_atom_idx: usize) -> Option<AtomInfo> {
        let abs_idx = self.start_idx + residue_atom_idx;
        if abs_idx < self.end_idx {
            Some(AtomInfo {
                index: abs_idx,
                coords: &self.parent.coords[abs_idx],
                element: &self.parent.elements[abs_idx],
                atom_name: &self.parent.atom_names[abs_idx],
                is_hetero: self.parent.is_hetero[abs_idx],
            })
        } else {
            None
        }
    }

    // Find atom by name within this residue
    pub fn find_atom_by_name(&self, name: &str) -> Option<AtomInfo> {
        (self.start_idx..self.end_idx)
            .find(|&i| self.parent.atom_names[i] == name)
            .map(|i| AtomInfo {
                index: i,
                coords: &self.parent.coords[i],
                element: &self.parent.elements[i],
                atom_name: &self.parent.atom_names[i],
                is_hetero: self.parent.is_hetero[i],
            })
    }
}
