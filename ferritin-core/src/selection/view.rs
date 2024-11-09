//! A crate for viewing and manipulating atomic coordinates and properties.
//!
//! This crate provides functionality for working with collections of atoms, including:
//!
//! - View-based access to atomic coordinates and properties
//! - Iteration over selected atoms
//! - Access to atomic properties like coordinates, residue IDs, residue names, and elements
//!
//! The main types are:
//!
//! - [`AtomView`] - A view into a subset of atoms in a collection
//! - [`AtomRef`] - A reference to atomic properties
//! - [`AtomIterator`] - An iterator over atoms in a view
//!

use super::selection::Selection;
use crate::AtomCollection;
use pdbtbx::Element;

pub struct AtomView<'a> {
    /// Reference to the underlying atom collection
    collection: &'a AtomCollection,
    /// The selected subset of atoms in the collection
    selection: Selection,
}

impl<'a> AtomView<'a> {
    pub(crate) fn new(collection: &'a AtomCollection, selection: Selection) -> Self {
        AtomView {
            collection,
            selection,
        }
    }
    pub fn coords(&self) -> Vec<[f32; 3]> {
        self.selection
            .indices
            .iter()
            .map(|&i| *self.collection.get_coord(i))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.selection.indices.len()
    }
}

/// A reference to an atom's properties including coordinates, residue info, and element
pub struct AtomRef<'a> {
    /// 3D coordinates of the atom [x, y, z]
    pub coords: &'a [f32; 3],
    /// Residue identifier number
    pub res_id: &'a i32,
    /// Residue name (e.g. ALA, GLY, etc)
    pub res_name: &'a String,
    /// Chemical element of the atom
    pub element: &'a Element,
    // ... other fields
}

/// An iterator over atoms in an [`AtomView`], yielding [`AtomRef`]s
pub struct AtomIterator<'a> {
    /// Reference to the atom view being iterated over
    view: &'a AtomView<'a>,
    /// Current index into the selection indices
    current: usize,
}

impl<'a> IntoIterator for &'a AtomView<'a> {
    type Item = AtomRef<'a>;
    type IntoIter = AtomIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        AtomIterator {
            view: self,
            current: 0,
        }
    }
}

impl<'a> Iterator for AtomIterator<'a> {
    type Item = AtomRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.view.selection.indices.len() {
            return None;
        }

        let idx = self.view.selection.indices[self.current];
        self.current += 1;

        // Get the actual values first
        let coords = self.view.collection.get_coord(idx);
        let res_id = self.view.collection.get_res_id(idx);
        let res_name = self.view.collection.get_res_name(idx);
        let element = self.view.collection.get_element(idx);

        Some(AtomRef {
            coords,
            res_id,
            res_name,
            element,
        })
    }
}
