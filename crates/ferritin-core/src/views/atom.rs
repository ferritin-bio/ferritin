//! Provides a read-only view into an atom from an AtomCollection.
//!
//! This module defines the `AtomView` struct, which is a lightweight handle that
//! references an atom in an existing `AtomCollection` without taking ownership.
//! It provides accessor methods to retrieve atom properties such as coordinates,
//! element, name, residue information, and more.

use crate::AtomCollection;
use crate::info::elements::Element;

pub struct AtomView<'a> {
    pub(crate) data: &'a AtomCollection,
    pub(crate) idx: usize,
}

impl<'a> AtomView<'a> {
    pub fn new(data: &'a AtomCollection, idx: usize) -> Self {
        AtomView { data, idx }
    }
    pub fn coords(&self) -> &'a [f32; 3] {
        self.data.get_coord(self.idx)
    }
    pub fn element(&self) -> &'a Element {
        self.data.get_element(self.idx)
    }
    pub fn atom_name(&self) -> &'a String {
        self.data.get_atom_name(self.idx)
    }
    pub fn residue_id(&self) -> i32 {
        *self.data.get_res_id(self.idx)
    }
    pub fn residue_name(&self) -> &'a String {
        self.data.get_res_name(self.idx)
    }
    pub fn chain_id(&self) -> &'a String {
        self.data.get_chain_id(self.idx)
    }
    pub fn is_hetero(&self) -> bool {
        self.data.get_is_hetero(self.idx)
    }
    // Add index accessor if needed for direct access
    pub fn index(&self) -> usize {
        self.idx
    }
}
