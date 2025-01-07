//! Module for handling atomic information in chemical/molecular structures.
//!
//! This module provides data structures and functionality for working with atomic data
//! including coordinates, element types, and atom metadata.
use pdbtbx::Element;

// Struct to hold atom information
#[derive(Debug)]
pub struct AtomInfo<'a> {
    pub index: usize,
    pub coords: &'a [f32; 3],
    pub element: &'a Element,
    pub atom_name: &'a String,
    pub is_hetero: bool,
}
