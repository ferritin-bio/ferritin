//! # AtomCollection View Module
//!
//! This module provides view-based iterators over atomcollection.
//! It enables efficient traversal of the molecular hierarchy (chains, residues, atoms)
//! without copying data, using a struct-of-arrays underlying representation.
//!

pub mod atom;
pub mod chain;
pub mod residue;
