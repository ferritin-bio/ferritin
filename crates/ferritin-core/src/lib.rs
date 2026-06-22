//! # ferritin-core
//!
//! A library for working with biomolecular structure files and performing common operations.
//!
//! __ferritin-core__ provides functionality for:
//! * Reading and writing common biomolecular file formats (PDB, mmCIF, etc.)
//! * Selecting atoms and residues based on various criteria
//! * Computing geometric properties like distances, angles, and dihedrals
//! * Basic molecular operations like superposition and RMSD calculations
//!
//! The main entry point is the [`AtomCollection`] struct which represents a biomolecular structure
//! and provides methods for manipulating and analyzing it.
//!
mod atomcollection;
mod bonds;
pub mod data;
pub mod info;
mod io;
pub mod model;
mod views;

pub use atomcollection::AtomCollection;
pub use bonds::{Bond, BondOrder};
pub use io::{load_structure, load_structure_from_string};
pub use model::{AtomicConformation, AtomicHierarchy, Bonds, Model};
pub use views::{AtomView, ChainView, ResidueView};
