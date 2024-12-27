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
mod conversions;
mod featurize;
mod info;
mod residue;
mod selection;

pub use self::atomcollection::AtomCollection;
pub use self::bonds::{Bond, BondOrder};
pub use self::featurize::{aa1to_int, aa3to1, int_to_aa1, ProteinFeatures, StructureFeatures};
pub use self::residue::ResidueAtoms;
pub use self::selection::Selection;
