//! Model layer (Layer 1): atomic hierarchy, conformation, and bond connectivity.
//!
//! This module provides the core data structures for representing molecular
//! structures.  It sits on top of the Layer 0 primitives in [`crate::data`].
//!
//! # Architecture
//!
//! ```text
//! Model  ──── Arc<AtomicHierarchy>  (topology: atoms, residues, chains, bonds)
//!        └─── AtomicConformation    (coordinates: x, y, z per atom)
//! ```
//!
//! Multiple `Model`s (trajectory frames) share one `AtomicHierarchy` via `Arc`.

pub mod bonds;
pub mod conformation;
mod error;
pub mod hierarchy;
pub mod model;
pub mod symmetry;
pub mod tables;

pub use bonds::Bonds;
pub use conformation::AtomicConformation;
pub use error::ModelError;
pub use hierarchy::AtomicHierarchy;
pub use model::Model;
pub use symmetry::{
    Assembly, AssemblyUnit, CrystalSymmetry, IDENTITY_MAT4, Mat4, SymmetryData, SymmetryOperator,
};
pub use tables::{AtomsTable, ChainsTable, MISSING_SEQ_ID, ResidueGroup, ResiduesTable};
