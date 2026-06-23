//! # View Module
//!
//! Lightweight read-only views over [`AtomCollection`] and [`Model`] data.
//! All views borrow their backing data and carry no heap allocation.
pub mod atom;
pub mod chain;
pub mod model_atom;
pub mod model_chain;
pub mod model_residue;
pub mod residue;
pub use atom::AtomView;
pub use chain::ChainView;
pub use model_atom::ModelAtomView;
pub use model_chain::ModelChainView;
pub use model_residue::ModelResidueView;
pub use residue::ResidueView;
