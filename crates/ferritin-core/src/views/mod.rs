//! # AtomCollection View Module
//!
//! This module provides view-based iterators over atomcollection.
//! It enables efficient traversal of the molecular hierarchy (chains, residues, atoms)
//! without copying data, using a struct-of-arrays underlying representation.
//!
//! ## Example
//! ```
//! use molecule::MoleculeData;
//!
//! fn process_molecule(mol: &MoleculeData) {
//!     // Iterate through all chains
//!     for chain in mol.chains() {
//!         println!("Chain: {}", chain.id());
//!
//!         // Iterate through residues in a chain
//!         for residue in chain.residues() {
//!             println!("  Residue: {}{}", residue.name(), residue.number());
//!
//!             // Iterate through atoms in a residue
//!             for atom in residue.atoms() {
//!                 println!("    Atom: {} at {:?}", atom.name(), atom.position());
//!             }
//!         }
//!     }
//! }
//! ```
mod atom;
mod chain;
mod residue;
