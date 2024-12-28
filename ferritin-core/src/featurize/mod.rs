//! Protein Featurizer for ProteinMPNN/LignadMPNN
//!
//! Extract protein features for ligandmpnn
//!
//! Returns a set of features calculated from protein structure
//! including:
//! - Residue-level features like amino acid type, secondary structure
//! - Geometric features like distances, angles
//! - Chemical features like hydrophobicity, charge
//! - Evolutionary features from MSA profiles
// mod ndarray_impl;
mod structure_features;
mod utilities;
