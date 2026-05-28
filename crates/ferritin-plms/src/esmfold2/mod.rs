//! ESMFold2 structure-prediction model.
//!
//! This module provides the input data structures, configuration, layers, and
//! model wiring for the ESMFold2 structure-prediction model from
//! EvolutionaryScale. It is ported from the Python `esm.models.esmfold2` SDK.
//!
//! The primary entry point is [`input_types::StructurePredictionInput`], which
//! is assembled from [`input_types::ProteinInput`], [`input_types::DNAInput`],
//! and [`input_types::LigandInput`] chains using a builder API.

pub mod config;
pub mod input_types;
pub mod layers;
pub mod model;
pub mod output;
pub mod pretrained;

pub use input_types::{
    ChainInput, DNAInput, LigandInput, Modification, ProteinInput, StructurePredictionInput,
    validate_dna_sequence, validate_protein_sequence,
};
