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

#[cfg(feature = "candle-backend")]
pub mod candle_impl;

#[cfg(feature = "ndarray-backend")]
pub mod ndarray_impl;

#[cfg(feature = "candle-backend")]
pub use candle_impl::*;

#[cfg(feature = "ndarray-backend")]
pub use ndarray_impl::*;

use crate::AtomCollection;
use std::collections::{HashMap, HashSet};

// /// Convert the AtomCollection into a struct that can be passed to a model.
// pub trait LMPNNFeatures {
//     fn encode_amino_acids(&self, device: &Device) -> Result<Tensor>; // ( residue types )
//     fn featurize(&self, device: &Device) -> Result<ProteinFeatures>; // need more control over this featurization process
//     fn get_res_index(&self) -> Vec<u32>;
//     fn to_numeric_backbone_atoms(&self, device: &Device) -> Result<Tensor>; // [residues, N/CA/C/O, xyz]

//     fn to_numeric_atom37(&self, device: &Device) -> Result<Tensor>; // [residues, N/CA/C/O....37, xyz]
//     fn to_numeric_ligand_atoms(&self, device: &Device) -> Result<(Tensor, Tensor, Tensor)>; // ( positions , elements, mask )
//     fn to_pdb(&self); //
// }

pub trait StructureFeatures<T> {
    type Error;

    /// Convert amino acid sequence to numeric representation
    fn encode_amino_acids(&self) -> Result<T, Self::Error>;

    /// Convert structure into complete feature set
    fn featurize(&self) -> Result<ProteinFeatures<T>, Self::Error>;

    /// Get residue indices
    fn get_res_index(&self) -> Vec<u32>;

    /// Extract backbone atom coordinates (N, CA, C, O)
    fn to_numeric_backbone_atoms(&self) -> Result<T, Self::Error>;

    /// Extract all atom coordinates in standard ordering
    fn to_numeric_atom37(&self) -> Result<T, Self::Error>;

    /// Extract ligand atom coordinates and properties
    fn to_numeric_ligand_atoms(&self) -> Result<(T, T, T), Self::Error>;

    /// Convert to PDB format
    fn to_pdb(&self);
}

pub struct ProteinFeatures<T> {
    /// Protein amino acid sequence encoding
    pub sequence: T,
    /// Protein atom coordinates
    pub coordinates: T,
    /// Protein atom mask
    pub atom_mask: Option<T>,
    /// Ligand coordinates
    pub ligand_coords: T,
    /// Ligand atom types
    pub ligand_types: T,
    /// Ligand mask
    pub ligand_mask: Option<T>,
    /// Residue indices
    pub residue_indices: T,
    /// Chain labels
    pub chain_labels: Option<Vec<f64>>,
    /// Chain letters
    pub chain_letters: Vec<String>,
    /// Chain mask
    pub chain_mask: Option<T>,
    /// List of chains
    pub chain_list: Vec<String>,
    pub device: Option<T>,
}

impl<T> ProteinFeatures<T> {
    pub fn get_coords(&self) -> &T {
        &self.coordinates
    }

    pub fn get_sequence(&self) -> &T {
        &self.sequence
    }

    pub fn get_sequence_mask(&self) -> Option<&T> {
        self.atom_mask.as_ref()
    }

    pub fn get_residue_index(&self) -> &T {
        &self.residue_indices
    }

    pub fn get_encoded(
        &self,
    ) -> Result<
        (Vec<String>, HashMap<String, usize>, HashMap<usize, String>),
        Box<dyn std::error::Error>,
    >
    where
        T: AsRef<[u32]>,
    {
        // Implementation
        todo!()
    }
}
