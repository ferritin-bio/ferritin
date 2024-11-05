//! PMPNN Core Config and Builder API
//!
//! This module provides configuration structs and builders for the PMPNN protein design system.
//!
//! # Core Configuration Types
//!
//! - `ModelTypes` - Enum of supported model architectures
//! - `ProteinMPNNConfig` - Core model parameters
//! - `AABiasConfig` - Amino acid biasing controls
//! - `LigandMPNNConfig` - LigandMPNN specific settings
//! - `MembraneMPNNConfig` - MembraneMPNN specific settings
//! - `MultiPDBConfig` - Multi-PDB mode configuration
//! - `ResidueControl` - Residue-level design controls
//! - `RunConfig` - Runtime execution parameters// Core Configs for handling CLI ARGs and Model Params

use clap::ValueEnum;

#[derive(Debug, Clone, ValueEnum)]
pub enum ModelTypes {
    #[value(name = "protein_mpnn")]
    ProteinMPNN,
    #[value(name = "ligand_mpnn")]
    LigandMPNN,
}

#[derive(Debug)]
/// Amino Acid Biasing
pub struct AABiasConfig {
    pub bias_AA: Option<String>,
    pub bias_AA_per_residue: Option<String>,
    pub omit_AA: Option<String>,
    pub omit_AA_per_residue: Option<String>,
}

/// LigandMPNN Specific
pub struct LigandMPNNConfig {
    pub checkpoint_ligand_mpnn: Option<String>,
    pub ligand_mpnn_use_atom_context: Option<i32>,
    pub ligand_mpnn_use_side_chain_context: Option<i32>,
    pub ligand_mpnn_cutoff_for_score: Option<String>,
}

/// Membrane MPNN Specific
pub struct MembraneMPNNConfig {
    pub global_transmembrane_label: Option<i32>,
    pub transmembrane_buried: Option<String>,
    pub transmembrane_interface: Option<String>,
}

/// Multi-PDB Related
pub struct MultiPDBConfig {
    pub pdb_path_multi: Option<String>,
    pub fixed_residues_multi: Option<String>,
    pub redesigned_residues_multi: Option<String>,
    pub omit_AA_per_residue_multi: Option<String>,
    pub bias_AA_per_residue_multi: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ProteinMPNNConfig {
    atom_context_num: usize,
    augment_eps: f32,
    dropout_ratio: f32,
    edge_features: i64,
    hidden_dim: i64,
    k_neighbors: i64,
    ligand_mpnn_use_side_chain_context: bool,
    model_type: ModelTypes,
    node_features: i64,
    num_decoder_layers: i64,
    num_encoder_layers: i64,
    num_letters: i64,
    num_rbf: i64,
    scale_factor: f64,
    vocab: i64,
}

impl ProteinMPNNConfig {
    pub fn proteinmpnn() -> Self {
        Self {
            atom_context_num: 0,
            augment_eps: 0.0,
            dropout_ratio: 0.1,
            edge_features: 128,
            hidden_dim: 128,
            k_neighbors: 24,
            ligand_mpnn_use_side_chain_context: false,
            model_type: ModelTypes::ProteinMPNN,
            node_features: 128,
            num_decoder_layers: 3,
            num_encoder_layers: 3,
            num_letters: 48,
            num_rbf: 16,
            scale_factor: 30.0,
            vocab: 48,
        }
    }
    fn ligandmpnn() {
        todo!()
    }
    fn membranempnn() {
        todo!()
    }
}

#[derive(Debug)]
pub struct ResidueControl {
    pub fixed_residues: Option<String>,
    pub redesigned_residues: Option<String>,
    pub symmetry_residues: Option<String>,
    pub symmetry_weights: Option<String>,
    pub chains_to_design: Option<String>,
    pub parse_these_chains_only: Option<String>,
}

#[derive(Debug)]
pub struct RunConfig {
    pub temperature: Option<f32>,
    pub verbose: Option<i32>,
    pub save_stats: Option<i32>,
    pub batch_size: Option<i32>,
    pub number_of_batches: Option<i32>,
    pub file_ending: Option<String>,
    pub zero_indexed: Option<i32>,
    pub homo_oligomer: Option<i32>,
    pub fasta_seq_separation: Option<String>,
}
