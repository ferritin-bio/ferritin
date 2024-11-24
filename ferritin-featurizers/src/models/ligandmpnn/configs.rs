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

use super::featurizer::ProteinFeatures;
use super::model::ProteinMPNN;
use crate::models::ligandmpnn::featurizer::LMPNNFeatures;
use anyhow::Error;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use clap::ValueEnum;
use ferritin_core::AtomCollection;
use ferritin_test_data::TestFile;

// All Data Needed for running a model
pub struct MPNNExecConfig {
    pub protein_data: ProteinFeatures,
    pub run_config: RunConfig,
    pub protein_mpnn_model_config: ProteinMPNNConfig,
    pub aabias_config: Option<AABiasConfig>,
    pub ligand_mpnn_config: Option<LigandMPNNConfig>,
    pub membrane_mpnn_config: Option<MembraneMPNNConfig>,
    pub multi_pdb_config: Option<MultiPDBConfig>,
    pub residue_control_config: Option<ResidueControl>,
    // device: &candle_core::Device,
    seed: i32,
}

impl MPNNExecConfig {
    pub fn new(
        seed: i32,
        device: &Device,
        pdb_path: String,
        model_type: ModelTypes,
        run_config: RunConfig,
        residue_config: Option<ResidueControl>,
        aa_bias: Option<AABiasConfig>,
        lig_mpnn_specific: Option<LigandMPNNConfig>,
        membrane_mpnn_specific: Option<MembraneMPNNConfig>,
        multi_pdb_specific: Option<MultiPDBConfig>,
    ) -> Result<Self, Error> {
        // Core Protein Features
        let (pdb, _) = pdbtbx::open(pdb_path).expect("A PDB  or CIF file");
        let ac = AtomCollection::from(&pdb);

        // Note: featurize should matchon model type
        let features = ac.featurize(device)?;

        // Model parameters
        let model_config = match model_type {
            ModelTypes::ProteinMPNN => ProteinMPNNConfig::proteinmpnn(),
            _ => todo!(),
        };

        Ok(MPNNExecConfig {
            protein_data: features,
            protein_mpnn_model_config: model_config,
            run_config,
            aabias_config: aa_bias,
            ligand_mpnn_config: lig_mpnn_specific,
            membrane_mpnn_config: membrane_mpnn_specific,
            residue_control_config: residue_config,
            multi_pdb_config: multi_pdb_specific,
            seed,
            // device: device,
        })
    }
    // Todo: refactor this to use loader.
    pub fn load_model(&self) -> Result<ProteinMPNN, Error> {
        // this is a hidden dep....
        let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
        let vb = VarBuilder::from_pth(mpnn_file, DType::F32, &Device::Cpu)?;
        let pconf = ProteinMPNNConfig::proteinmpnn();
        // ProteinMPNN::load(self.protein_mpnn_model_config.clone(), vb)
        Ok(ProteinMPNN::load(vb, &pconf).expect("Unable to load the PMPNN Model"))
    }
    pub fn load_protein(&mut self) {
        todo!();
    }
}

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
    pub bias_aa: Option<String>,
    pub bias_aa_per_residue: Option<String>,
    pub omit_aa: Option<String>,
    pub omit_aa_per_residue: Option<String>,
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
    pub omit_aa_per_residue_multi: Option<String>,
    pub bias_aa_per_residue_multi: Option<String>,
}
#[derive(Clone, Debug)]
pub struct ProteinMPNNConfig {
    pub atom_context_num: usize,
    pub augment_eps: f32,
    pub dropout_ratio: f32,
    pub edge_features: i64,
    pub hidden_dim: i64,
    pub k_neighbors: i64,
    pub ligand_mpnn_use_side_chain_context: bool,
    pub model_type: ModelTypes,
    pub node_features: i64,
    pub num_decoder_layers: i64,
    pub num_encoder_layers: i64,
    pub num_letters: i64,
    pub num_rbf: i64,
    pub scale_factor: f64,
    pub vocab: i64,
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
