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
use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use clap::ValueEnum;
use ferritin_core::AtomCollection;
use ferritin_test_data::TestFile;

/// Responsible for taking CLI args and returning the Features and Model
///
pub struct MPNNExecConfig {
    pub(crate) protein_inputs: String, // Todo: make this optionally plural
    pub(crate) run_config: RunConfig,
    pub(crate) aabias_config: Option<AABiasConfig>,
    pub(crate) ligand_mpnn_config: Option<LigandMPNNConfig>,
    pub(crate) membrane_mpnn_config: Option<MembraneMPNNConfig>,
    pub(crate) multi_pdb_config: Option<MultiPDBConfig>,
    pub(crate) residue_control_config: Option<ResidueControl>,
    pub(crate) device: Device,
}

impl MPNNExecConfig {
    pub fn new(
        device: Device,
        pdb_path: String,
        run_config: RunConfig,
        residue_config: Option<ResidueControl>,
        aa_bias: Option<AABiasConfig>,
        lig_mpnn_specific: Option<LigandMPNNConfig>,
        membrane_mpnn_specific: Option<MembraneMPNNConfig>,
        multi_pdb_specific: Option<MultiPDBConfig>,
    ) -> Result<Self, Error> {
        Ok(MPNNExecConfig {
            protein_inputs: pdb_path,
            run_config,
            aabias_config: aa_bias,
            ligand_mpnn_config: lig_mpnn_specific,
            membrane_mpnn_config: membrane_mpnn_specific,
            residue_control_config: residue_config,
            multi_pdb_config: multi_pdb_specific,
            device: device,
        })
    }
    // Todo: refactor this to use loader.
    pub fn load_model(&self, model_type: ModelTypes) -> Result<ProteinMPNN, Error> {
        let default_dtype = DType::F32;
        match model_type {
            ModelTypes::ProteinMPNN => {
                // this is a hidden dep....
                // todo: use hf_hub
                let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
                let pth = PthTensors::new(mpnn_file, Some("model_state_dict"))?;
                let vb =
                    VarBuilder::from_backend(Box::new(pth), default_dtype, self.device.clone());
                let pconf = ProteinMPNNConfig::proteinmpnn();
                Ok(ProteinMPNN::load(vb, &pconf).expect("Unable to load the PMPNN Model"))
            }
            _ => panic!("not implented!"),
        }
    }
    pub fn generate_model(self) {
        todo!()
    }
    pub fn generate_protein_features(&self) -> Result<ProteinFeatures, Error> {
        let device = self.device.clone();
        let base_dtype = DType::F32;

        // init the Protein Features
        let (pdb, _) = pdbtbx::open(self.protein_inputs.clone()).expect("A PDB  or CIF file");
        let ac = AtomCollection::from(&pdb);

        // let s = ac.encode_amino_acids(&device)?;
        let s = ac
            .encode_amino_acids(&device)
            .expect("A complete convertion to locations");

        let x_37 = ac.to_numeric_atom37(&device)?;

        // Note: default to 1!
        let x_37_mask = Tensor::ones((x_37.dim(0)?, x_37.dim(1)?), base_dtype, &device)?;
        // println!("This is the atom map: {:?}", x_37_mask.dims());

        let (y, y_t, y_m) = ac.to_numeric_ligand_atoms(&device)?;

        // R_idx = np.array(CA_resnums, dtype=np.int32)
        let res_idx = ac.get_res_index();
        let res_idx_len = res_idx.len();
        let res_idx_tensor = Tensor::from_vec(res_idx, (1, res_idx_len), &device)?;

        // chain residues
        let chain_letters: Vec<String> = ac
            .iter_residues_aminoacid()
            .map(|res| res.chain_id)
            .collect();

        // unique Chains
        let chain_list: Vec<String> = chain_letters
            .clone()
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // assert_eq!(trul, false);

        // update residue info
        // residue_config: Option<ResidueControl>,
        // handle these:
        // pub fixed_residues: Option<String>,
        // pub redesigned_residues: Option<String>,
        // pub symmetry_residues: Option<String>,
        // pub symmetry_weights: Option<String>,
        // pub chains_to_design: Option<String>,
        // pub parse_these_chains_only: Option<String>,

        // update AA bias
        // handle these:
        // aa_bias: Option<AABiasConfig>,
        // pub bias_aa: Option<String>,
        // pub bias_aa_per_residue: Option<String>,
        // pub omit_aa: Option<String>,
        // pub omit_aa_per_residue: Option<String>,

        // update LigmpnnConfif
        // lig_mpnn_specific: Option<LigandMPNNConfig>,
        // handle these:
        // pub checkpoint_ligand_mpnn: Option<String>,
        // pub ligand_mpnn_use_atom_context: Option<i32>,
        // pub ligand_mpnn_use_side_chain_context: Option<i32>,
        // pub ligand_mpnn_cutoff_for_score: Option<String>,

        // update Membrane MPNN Config
        // membrane_mpnn_specific: Option<MembraneMPNNConfig>,
        // handle these:
        // pub global_transmembrane_label: Option<i32>,
        // pub transmembrane_buried: Option<String>,
        // pub transmembrane_interface: Option<String>,

        // update multipdb
        // multi_pdb_specific: Option<MultiPDBConfig>,
        // pub pdb_path_multi: Option<String>,
        // pub fixed_residues_multi: Option<String>,
        // pub redesigned_residues_multi: Option<String>,
        // pub omit_aa_per_residue_multi: Option<String>,
        // pub bias_aa_per_residue_multi: Option<String>,

        // println!("Returning Protein Features....");
        // return ligand MPNN.
        Ok(ProteinFeatures {
            s,                            // protein amino acids sequences as 1D Tensor of u32
            x: x_37,                      // protein co-oords by residue [1, 37, 4]
            x_mask: Some(x_37_mask),      // protein mask by residue
            y,                            // ligand coords
            y_t,                          // encoded ligand atom names
            y_m: Some(y_m),               // ligand mask
            r_idx: res_idx_tensor,        // protein residue indices shape=[length]
            chain_labels: None,           //  # protein chain letters shape=[length]
            chain_letters: chain_letters, // chain_letters: shape=[length]
            mask_c: None,                 // mask_c:  shape=[length]
            chain_list: chain_list,
        })
    }
}

#[derive(Debug, Clone, ValueEnum, Copy)]
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
            num_letters: 21,
            num_rbf: 16,
            scale_factor: 1.0,
            vocab: 21,
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
    pub model_type: Option<ModelTypes>,
    pub seed: Option<i32>,
    pub temperature: Option<f32>,
    pub verbose: Option<i32>,
    pub save_stats: Option<bool>,
    pub batch_size: Option<i32>,
    pub number_of_batches: Option<i32>,
    pub file_ending: Option<String>,
    pub zero_indexed: Option<i32>,
    pub homo_oligomer: Option<i32>,
    pub fasta_seq_separation: Option<String>,
}
