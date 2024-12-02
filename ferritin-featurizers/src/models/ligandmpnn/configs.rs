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
use crate::models::ligandmpnn::model::ScoreOutput;
use crate::models::ligandmpnn::utilities::aa1to_int;
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
    pub(crate) model_type: ModelTypes,
    pub(crate) aabias_config: Option<AABiasConfig>,
    pub(crate) ligand_mpnn_config: Option<LigandMPNNConfig>,
    pub(crate) membrane_mpnn_config: Option<MembraneMPNNConfig>,
    pub(crate) multi_pdb_config: Option<MultiPDBConfig>,
    pub(crate) residue_control_config: Option<ResidueControl>,
    pub(crate) device: Device,
    pub(crate) seed: i32,
}

impl MPNNExecConfig {
    pub fn new(
        seed: i32,
        device: Device,
        pdb_path: String,
        model_type: ModelTypes,
        run_config: RunConfig,
        residue_config: Option<ResidueControl>,
        aa_bias: Option<AABiasConfig>,
        lig_mpnn_specific: Option<LigandMPNNConfig>,
        membrane_mpnn_specific: Option<MembraneMPNNConfig>,
        multi_pdb_specific: Option<MultiPDBConfig>,
    ) -> Result<Self, Error> {
        Ok(MPNNExecConfig {
            protein_inputs: pdb_path,
            model_type: model_type,
            run_config,
            aabias_config: aa_bias,
            ligand_mpnn_config: lig_mpnn_specific,
            membrane_mpnn_config: membrane_mpnn_specific,
            residue_control_config: residue_config,
            multi_pdb_config: multi_pdb_specific,
            seed,
            device: device,
        })
    }
    // Todo: refactor this to use loader.
    pub fn load_model(&self) -> Result<ProteinMPNN, Error> {
        let default_dtype = DType::F32;

        // this is a hidden dep....
        // todo: use hf_hub
        let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
        let pth = PthTensors::new(mpnn_file, Some("model_state_dict"))?;
        let vb = VarBuilder::from_backend(Box::new(pth), default_dtype, self.device.clone());
        let pconf = ProteinMPNNConfig::proteinmpnn();
        Ok(ProteinMPNN::load(vb, &pconf).expect("Unable to load the PMPNN Model"))
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
            s,                           // protein amino acids sequences as 1D Tensor of u32
            x: x_37,                     // protein co-oords by residue [1, 37, 4]
            x_mask: Some(x_37_mask),     // protein mask by residue
            y,                           // ligand coords
            y_t,                         // encoded ligand atom names
            y_m: Some(y_m),              // ligand mask
            r_idx: Some(res_idx_tensor), // protein residue indices shape=[length]
            chain_labels: None,          //  # protein chain letters shape=[length]
            chain_letters: None,         // chain_letters: shape=[length]
            mask_c: None,                // mask_c:  shape=[length]
            chain_list: None,
        })
    }

    pub fn create_fasta_string(&self, score: ScoreOutput) -> Result<String, Error> {
        // https://github.com/dauparas/LigandMPNN/blob/main/run.py#L543C1-L558C18
        //
        // with open(output_fasta, "w") as f:
        //     f.write(
        //         ">{}, T={}, seed={}, num_res={}, num_ligand_res={}, use_ligand_context={}, ligand_cutoff_distance={}, batch_size={}, number_of_batches={}, model_path={}\n{}\n".format(
        //             name,
        //             args.temperature,
        //             seed,
        //             torch.sum(rec_mask).cpu().numpy(),
        //             torch.sum(combined_mask[:1]).cpu().numpy(),
        //             bool(args.ligand_mpnn_use_atom_context),
        //             float(args.ligand_mpnn_cutoff_for_score),
        //             args.batch_size,
        //             args.number_of_batches,
        //             checkpoint_path,
        //             seq_out_str,
        //         )
        //     )
        // >1BC8, T=0.1, seed=111, num_res=93, num_ligand_res=93, use_ligand_context=True, ligand_cutoff_distance=8.0, batch_size=1, number_of_batches=1, model_path=./model_params/proteinmpnn_v_48_020.pt
        // MDSAITLWQFLLQLLQKPQNKHMICWTSNDGQFKLLQAEEVARLWGIRKNKPNMNYDKLSRALRYYYVKNIIKKVNGQKFVYKFVSYPEILNM
        // >1BC8, id=1, T=0.1, seed=111, overall_confidence=0.3987, ligand_confidence=0.3987, seq_rec=0.5161
        // GTSSISLHEFLLKLLSDPAYKDIIEWTSDDGEFKLKKPEAVAKLWGEEKGEPDMNYKKMEKELKKYEKKKIIEKVKGKPNHYKFVNYPEILFP

        println!("TEMP: {:?}", self.run_config.temperature);
        // probably should move `seed` to Runconfig
        println!("Seed: {}", self.seed);
        // println!("REC_MASK: {}", self.seed);
        // println!("COMBINED_Mask: {}", self.seed);
        // println!("ligand_mpnn_use_atom_context: {}", self.seed);
        // println!("ligand_mpnn_cutoff_for_score: {}", self.seed);
        println!("batch_size: {:?}", self.run_config.batch_size);
        println!("batch_size: {:?}", self.run_config.number_of_batches);
        // println!("checkpoint_path: {}", self.run_config.ch );
        // println!("seq_out_str: {}", self.run_config.ch );

        // name = pdb[pdb.rfind("/") + 1 :] // L366

        // First Record
        //
        // https://github.com/dauparas/LigandMPNN/blob/main/run.py#L464C1-L470C1
        // native_seq = "".join(
        //     [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()])
        // seq_np = np.array(list(native_seq))
        // seq_out_str = []
        // for mask in protein_dict["mask_c"]:
        //     seq_out_str += list(seq_np[mask.cpu().numpy()])
        //     seq_out_str += [args.fasta_seq_separation]
        // seq_out_str = "".join(seq_out_str)[:-1]

        // Subsequent Records
        //
        // https://github.com/dauparas/LigandMPNN/blob/main/run.py#L641
        // seq = "".join(
        //     [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
        // )
        // seq_np = np.array(list(seq))
        // seq_out_str = []
        // for mask in protein_dict["mask_c"]:
        //     seq_out_str += list(seq_np[mask.cpu().numpy()])
        //     seq_out_str += [args.fasta_seq_separation]
        // seq_out_str = "".join(seq_out_str)[:-1]
        //
        // if ix == S_stack.shape[0] - 1:
        //     # final 2 lines
        //     f.write(
        //         ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}".format(
        //             name,
        //             ix_suffix,
        //             args.temperature,
        //             seed,
        //             loss_np,
        //             loss_XY_np,
        //             seq_rec_print,
        //             seq_out_str,
        //         )
        //     )
        // else:
        //     f.write(
        //         ">{}, id={}, T={}, seed={}, overall_confidence={}, ligand_confidence={}, seq_rec={}\n{}\n".format(
        //             name,
        //             ix_suffix,
        //             args.temperature,
        //             seed,
        //             loss_np,
        //             loss_XY_np,
        //             seq_rec_print,
        //             seq_out_str,
        //         )
        //     )

        println!("In the Config....");
        println!("S: {:?}", score.s);
        println!("log_probs: {:?}", score.log_probs);
        println!("logits: {:?}", score.logits);
        println!("decoding_order: {:?}", score.decoding_order);

        // seq = "".join(
        //     [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
        // )

        // todo!();
        Ok("FASTA_STRING".to_string())

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
            num_letters: 21, // whats the difference between the num_letters and the vocab?
            num_rbf: 16,
            scale_factor: 30.0,
            // vocab: 48,
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
