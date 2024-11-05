use crate::cli::ModelTypes;
use crate::models::ligandmpnn::featurizer::LMPNNFeatures;
use crate::models::ligandmpnn::model::{ProteinMPNN, ProteinMPNNConfig, ScoreOutput};
use candle_core::{DType, Device};
use candle_nn::{Module, VarBuilder};
use ferritin_core::AtomCollection;
// use pdbtbx;

// First, create separate structs for different argument groups
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
/// Amino Acid Biasing
pub struct AABiasConfig {
    pub bias_AA: Option<String>,
    pub bias_AA_per_residue: Option<String>,
    pub omit_AA: Option<String>,
    pub omit_AA_per_residue: Option<String>,
}

/// Multi-PDB Related
pub struct MultiPDBConfig {
    pub pdb_path_multi: Option<String>,
    pub fixed_residues_multi: Option<String>,
    pub redesigned_residues_multi: Option<String>,
    pub omit_AA_per_residue_multi: Option<String>,
    pub bias_AA_per_residue_multi: Option<String>,
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

pub fn execute(
    seed: i32,
    pdb_path: String,
    out_folder: String,
    model_type: ModelTypes,
    runconfig: RunConfig,
    residue_control: ResidueControl,
    aa_bias: AABiasConfig,
    lig_mpnn_specific: LigandMPNNConfig,
    membrane_mpnn_specific: MembraneMPNNConfig,
    multi_pdb: MultiPDBConfig,
) -> anyhow::Result<()> {
    println!(
        "This run script is very crude at the moment and does not handle MOST of the CLI args....."
    );

    // get basic mocdel params
    let model_config = match model_type {
        ModelTypes::ProteinMPNN => ProteinMPNNConfig::proteinmpnn(),
        _ => panic!(),
    };

    // not sure this is the best wat to init the VarBuilder
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);

    // initialize the model
    let model = ProteinMPNN::new(model_config, vb);

    // Predict
    // model.predict()

    // Train
    // model.train()

    // Encode
    // model.encode() ->  Ok((h_v, h_e, e_idx)) // ??

    // Score
    // model.score() -> Result<ScoreOutput>

    // Sample
    // model.sample() -> Result<ScoreOutput>

    Ok(())
}
