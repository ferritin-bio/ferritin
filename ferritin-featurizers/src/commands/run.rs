use crate::featurizer::LMPNNFeatures;
use candle_core;
use ferritin_core::AtomCollection;
use pdbtbx;

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
    model_type: String,

    temperature: Option<f32>,
    verbose: Option<i32>,
    save_stats: Option<i32>,
    batch_size: Option<i32>,
    number_of_batches: Option<i32>,
    file_ending: Option<String>,
    zero_indexed: Option<i32>,
    homo_oligomer: Option<i32>,
    fasta_seq_separation: Option<String>,
    fixed_residues: Option<String>,
    redesigned_residues: Option<String>,
    symmetry_residues: Option<String>,
    symmetry_weights: Option<String>,
    chains_to_design: Option<String>,
    parse_these_chains_only: Option<String>,
    bias_AA: Option<String>,
    bias_AA_per_residue: Option<String>,
    omit_AA: Option<String>,
    omit_AA_per_residue: Option<String>,
    pdb_path_multi: Option<String>,
    fixed_residues_multi: Option<String>,
    redesigned_residues_multi: Option<String>,
    omit_AA_per_residue_multi: Option<String>,
    bias_AA_per_residue_multi: Option<String>,
    checkpoint_ligand_mpnn: Option<String>,
    ligand_mpnn_use_atom_context: Option<i32>,
    ligand_mpnn_use_side_chain_context: Option<i32>,
    ligand_mpnn_cutoff_for_score: Option<String>,
    global_transmembrane_label: Option<i32>,
    transmembrane_buried: Option<String>,
    transmembrane_interface: Option<String>,
) -> anyhow::Result<()> {
    // seed: i32,
    // pdb_path: String,
    // out_folder: String,
    // model_type: String,

    println!("Need to implement the seed!");
    let (pdb, _) = pdbtbx::open(input).expect("A PDB or CIF file");
    let ac = AtomCollection::from(&pdb);
    let features = ac.featurize(&candle_core::Device::Cpu)?;
    let _ = features.save_to_safetensor(&output)?;
    Ok(())
}
