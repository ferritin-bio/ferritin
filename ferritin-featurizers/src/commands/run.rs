use crate::featurizer::LMPNNFeatures;
use candle_core;
use ferritin_core::AtomCollection;
use pdbtbx;

// First, create separate structs for different argument groups
#[derive(Debug)]
struct RunConfig {
    temperature: Option<f32>,
    verbose: Option<i32>,
    save_stats: Option<i32>,
    batch_size: Option<i32>,
    number_of_batches: Option<i32>,
    file_ending: Option<String>,
    zero_indexed: Option<i32>,
    homo_oligomer: Option<i32>,
    fasta_seq_separation: Option<String>,
    // ... other config fields
}

#[derive(Debug)]
struct ResidueControl {
    fixed_residues: Option<String>,
    redesigned_residues: Option<String>,
    symmetry_residues: Option<String>,
    symmetry_weights: Option<String>,
    chains_to_design: Option<String>,
    parse_these_chains_only: Option<String>,
}

#[derive(Debug)]
struct AABiasConfig {
    // // Amino Acid Biasing
    // #[arg(long)]
    // bias_AA: Option<String>,
    // #[arg(long)]
    // bias_AA_per_residue: Option<String>,
    // #[arg(long)]
    // omit_AA: Option<String>,
    // #[arg(long)]
    // omit_AA_per_residue: Option<String>,
}

struct MultiPDBConfig {
    // // Multi-PDB Related
    // #[arg(long)]
    // pdb_path_multi: Option<String>,
    // #[arg(long)]
    // fixed_residues_multi: Option<String>,
    // #[arg(long)]
    // redesigned_residues_multi: Option<String>,
    // #[arg(long)]
    // omit_AA_per_residue_multi: Option<String>,
    //       #[arg(long)]
    // bias_AA_per_residue_multi: Option<String>,
}

struct LigandMPNNConfig {
    // // LigandMPNN Specific
    // #[arg(long)]
    // checkpoint_ligand_mpnn: Option<String>,
    // #[arg(long)]
    // ligand_mpnn_use_atom_context: Option<i32>,
    // #[arg(long)]
    // ligand_mpnn_use_side_chain_context: Option<i32>,
    // #[arg(long)]
    // ligand_mpnn_cutoff_for_score: Option<String>,
}

struct MembraneMPNNConfig {
    // // Membrane MPNN Specific
    // #[arg(long)]
    // global_transmembrane_label: Option<i32>,
    // #[arg(long)]
    // transmembrane_buried: Option<String>,
    // #[arg(long)]
    // transmembrane_interface: Option<String>,
}

// todo: refactor this commands out to
// have lgincal, rlated grouping
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
