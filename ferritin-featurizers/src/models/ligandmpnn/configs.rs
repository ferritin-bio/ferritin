use clap::ValueEnum;

// #[derive(Clone, Debug)]
// pub enum PMPNNModelType {
//     LigandMPNN,
//     ProteinMPNN,
//     SolubleMPNN,
// }

#[derive(Debug, Clone, ValueEnum)] // Need Clone and ValueEnum for CLAP
pub enum ModelTypes {
    #[value(name = "protein_mpnn")] // Optional: customize CLI name
    ProteinMPNN,
    #[value(name = "ligand_mpnn")]
    LigandMPNN,
}

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

#[derive(Clone, Debug)]
pub struct ProteinMPNNConfig {
    atom_context_num: usize,
    augment_eps: f32,
    dropout_ratio: f32,
    edge_features: i64,
    hidden_dim: i64,
    k_neighbors: i64,
    ligand_mpnn_use_side_chain_context: bool,
    model_type: PMPNNModelType,
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
            model_type: PMPNNModelType::ProteinMPNN,
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
