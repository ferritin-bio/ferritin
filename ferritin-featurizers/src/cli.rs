use super::commands;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Featurize {
        #[arg(short, long)]
        input: String,
        #[arg(short, long)]
        output: String,
    },
    #[derive(Parser, Debug)]
    #[command(author, version, about, long_about = None)]
    struct Run {
        // Required Basic Arguments
        #[arg(long, required = true)]
        seed: i32,

        #[arg(long, required = true)]
        pdb_path: String,

        #[arg(long, required = true)]
        out_folder: String,

        #[arg(long, required = true)]
        model_type: String,

        // Configuration Arguments
        #[arg(long)]
        temperature: Option<f32>,

        #[arg(long)]
        verbose: Option<i32>,

        #[arg(long)]
        save_stats: Option<i32>,

        #[arg(long)]
        batch_size: Option<i32>,

        #[arg(long)]
        number_of_batches: Option<i32>,

        #[arg(long)]
        file_ending: Option<String>,

        #[arg(long)]
        zero_indexed: Option<i32>,

        #[arg(long)]
        homo_oligomer: Option<i32>,

        #[arg(long)]
        fasta_seq_separation: Option<String>,

        // Residue Control
        #[arg(long)]
        fixed_residues: Option<String>,

        #[arg(long)]
        redesigned_residues: Option<String>,

        #[arg(long)]
        symmetry_residues: Option<String>,

        #[arg(long)]
        symmetry_weights: Option<String>,

        #[arg(long)]
        chains_to_design: Option<String>,

        #[arg(long)]
        parse_these_chains_only: Option<String>,

        // Amino Acid Biasing
        #[arg(long)]
        bias_AA: Option<String>,

        #[arg(long)]
        bias_AA_per_residue: Option<String>,

        #[arg(long)]
        omit_AA: Option<String>,

        #[arg(long)]
        omit_AA_per_residue: Option<String>,

        // Multi-PDB Related
        #[arg(long)]
        pdb_path_multi: Option<String>,

        #[arg(long)]
        fixed_residues_multi: Option<String>,

        #[arg(long)]
        redesigned_residues_multi: Option<String>,

        #[arg(long)]
        omit_AA_per_residue_multi: Option<String>,

        #[arg(long)]
        bias_AA_per_residue_multi: Option<String>,

        // LigandMPNN Specific
        #[arg(long)]
        checkpoint_ligand_mpnn: Option<String>,

        #[arg(long)]
        ligand_mpnn_use_atom_context: Option<i32>,

        #[arg(long)]
        ligand_mpnn_use_side_chain_context: Option<i32>,

        #[arg(long)]
        ligand_mpnn_cutoff_for_score: Option<String>,

        // Membrane MPNN Specific
        #[arg(long)]
        global_transmembrane_label: Option<i32>,

        #[arg(long)]
        transmembrane_buried: Option<String>,

        #[arg(long)]
        transmembrane_interface: Option<String>,
    }
}

impl Cli {
    pub fn execute(self) -> anyhow::Result<()> {
        match self.command {
            Commands::Featurize { input, output } => commands::featurize::execute(input, output),
        }
    }
}
