//! Defining CLI interface.
//!
//! ** work in progress **
//!
use super::commands;
use crate::models::ligandmpnn::configs::{
    AABiasConfig, LigandMPNNConfig, MembraneMPNNConfig, ModelTypes, MultiPDBConfig, ResidueControl,
    RunConfig,
};
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
    Run {
        // Required Basic Arguments
        #[arg(long, required = true)]
        seed: i32,
        #[arg(long, required = true)]
        pdb_path: String,
        #[arg(long, required = true)]
        out_folder: String,
        #[arg(long, required = true, value_enum)]
        model_type: ModelTypes, // Use the enum type

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
        bias_aa: Option<String>,
        #[arg(long)]
        bias_aa_per_residue: Option<String>,
        #[arg(long)]
        omit_aa: Option<String>,
        #[arg(long)]
        omit_aa_per_residue: Option<String>,

        // Multi-PDB Related
        #[arg(long)]
        pdb_path_multi: Option<String>,
        #[arg(long)]
        fixed_residues_multi: Option<String>,
        #[arg(long)]
        redesigned_residues_multi: Option<String>,
        #[arg(long)]
        omit_aa_per_residue_multi: Option<String>,
        #[arg(long)]
        bias_aa_per_residue_multi: Option<String>,

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
    },
}

impl Cli {
    pub fn execute(self) -> anyhow::Result<()> {
        match self.command {
            Commands::Featurize { input, output } => commands::featurize::execute(input, output),
            Commands::Run {
                seed,
                pdb_path,
                out_folder,
                model_type,
                temperature,
                verbose,
                save_stats,
                batch_size,
                number_of_batches,
                file_ending,
                zero_indexed,
                homo_oligomer,
                fasta_seq_separation,
                fixed_residues,
                redesigned_residues,
                symmetry_residues,
                symmetry_weights,
                chains_to_design,
                parse_these_chains_only,
                bias_aa,
                bias_aa_per_residue,
                omit_aa,
                omit_aa_per_residue,
                pdb_path_multi,
                fixed_residues_multi,
                redesigned_residues_multi,
                omit_aa_per_residue_multi,
                bias_aa_per_residue_multi,
                checkpoint_ligand_mpnn,
                ligand_mpnn_use_atom_context,
                ligand_mpnn_use_side_chain_context,
                ligand_mpnn_cutoff_for_score,
                global_transmembrane_label,
                transmembrane_buried,
                transmembrane_interface,
            } => {
                let run_config = RunConfig {
                    temperature,
                    verbose,
                    save_stats,
                    batch_size,
                    number_of_batches,
                    file_ending,
                    zero_indexed,
                    homo_oligomer,
                    fasta_seq_separation,
                };

                let residue_control = ResidueControl {
                    fixed_residues,
                    redesigned_residues,
                    symmetry_residues,
                    symmetry_weights,
                    chains_to_design,
                    parse_these_chains_only,
                };

                let aa_bias = AABiasConfig {
                    bias_aa,
                    bias_aa_per_residue,
                    omit_aa,
                    omit_aa_per_residue,
                };

                let lig_mpnn_specific = LigandMPNNConfig {
                    checkpoint_ligand_mpnn,
                    ligand_mpnn_use_atom_context,
                    ligand_mpnn_use_side_chain_context,
                    ligand_mpnn_cutoff_for_score,
                };

                let membrane_mpnn_specific = MembraneMPNNConfig {
                    global_transmembrane_label,
                    transmembrane_buried,
                    transmembrane_interface,
                };

                let multi_pdb = MultiPDBConfig {
                    pdb_path_multi,
                    fixed_residues_multi,
                    redesigned_residues_multi,
                    omit_aa_per_residue_multi,
                    bias_aa_per_residue_multi,
                };

                let _ = commands::run::execute(
                    seed,
                    pdb_path,
                    out_folder,
                    model_type,
                    run_config,
                    residue_control,
                    aa_bias,
                    lig_mpnn_specific,
                    membrane_mpnn_specific,
                    multi_pdb,
                );

                Ok(())
            }
        }
    }
}
