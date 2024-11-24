use crate::models::ligandmpnn::configs::{
    AABiasConfig, LigandMPNNConfig, MPNNExecConfig, MembraneMPNNConfig, ModelTypes, MultiPDBConfig,
    ResidueControl, RunConfig,
};
use candle_core::Device;

pub fn execute(
    seed: i32,
    pdb_path: String,
    out_folder: String,
    model_type: Option<ModelTypes>,
    run_config: RunConfig,
    residue_control_config: ResidueControl,
    aa_bias_config: AABiasConfig,
    lig_mpnn_specific: LigandMPNNConfig,
    membrane_mpnn_specific: MembraneMPNNConfig,
    multi_pdb_config: MultiPDBConfig,
) -> anyhow::Result<()> {
    println!(
        "This run script is very crude at the moment and does not handle MOST of the CLI args....."
    );

    // todo - whats the best way to handle device?
    let device = &Device::Cpu;

    let model_type = model_type.unwrap_or(ModelTypes::ProteinMPNN);

    let exec = MPNNExecConfig::new(
        seed,
        device,
        pdb_path, // will need to omdify this for multiple
        model_type,
        run_config,
        Some(residue_control_config),
        Some(aa_bias_config),
        Some(lig_mpnn_specific),
        Some(membrane_mpnn_specific),
        Some(multi_pdb_config),
    )?;

    let model = exec.load_model()?;

    let prot_features = exec.protein_data;

    // Make the Ooutput Directories
    // if not os.path.exists(base_folder + "seqs"):
    //       os.makedirs(base_folder + "seqs", exist_ok=True)
    //   if not os.path.exists(base_folder + "backbones"):
    //       os.makedirs(base_folder + "backbones", exist_ok=True)
    //   if not os.path.exists(base_folder + "packed"):
    //       os.makedirs(base_folder + "packed", exist_ok=True)

    //
    // Run the Model!
    //

    // Encode the inputs:
    // for pdb in pdb_paths:
    //     if args.verbose:
    //         print("Designing protein from this path:", pdb)
    //     fixed_residues = fixed_residues_multi[pdb]
    //     redesigned_residues = redesigned_residues_multi[pdb]
    //     parse_all_atoms_flag = args.ligand_mpnn_use_side_chain_context or (
    //         args.pack_side_chains and not args.repack_everything
    //     )
    //     protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
    //         pdb,
    //         device=device,
    //         chains=parse_these_chains_only_list,
    //         parse_all_atoms=parse_all_atoms_flag,
    //         parse_atoms_with_zero_occupancy=args.parse_atoms_with_zero_occupancy,
    //     )

    // Score
    // model.score() -> Result<ScoreOutput>

    // Sample
    // model.sample() -> Result<ScoreOutput>

    Ok(())
}
