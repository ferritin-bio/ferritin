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
        pdb_path,
        model_type,
        run_config,
        Some(residue_control_config),
        Some(aa_bias_config),
        Some(lig_mpnn_specific),
        Some(membrane_mpnn_specific),
        Some(multi_pdb_config),
    )?;

    let model = exec.load_model()?;

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
