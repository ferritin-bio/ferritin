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

    println!("About to Load the model!");
    let model = exec.load_model()?;
    println!("Model Loaded!");

    println!("Generating Protein Features");
    let prot_features = exec.generate_protein_features()?;
    println!("Protein Features Loaded!");

    // Create the output folders
    println!("Creating the Outputs");
    std::fs::create_dir_all(format!("{}/seqs", out_folder))?;
    std::fs::create_dir_all(format!("{}/backbones", out_folder))?;
    std::fs::create_dir_all(format!("{}/packed", out_folder))?;

    // Score a Protein!
    // println!("Scoring the Protein...");
    // let model_score = model.score(&prot_features, false);
    // println!("{:?}", model_score);

    // Sample from the Model!
    // Note: sampling from the model
    println!("Sampling from the Model...");
    let model_sample = model.sample(&prot_features);
    println!("{:?}", model_sample);

    // model.score() -> Result<ScoreOutput>

    // Sample
    // model.sample() -> Result<ScoreOutput>

    Ok(())
}
