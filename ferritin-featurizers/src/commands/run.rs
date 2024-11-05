use crate::models::ligandmpnn::configs::{
    AABiasConfig, LigandMPNNConfig, MembraneMPNNConfig, ModelTypes, MultiPDBConfig,
    ProteinMPNNConfig, ResidueControl, RunConfig,
};
use crate::models::ligandmpnn::featurizer::LMPNNFeatures;
use crate::models::ligandmpnn::model::{ProteinMPNN, ScoreOutput};
use candle_core::{DType, Device};
use candle_nn::{Module, VarBuilder};
use clap::{Parser, Subcommand, ValueEnum};
use ferritin_core::AtomCollection;

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

    // From File -> Protein Features
    let (pdb, _) = pdbtbx::open(pdb_path).expect("A PDB  or CIF file");
    let ac = AtomCollection::from(&pdb);
    let features = ac.featurize(&candle_core::Device::Cpu)?;

    // runconfig: RunConfig,
    // residue_control: ResidueControl,
    // aa_bias: AABiasConfig,
    // lig_mpnn_specific: LigandMPNNConfig,
    // membrane_mpnn_specific: MembraneMPNNConfig,
    // multi_pdb: MultiPDBConfig,

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
