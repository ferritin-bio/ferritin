use crate::models::ligandmpnn::configs::{
    AABiasConfig, LigandMPNNConfig, MPNNExecConfig, MembraneMPNNConfig, ModelTypes, MultiPDBConfig,
    ResidueControl, RunConfig,
};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

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
    let device = device(false)?;

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
    println!("Model Loaded!");

    println!("Generating Protein Features");
    let prot_features = exec.generate_protein_features()?;
    println!("Protein Features Loaded!");

    // Create the output folders
    println!("Creating the Outputs");
    std::fs::create_dir_all(format!("{}/seqs", out_folder))?;
    std::fs::create_dir_all(format!("{}/backbones", out_folder))?;
    std::fs::create_dir_all(format!("{}/packed", out_folder))?;

    // Loading Dependent Factors
    let temperature = exec.run_config.temperature.unwrap_or(0.5);

    // if args.seed:
    //      seed = args.seed
    //  else:
    //      seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

    println!("Sampling from the Model...");
    println!("Temp and Seed are: temp: {:}, seed: {:}", temperature, seed);
    let model_sample = model.sample(&prot_features, temperature as f64, seed as u64)?;
    println!("{:?}", model_sample);

    std::fs::create_dir_all(format!("{}/seqs", out_folder))?;
    let sequences = model_sample.get_sequences()?;
    println!("OUTPUT FASTA: {:?}", sequences);
    println!("DECODING ORDER: {:?}", model_sample.get_decoding_order()?);

    let fasta_path = format!("{}/seqs/output.fasta", out_folder);

    let mut fasta_content = String::new();
    for (i, seq) in sequences.iter().enumerate() {
        fasta_content.push_str(&format!(">sequence_{}\n{}\n", i + 1, seq));
    }
    std::fs::write(fasta_path, fasta_content)?;

    // Score a Protein!
    println!("Scoring the Protein...");
    let model_score = model.score(&prot_features, false)?;
    println!("Protein Score: {:?}", model_score);
    // Sample from the Model!
    // Note: sampling from the model
    // println!("Sampling from the Model...");
    // let model_sample = model.sample(&prot_features);
    // println!("{:?}", model_sample);

    // assert_eq!(true, false);

    // prot_features
    // generate_protein_features()

    // model.score() -> Result<ScoreOutput>

    // Sample
    // model.sample() -> Result<ScoreOutput>

    Ok(())
}
