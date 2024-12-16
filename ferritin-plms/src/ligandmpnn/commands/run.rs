use crate::ligandmpnn::ligandmpnn::configs::{
    AABiasConfig, LigandMPNNConfig, MPNNExecConfig, MembraneMPNNConfig, ModelTypes, MultiPDBConfig,
    ResidueControl, RunConfig,
};
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result, Tensor};
use rand::Rng;

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
    pdb_path: String,
    out_folder: String,
    run_config: RunConfig,
    residue_control_config: ResidueControl,
    aa_bias_config: AABiasConfig,
    lig_mpnn_specific: LigandMPNNConfig,
    membrane_mpnn_specific: MembraneMPNNConfig,
    multi_pdb_config: MultiPDBConfig,
) -> anyhow::Result<()> {
    // todo - whats the best way to handle device?
    let device = device(false)?;

    let exec = MPNNExecConfig::new(
        device.clone(),
        pdb_path, // will need to omdify this for multiple
        run_config,
        Some(residue_control_config),
        Some(aa_bias_config),
        Some(lig_mpnn_specific),
        Some(membrane_mpnn_specific),
        Some(multi_pdb_config),
    )?;

    // Create Default Values ------------------------------------------------------------
    //
    let model_type = exec
        .run_config
        .model_type
        .unwrap_or(ModelTypes::ProteinMPNN);

    let seed = match exec.run_config.seed {
        Some(s) => s,
        None => {
            let mut rng = rand::thread_rng();
            rng.gen_range(0..99999) as i32
        }
    };

    let temperature = exec.run_config.temperature.unwrap_or(0.1);
    let save_stats = exec.run_config.save_stats.unwrap_or(false);

    // Load The model ------------------------------------------------------------

    let model = exec.load_model(model_type)?;
    let mut prot_features = exec.generate_protein_features()?;

    // Calculate Masks  ------------------------------------------------------------

    println!("Generating Chains to Design. Tensor of [B,L]");
    let chains_to_design: Vec<String> = match &exec.residue_control_config {
        None => prot_features.chain_letters.clone(),
        Some(config) => match &config.chains_to_design {
            None => prot_features.chain_letters.clone(),
            Some(chains) => chains.split(' ').map(String::from).collect(),
        },
    };

    // Chain tensor is the base. Additional Tensors can be added on top.
    let mut chain_mask_tensor = prot_features.get_chain_mask_tensor(chains_to_design, &device)?;

    // Residue-Related -------------------------------------------
    if let Some(res) = exec.residue_control_config {
        // Residues
        // let fixed_residues = res.fixed_residues;
        // let fixed_positions_tensor = prot_features.get_encoded_tensor(fixed_residues, &device)?;
        // // multiply the fixed positions to the chain tensor
        // chain_mask_tensor = chain_mask_tensor.mul(&fixed_positions_tensor)?;
    }

    // bias-Related -------------------------------------------
    // Todo
    // if let Some(aabias) = exec.aabias_config {
    //     let bias_tensor = &prot_features.create_bias_tensor(exec.aabias_config?.bias_aa).unwrap_or('');
    // let (batch_size, seq_length) = &prot_features.s.dims2()?;
    // let mut base_bias = Tensor::zeros_like(&prot_features.s)?;
    // println!("BIAS!! Dims for S {:?}", base_bias.dims());
    // // let bias_aa: Tensor = match aabias.bias_aa {
    // None =>
    // }

    // Update the Mask Here
    prot_features.update_mask(chain_mask_tensor)?;

    // Sample from the Model  -------------------------------------------
    println!("Sampling from the Model...");
    println!("Temp and Seed are: temp: {:}, seed: {:}", temperature, seed);
    let model_sample = model.sample(&prot_features, temperature as f64, seed as u64)?;
    println!("{:?}", model_sample);

    let _ = {
        // Create the output folders
        println!("Creating the Outputs");
        std::fs::create_dir_all(format!("{}/seqs", out_folder))?;
        std::fs::create_dir_all(format!("{}/backbones", out_folder))?;
        std::fs::create_dir_all(format!("{}/packed", out_folder))?;
        std::fs::create_dir_all(format!("{}/stats", out_folder))?;
        std::fs::create_dir_all(format!("{}/seqs", out_folder))?;
        let sequences = model_sample.get_sequences()?;
        let fasta_path = format!("{}/seqs/output.fasta", out_folder);
        let mut fasta_content = String::new();
        for (i, seq) in sequences.iter().enumerate() {
            fasta_content.push_str(&format!(">sequence_{}\n{}\n", i + 1, seq));
        }
        std::fs::write(fasta_path, fasta_content)?;
    };

    // note this is only the  Score outputs.
    // It doesn't have the other fields in the pytorch implmentation
    if save_stats {
        // out_dict = {}
        // out_dict["generated_sequences"] = S_stack.cpu()
        // out_dict["sampling_probs"] = sampling_probs_stack.cpu()
        // out_dict["log_probs"] = log_probs_stack.cpu()
        // out_dict["decoding_order"] = decoding_order_stack.cpu()
        // out_dict["native_sequence"] = feature_dict["S"][0].cpu()
        // out_dict["mask"] = feature_dict["mask"][0].cpu()
        // out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
        // out_dict["seed"] = seed
        // out_dict["temperature"] = args.temperature
        let outfile = format!("{}/stats/stats.safetensors", out_folder);
        model_sample.save_as_safetensors(outfile);
    }
    Ok(())
}
