//! Tests for ProteinMPNNRunner — model loading and inference via the public API.

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use ferritin_plms::ligandmpnn::configs::{MPNNExecConfig, ModelTypes, RunConfig};
    use ferritin_plms::ProteinMPNNRunner;
    use ferritin_test_data::TestFile;

    fn run_config() -> RunConfig {
        RunConfig {
            model_type: Some(ModelTypes::ProteinMPNN),
            seed: None,
            temperature: None,
            verbose: None,
            save_stats: None,
            batch_size: None,
            number_of_batches: None,
            file_ending: None,
            zero_indexed: None,
            homo_oligomer: None,
            fasta_seq_separation: None,
        }
    }

    #[test]
    fn test_pmpnn_runner_from_path_loads() {
        let (weights_path, _handle) = TestFile::ligmpnn_pmpnn_01()
            .create_temp()
            .expect("Failed to extract test weights");
        let runner = ProteinMPNNRunner::from_path(&weights_path, Device::Cpu);
        assert!(
            runner.is_ok(),
            "ProteinMPNNRunner::from_path should succeed with embedded test weights: {:?}",
            runner.err()
        );
    }

    #[test]
    fn test_pmpnn_runner_get_pseudo_probabilities() {
        let (weights_path, _handle) = TestFile::ligmpnn_pmpnn_01()
            .create_temp()
            .expect("Failed to extract test weights");
        let runner = ProteinMPNNRunner::from_path(&weights_path, Device::Cpu)
            .expect("Failed to load ProteinMPNNRunner");

        let (pdb_path, _pdb_handle) = TestFile::protein_02()
            .create_temp()
            .expect("Failed to extract test PDB");

        let exec_config = MPNNExecConfig::new(
            Device::Cpu,
            pdb_path,
            run_config(),
            None,
            None,
            None,
            None,
            None,
        )
        .expect("Failed to create MPNNExecConfig");

        let features = exec_config
            .generate_protein_features()
            .expect("Failed to generate protein features");

        let probs = runner
            .get_pseudo_probabilities(&features)
            .expect("get_pseudo_probabilities should succeed");

        assert!(
            !probs.is_empty(),
            "Expected non-empty pseudo-probability results"
        );
        for pp in &probs {
            assert!(pp.pseudo_prob >= 0.0 && pp.pseudo_prob <= 1.0);
        }
    }
}
