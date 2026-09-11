//! Tests for ProteinMPNNRunner — model loading and inference via the public API.

#[path = "support/mod.rs"]
mod support;

#[cfg(test)]
mod tests {
    use super::support::parity::{
        ParityFixture, assert_distribution_close, assert_embeddings_close,
    };
    use candle_core::Device;
    use ferritin_plms::ProteinMPNNRunner;
    use ferritin_plms::ligandmpnn::configs::{
        MPNNExecConfig, ModelTypes, ProteinMPNNConfig, RunConfig,
    };
    use ferritin_plms::loader::LoadOptions;
    use ferritin_test_data::TestFile;

    /// Both MPNN ports agree with the reference to KL < 1e-6 and fail at
    /// 1e-7, i.e. at f32 round-off. The committed threshold keeps one decade
    /// of headroom so platform-level float nondeterminism cannot flake it —
    /// still four decades tighter than the 0.01 the other ports use.
    const KL_THRESHOLD: f64 = 1e-5;
    /// Encoder embeddings are compared by cosine rather than absolute
    /// difference: they are an intermediate with no fixed scale, and the
    /// distribution assertion is what pins the actual output. Measured
    /// agreement exceeds 0.999999.
    const H_V_COSINE_MIN: f32 = 0.99999;

    fn run_config() -> RunConfig {
        run_config_for(ModelTypes::ProteinMPNN)
    }

    fn run_config_for(model_type: ModelTypes) -> RunConfig {
        RunConfig {
            model_type: Some(model_type),
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

    /// Numerical parity: ProteinMPNN in Rust vs the LigandMPNN Python reference.
    ///
    /// Fixture: `scripts/generate_mpnn_fixtures.py`. It pins the structure-only
    /// forward pass — the decoder over plain encoder embeddings, no sequence
    /// context — because `score()`'s decoding order is drawn from `randn` and
    /// so is not reproducible. That is what `simple_decode` computes here.
    ///
    /// This test ran vacuously for a long time: it skipped on a missing fixture
    /// while the port it was meant to check disagreed with the reference at
    /// every position (2/93 argmax agreement). See ferritin-100.11 for the six
    /// defects that fixed.
    #[test]
    fn test_pmpnn_parity_vs_python_reference() {
        let device = Device::Cpu;

        // Python-reference log-probs, shape (L, 21). ProteinMPNN has no special
        // tokens, so the reference rows align 1:1 with residues.
        let Some(fixture) = ParityFixture::load_or_skip("proteinmpnn_parity", &device)
            .expect("ProteinMPNN parity fixture lookup failed")
        else {
            return;
        };
        let ref_log_probs = fixture.tensor("log_probs").expect("missing 'log_probs'");
        let (_, vocab_size) = ref_log_probs.dims2().expect("expected 2D tensor");
        assert_eq!(vocab_size, 21, "ProteinMPNN vocab is 21 amino acids");
        let ref_probs = ref_log_probs.exp().expect("exp failed");

        // Run Rust model on 1BC8.pdb (same structure used for fixture generation)
        let (weights_path, _weights_handle) = TestFile::ligmpnn_pmpnn_01()
            .create_temp()
            .expect("Failed to extract test weights");
        let runner = ProteinMPNNRunner::from_path(&weights_path, device.clone())
            .expect("Failed to load ProteinMPNNRunner");

        let (pdb_path, _pdb_handle) = TestFile::protein_02()
            .create_temp()
            .expect("Failed to extract 1BC8.pdb");

        let exec_config = MPNNExecConfig::new(
            device.clone(),
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

        // get_log_probs returns (L, 21) log-probabilities
        let rust_probs = runner
            .get_log_probs(&features)
            .expect("get_log_probs failed")
            .exp()
            .expect("exp failed");

        // ProteinMPNN has no special tokens; reference and Rust rows align 1:1.
        assert_distribution_close(&rust_probs, &ref_probs, KL_THRESHOLD)
            .expect("ProteinMPNN distribution parity");
    }

    /// Numerical parity: LigandMPNN in Rust vs the LigandMPNN Python reference.
    ///
    /// 1BC8 is a zinc-finger/DNA complex, so its ligand set is a real one —
    /// 406 atoms of DNA plus two zincs — and the 25-atom per-residue context
    /// is genuinely populated rather than all padding.
    ///
    /// Checks the encoder node embeddings as well as the final distribution:
    /// the ligand context path feeds back into `h_V` and nothing else, so a
    /// break there is invisible in `E_idx` and obvious in `enc_h_V`.
    #[test]
    fn test_ligandmpnn_parity_vs_python_reference() {
        let device = Device::Cpu;

        let Some(fixture) = ParityFixture::load_or_skip("ligandmpnn_parity", &device)
            .expect("LigandMPNN parity fixture lookup failed")
        else {
            return;
        };
        let ref_log_probs = fixture.tensor("log_probs").expect("missing 'log_probs'");
        let ref_h_v = fixture.tensor("enc_h_V").expect("missing 'enc_h_V'");
        let ref_e_idx = fixture.tensor("E_idx").expect("missing 'E_idx'");

        let (weights_path, _weights_handle) = TestFile::ligmpnn_lmpnn_01()
            .create_temp()
            .expect("Failed to extract LigandMPNN test weights");
        let runner = ProteinMPNNRunner::from_path_as(
            &weights_path,
            &ProteinMPNNConfig::ligandmpnn(),
            &LoadOptions::new(device.clone()),
        )
        .expect("Failed to load LigandMPNN");

        let (pdb_path, _pdb_handle) = TestFile::protein_02()
            .create_temp()
            .expect("Failed to extract 1BC8.pdb");
        let exec_config = MPNNExecConfig::new(
            device.clone(),
            pdb_path,
            run_config_for(ModelTypes::LigandMPNN),
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

        // Stage 1: the protein neighbour graph. Exact integers — any
        // disagreement means the featurizer, not the numerics.
        let model = runner.into_model();
        let (h_v, _h_e, e_idx) = model.encode(&features).expect("encode failed");
        let rust_idx: Vec<u32> = e_idx
            .squeeze(0)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .expect("E_idx");
        let ref_idx: Vec<u32> = ref_e_idx
            .to_dtype(candle_core::DType::U32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .expect("reference E_idx");
        assert_eq!(
            rust_idx, ref_idx,
            "neighbour graph differs from the reference"
        );

        // Stage 2: encoder node embeddings, after the ligand context path.
        let h_v = h_v.squeeze(0).expect("squeeze batch");
        assert_embeddings_close(&h_v, ref_h_v, H_V_COSINE_MIN)
            .expect("LigandMPNN encoder embedding parity");

        // Stage 3: the output distribution.
        let rust_probs = model
            .simple_decode(&features)
            .expect("simple_decode failed")
            .get_log_probs()
            .squeeze(0)
            .and_then(|t| t.exp())
            .expect("log_probs");
        assert_distribution_close(
            &rust_probs,
            &ref_log_probs.exp().expect("exp"),
            KL_THRESHOLD,
        )
        .expect("LigandMPNN distribution parity");
    }
}
