//! Integration smoke tests for ESMC.
//!
//! Tests that require downloading model weights are marked `#[ignore]` so
//! they do not run in CI by default. Run them explicitly with:
//!
//! ```shell
//! cargo test -p ferritin-plms test_esmc -- --ignored --nocapture
//! cargo test -p ferritin-plms test_esmc -- --ignored --nocapture --features metal
//! ```

mod support;

use anyhow::Result;
use candle_core::Device;
use ferritin_plms::{
    ESMCConfig,
    esmc::models::esmc::{ESMC, LogitsConfig},
};

/// Cosine-similarity floor for ESMC-300M per-residue embedding parity.
/// A 30-layer transformer accumulates F32 rounding, so this is looser than the
/// 1e-3 logit tolerance used for the shallower ESM2/AMPLIFY ports.
const ESMC_EMBED_COSINE_FLOOR: f32 = 0.999;

/// Reference sequence for the ESMC embedding parity fixture.
const ESMC_PARITY_SEQ: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

/// GFP — used as the primary smoke-test sequence (matches the biohub model card).
const GFP: &str = concat!(
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCF",
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE",
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE",
    "KRDHMVLLEFVTAAGITHGMDELYK"
);

// ---------------------------------------------------------------------------
// Unit tests — no model download needed
// ---------------------------------------------------------------------------

/// ESMCConfig should instantiate all three model sizes without panicking.
#[test]
fn test_esmc_configs_instantiate() {
    let c300 = ESMCConfig::esmc_300m();
    assert_eq!(c300.d_model, 960);
    assert_eq!(c300.n_heads, 15);
    assert_eq!(c300.n_layers, 30);
    assert_eq!(c300.n_layers_geom, 0);

    let c600 = ESMCConfig::esmc_600m();
    assert_eq!(c600.d_model, 1152);
    assert_eq!(c600.n_heads, 18);
    assert_eq!(c600.n_layers, 36);
    assert_eq!(c600.n_layers_geom, 0);

    let c6b = ESMCConfig::esmc_6b();
    assert_eq!(c6b.d_model, 2560);
    assert_eq!(c6b.n_heads, 40);
    assert_eq!(c6b.n_layers, 80);
    assert_eq!(c6b.n_layers_geom, 0);
}

/// residue_scaling_factor = sqrt(n_layers / 36) for all sizes.
#[test]
fn test_esmc_residue_scaling_factors() {
    let c300 = ESMCConfig::esmc_300m();
    let expected_300 = (30f64 / 36.).sqrt();
    assert!(
        (c300.residue_scaling_factor - expected_300).abs() < 1e-10,
        "300M scaling factor mismatch: {} vs {}",
        c300.residue_scaling_factor,
        expected_300
    );

    let c600 = ESMCConfig::esmc_600m();
    assert!(
        (c600.residue_scaling_factor - 1.0).abs() < 1e-10,
        "600M scaling factor should be 1.0, got {}",
        c600.residue_scaling_factor
    );

    let c6b = ESMCConfig::esmc_6b();
    let expected_6b = (80f64 / 36.).sqrt();
    assert!(
        (c6b.residue_scaling_factor - expected_6b).abs() < 1e-10,
        "6B scaling factor mismatch: {} vs {}",
        c6b.residue_scaling_factor,
        expected_6b
    );
}

/// Tokenization round-trip: encode a sequence then decode back; result should
/// match the input (stripping BOS/EOS).
#[test]
fn test_esmc_tokenize_roundtrip() -> Result<()> {
    let device = Device::Cpu;
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
    let config = ESMCConfig::esmc_300m();
    let model = ESMC::load(vb, config)?;

    let seq = "ACDEFGHIKLMNPQRSTVWY";
    let tokens = model.encode(seq)?;
    // Shape should be (L+2,) — 20 residues + BOS + EOS
    assert_eq!(tokens.dims1()?, seq.len() + 2);

    let decoded = model.decode(&tokens)?;
    assert_eq!(decoded, seq, "Round-trip mismatch: got {decoded:?}");
    Ok(())
}

/// Zero-weight forward pass: logits shape should be (1, L+2, 64).
#[test]
fn test_esmc_forward_shape_zeroed_weights() -> Result<()> {
    let device = Device::Cpu;
    let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
    let config = ESMCConfig::esmc_300m();
    let model = ESMC::load(vb, config)?;

    let seq = "ACDEFGHIK"; // 9 residues → L+2 = 11 tokens
    let tokens = model.encode(seq)?;
    let tokens_batched = tokens.unsqueeze(0)?; // (1, L+2)

    let output = model.logits(
        &tokens_batched,
        LogitsConfig {
            sequence: true,
            return_embeddings: true,
            return_hidden_states: false,
        },
    )?;

    let expected_len = seq.len() + 2; // 11

    let seq_logits = output
        .sequence_logits
        .expect("sequence logits should be present");
    assert_eq!(
        seq_logits.dims(),
        &[1, expected_len, 64],
        "logits shape mismatch"
    );

    let embeddings = output.embeddings.expect("embeddings should be present");
    assert_eq!(
        embeddings.dims(),
        &[1, expected_len, 960],
        "embedding shape mismatch (d_model=960 for 300M)"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Integration tests — require downloading model weights (~1.3 GB)
// ---------------------------------------------------------------------------

/// End-to-end smoke: load ESMC-300M from HuggingFace and embed GFP.
/// Validates embedding shape `(1, L+2, 960)`.
#[test]
#[ignore = "requires downloading biohub/ESMC-300M weights (~1.3 GB)"]
fn test_esmc_300m_embed_gfp() -> Result<()> {
    use ferritin_plms::{ESMCModels, ESMCRunner};

    let device = Device::Cpu;
    let runner = ESMCRunner::from_pretrained(ESMCModels::ESMC300M, device)?;
    let embeddings = runner.embed_sequence(GFP)?;

    let expected_len = GFP.len() + 2; // BOS + sequence + EOS
    assert_eq!(
        embeddings.dims(),
        &[1, expected_len, 960],
        "ESMC-300M embedding shape mismatch"
    );
    Ok(())
}

/// End-to-end smoke: ESMC-300M logits shape should be (1, L+2, 64).
#[test]
#[ignore = "requires downloading biohub/ESMC-300M weights (~1.3 GB)"]
fn test_esmc_300m_logits_shape() -> Result<()> {
    use ferritin_plms::esmc::pretrained::ESMCModels as Variants;
    use ferritin_plms::loader::{LoadOptions, optional_prefix};

    let device = Device::Cpu;
    let (source, filename, config) = Variants::ESMC300M.model_info()?;
    let vb = source.var_builder(filename, &LoadOptions::new(device.clone()))?;
    let vb_root = optional_prefix(vb, "esmc", "embed.weight");
    let model = ESMC::load(vb_root, config)?;

    let short_seq = "MQIFVKTLTGKTITLEVEP"; // 19 residues → 21 tokens
    let tokens = model.encode(short_seq)?.unsqueeze(0)?;
    let output = model.forward(&tokens, None, false)?;

    let expected_len = short_seq.len() + 2;
    assert_eq!(
        output.sequence_logits.dims(),
        &[1, expected_len, 64],
        "ESMC-300M logits shape wrong"
    );
    Ok(())
}

/// Numerical parity: ESMC-300M Rust per-residue embeddings vs the Python
/// reference. The reference `(L+2, 960)` tensor **keeps** the BOS/EOS rows, so
/// the Rust `(1, L+2, 960)` output aligns 1:1 after squeezing the batch dim
/// (`SpecialTokens::NONE`) — a mismatch in BOS/EOS placement therefore shows up
/// as a failure at position 0 or L+1 rather than passing silently.
///
/// Requires:
///   1. `biohub/ESMC-300M` weights (~1.3 GB, cached by hf_hub on first run)
///   2. Fixture `tests/fixtures/esmc_parity.safetensors` from
///      `scripts/generate_esmc_fixtures.py`
///
/// Run: `cargo test -p ferritin-plms test_esmc_parity -- --ignored --nocapture`
#[test]
#[ignore = "requires HF model download and fixture: run scripts/generate_esmc_fixtures.py"]
fn test_esmc_parity_vs_python_reference() -> Result<()> {
    use ferritin_plms::{ESMCModels, ESMCRunner};
    use support::parity::{ParityFixture, SpecialTokens, align_rows, assert_embeddings_close};

    let device = Device::Cpu;
    // Skips (loudly) while no esmc_parity fixture is committed — see
    // PARITY_COVERAGE and ferritin-100.20.
    let Some(fixture) = ParityFixture::load_or_skip("esmc_parity", &device)? else {
        return Ok(());
    };
    let ref_embeddings = fixture.tensor("embeddings")?;

    let runner = ESMCRunner::from_pretrained(ESMCModels::ESMC300M, device)?;
    let rust = runner.embed_sequence(ESMC_PARITY_SEQ)?; // (1, L+2, 960)
    let rust_embeddings = align_rows(&rust, SpecialTokens::NONE)?;

    assert_embeddings_close(&rust_embeddings, ref_embeddings, ESMC_EMBED_COSINE_FLOOR)?;
    println!("ESMC-300M embedding parity OK (cosine floor {ESMC_EMBED_COSINE_FLOOR})");
    Ok(())
}

/// 300M and 600M point at the EvolutionaryScale originals, not the fused
/// `biohub/ESMC-*` re-exports whose layout nothing in this port could resolve
/// (ferritin-100.23).
#[test]
fn test_esmc_model_info_targets_evolutionaryscale() {
    use ferritin_plms::esmc::pretrained::ESMCModels;

    for (variant, repo, file) in [
        (
            ESMCModels::ESMC300M,
            "EvolutionaryScale/esmc-300m-2024-12",
            "data/weights/esmc_300m_2024_12_v0.pth",
        ),
        (
            ESMCModels::ESMC600M,
            "EvolutionaryScale/esmc-600m-2024-12",
            "data/weights/esmc_600m_2024_12_v0.pth",
        ),
        (
            ESMCModels::ESMC6B,
            "EvolutionaryScale/esmc-6b-2024-12",
            "model.safetensors.index.json",
        ),
    ] {
        let (source, filename, _config) =
            variant.model_info().expect("variant should be supported");
        assert_eq!(source.repo_id, repo);
        assert_eq!(filename, file);
    }
}

/// Every tensor path the ESMC loader will request exists in the real ESMC-6B
/// shard index (ferritin-100.24).
///
/// The 6B model itself cannot be loaded on a modest machine — ~24 GB at F32,
/// ~12 GB at F16 — so this checks the part that can be checked cheaply: that
/// the names the loader asks for are the names the checkpoint has. It fetches
/// only `model.safetensors.index.json` (~60 KB), not the 12 GB of shards.
///
/// Catches the two things that made 6B special: the backbone nests under
/// `esmc.` while the head sits at the top level as `lm_head`, and the weights
/// are split across six shards.
#[test]
#[ignore = "fetches the ESMC-6B shard index (~60 KB, no weights)"]
fn test_esmc_6b_index_contains_every_path_the_loader_requests() -> Result<()> {
    use ferritin_plms::esmc::pretrained::ESMCModels;
    use std::collections::BTreeMap;

    let (source, index_file, config) = ESMCModels::ESMC6B.model_info()?;
    assert!(
        index_file.ends_with(".index.json"),
        "6B should be loaded through its shard index; got {index_file}"
    );

    let path = source.fetch(index_file)?;
    let raw = std::fs::read_to_string(&path)?;
    let index: serde_json::Value = serde_json::from_str(&raw)?;
    let weight_map: BTreeMap<String, String> = serde_json::from_value(index["weight_map"].clone())?;

    assert_eq!(
        weight_map
            .values()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        6,
        "ESMC-6B is published as six shards"
    );

    let mut expected = vec![
        // Backbone, under the "esmc." prefix optional_prefix descends into.
        "esmc.embed.weight".to_string(),
        "esmc.transformer.norm.weight".to_string(),
        // Head, at the TOP level and named lm_head — not sequence_head.
        "lm_head.0.weight".to_string(),
        "lm_head.2.weight".to_string(),
        "lm_head.3.weight".to_string(),
    ];
    for layer in 0..config.n_layers {
        expected.push(format!(
            "esmc.transformer.blocks.{layer}.attn.out_proj.weight"
        ));
        expected.push(format!("esmc.transformer.blocks.{layer}.ffn.1.weight"));
    }

    for name in &expected {
        assert!(
            weight_map.contains_key(name),
            "the loader will request {name}, which the 6B index does not contain"
        );
    }

    // And the head is genuinely NOT where the flat checkpoints keep it.
    assert!(
        !weight_map.contains_key("esmc.sequence_head.0.weight")
            && !weight_map.contains_key("sequence_head.0.weight"),
        "6B should have no sequence_head; that is why load_with_head exists"
    );
    Ok(())
}
