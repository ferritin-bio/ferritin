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
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use ferritin_plms::esmc::pretrained::ESMCModels as Variants;
    use hf_hub::HFClientSync;

    let device = Device::Cpu;
    let (repo_id, config) = Variants::ESMC300M.model_info();
    let (owner, name) = repo_id.split_once('/').unwrap_or(("", repo_id));
    let client = HFClientSync::new()?;
    let weights_path = client
        .model(owner, name)
        .download_file()
        .filename("model.safetensors")
        .send()?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)? };
    let vb_root = if vb
        .get((config.embedding_dim, config.d_model), "esmc.embed.weight")
        .is_ok()
    {
        vb.pp("esmc")
    } else {
        vb
    };
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
