//! Integration smoke tests for ESM3.
//!
//! Tests that require downloading model weights are marked `#[ignore]` so
//! they do not run in CI by default. Run them explicitly with:
//!
//! ```shell
//! cargo test -p ferritin-plms test_esm3 -- --ignored --nocapture
//! ```

mod support;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use ferritin_plms::{
    ESM3Config,
    esm3::{
        models::esm3::ESM3,
        tokenization::sequence::{decode_sequence, tokenize_sequence},
    },
};

/// Cosine-similarity floor for ESM3 sm-open per-residue embedding parity.
/// A 48-layer trunk accumulates more F32 rounding than the shallower ports, so
/// this is looser than the 1e-3 logit tolerance used for ESM2/AMPLIFY.
const ESM3_EMBED_COSINE_FLOOR: f32 = 0.99;

// ── Helpers ────────────────────────────────────────────────────────────────

/// Tiny ESM3 config for fast zeroed-weight unit tests.
///
/// Architecture is structurally identical to sm-open but with tiny dims so
/// VarBuilder::zeros completes quickly.
fn mini_esm3_config() -> ESM3Config {
    ESM3Config {
        d_model: 64,
        n_heads: 4,
        n_layers: 2,
        n_layers_geom: 1,
        v_head_transformer: 8,
        expansion_ratio: 2.0,
        scale_residue: false,
        mask_and_zero_frameless: false,
        qk_layernorm: true,
        bias: false,
        d_sequence_vocab: 64,
        d_structure_vocab: 4096,
        d_ss8_vocab: 11,
        d_sasa_vocab: 19,
        n_function_tracks: 8,
        d_function_vocab: 260,
        d_residue_vocab: 1478,
        n_rbf_bins: 16,
    }
}

const SHORT_SEQ: &str = "ACDEFGHIK"; // 9 residues → 11 tokens with BOS+EOS

const GFP: &str = concat!(
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCF",
    "SRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLE",
    "YNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNE",
    "KRDHMVLLEFVTAAGITHGMDELYK"
);

// ── Unit tests — no model download ─────────────────────────────────────────

#[test]
fn test_esm3_config_sm_open_fields() {
    let cfg = ESM3Config::sm_open();
    assert_eq!(cfg.d_model, 1536);
    assert_eq!(cfg.n_heads, 24);
    assert_eq!(cfg.n_layers, 48);
    assert_eq!(cfg.n_layers_geom, 1);
    assert_eq!(cfg.v_head_transformer, 256);
    assert_eq!(cfg.d_sequence_vocab, 64);
    assert_eq!(cfg.d_structure_vocab, 4096);
    assert_eq!(cfg.n_function_tracks, 8);
    assert_eq!(cfg.n_rbf_bins, 16);
}

#[test]
fn test_esm3_sequence_tokenize_roundtrip() {
    let seq = "ACDEFGHIKLMNPQRSTVWY";
    let tokens = tokenize_sequence(seq, true);
    // BOS + 20 residues + EOS = 22
    assert_eq!(tokens.len(), seq.len() + 2, "token count mismatch");

    // Decode strips BOS/EOS/MASK/PAD
    let decoded = decode_sequence(&tokens);
    assert_eq!(decoded, seq, "roundtrip mismatch");
}

#[test]
fn test_esm3_sequence_tokenize_no_special() {
    let seq = "ACDE";
    let tokens = tokenize_sequence(seq, false);
    assert_eq!(tokens.len(), 4);
    let decoded = decode_sequence(&tokens);
    assert_eq!(decoded, seq);
}

#[test]
fn test_esm3_forward_embedding_shape_zeroed_weights() -> Result<()> {
    let device = Device::Cpu;
    let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
    let config = mini_esm3_config();
    let d_model = config.d_model;
    let model = ESM3::load(vb, config)?;

    let token_ids = tokenize_sequence(SHORT_SEQ, true);
    let tokens = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?; // (1, L+2)

    let output = model.forward(
        Some(&tokens),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    let expected_len = SHORT_SEQ.len() + 2; // 11
    let embeddings = output.embeddings.expect("forward should return embeddings");
    assert_eq!(
        embeddings.dims(),
        &[1, expected_len, d_model],
        "embedding shape mismatch"
    );
    Ok(())
}

#[test]
fn test_esm3_forward_logit_shapes_zeroed_weights() -> Result<()> {
    let device = Device::Cpu;
    let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
    let config = mini_esm3_config();
    let model = ESM3::load(vb, config)?;

    let token_ids = tokenize_sequence(SHORT_SEQ, true);
    let tokens = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;
    let expected_len = SHORT_SEQ.len() + 2;

    let output = model.forward(
        Some(&tokens),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    let seq_logits = output.sequence_logits.expect("sequence logits");
    assert_eq!(
        seq_logits.dims(),
        &[1, expected_len, 64],
        "sequence logits shape"
    );

    let struct_logits = output.structure_logits.expect("structure logits");
    assert_eq!(
        struct_logits.dims(),
        &[1, expected_len, 4096],
        "structure logits shape"
    );

    let func_logits = output.function_logits.expect("function logits");
    assert_eq!(
        func_logits.dims(),
        &[1, expected_len, 8, 260],
        "function logits shape"
    );

    Ok(())
}

// ── Integration tests — require HuggingFace weights ───────────────────────

/// End-to-end: load esm3-sm-open-v1, embed a short sequence, validate shape.
///
/// Requires gated access: accept the Cambrian Non-Commercial license at
/// <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1> and run
/// `huggingface-cli login`.
#[test]
#[ignore = "requires downloading EvolutionaryScale/esm3-sm-open-v1 weights (~5 GB)"]
fn test_esm3_sm_open_embed_short_sequence() -> Result<()> {
    use ferritin_plms::{ESM3Models, ESM3Runner};

    let device = Device::Cpu;
    let runner = ESM3Runner::from_pretrained(ESM3Models::SmOpen, device)?;
    let embeddings = runner.embed_sequence(SHORT_SEQ)?;

    let expected_len = SHORT_SEQ.len() + 2;
    assert_eq!(
        embeddings.dims(),
        &[1, expected_len, 1536],
        "embedding shape mismatch: expected (1, {expected_len}, 1536)"
    );
    Ok(())
}

/// End-to-end: ESM3 sequence logits shape on GFP.
#[test]
#[ignore = "requires downloading EvolutionaryScale/esm3-sm-open-v1 weights (~5 GB)"]
fn test_esm3_sm_open_gfp_logit_shape() -> Result<()> {
    use candle_core::DType;
    use candle_core::pickle::PthTensors;
    use candle_nn::VarBuilder;
    use hf_hub::HFClientSync;

    let device = Device::Cpu;
    let client = HFClientSync::new()?;
    let weights_path = client
        .model("EvolutionaryScale", "esm3-sm-open-v1")
        .download_file()
        .filename("esm3_sm_open_v1.pth")
        .send()?;

    let pth = PthTensors::new(&weights_path, None)?;
    let vb = VarBuilder::from_backend(Box::new(pth), DType::F32, device.clone());
    let model = ESM3::load(vb, ESM3Config::sm_open())?;

    let token_ids = tokenize_sequence(GFP, true);
    let tokens = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;

    let output = model.forward(
        Some(&tokens),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )?;

    let expected_len = GFP.len() + 2;
    let seq_logits = output.sequence_logits.expect("sequence logits");
    assert_eq!(seq_logits.dims(), &[1, expected_len, 64]);
    Ok(())
}

/// Numerical parity: ESM3 sm-open Rust per-residue embeddings vs the Python
/// reference. The reference `(L+2, 1536)` tensor keeps the BOS/EOS rows, so the
/// Rust `(1, L+2, 1536)` output aligns 1:1 after squeezing the batch dim
/// (`SpecialTokens::NONE`); a BOS/EOS placement mismatch surfaces as a failure
/// at position 0 or L+1 rather than passing silently.
///
/// Requires gated access to `EvolutionaryScale/esm3-sm-open-v1` (accept the
/// Cambrian Non-Commercial license and `huggingface-cli login`) plus the
/// fixture `tests/fixtures/esm3_parity.safetensors` from
/// `scripts/generate_esm3_fixtures.py`.
///
/// Run: `cargo test -p ferritin-plms test_esm3_parity -- --ignored --nocapture`
#[test]
#[ignore = "requires gated HF download and fixture: run scripts/generate_esm3_fixtures.py"]
fn test_esm3_parity_vs_python_reference() -> Result<()> {
    use ferritin_plms::{ESM3Models, ESM3Runner};
    use support::parity::{ParityFixture, SpecialTokens, align_rows, assert_embeddings_close};

    let device = Device::Cpu;
    let fixture = ParityFixture::load("esm3_parity", &device)?;
    let ref_embeddings = fixture.tensor("embeddings")?;

    let runner = ESM3Runner::from_pretrained(ESM3Models::SmOpen, device)?;
    let rust = runner.embed_sequence(SHORT_SEQ)?; // (1, L+2, 1536)
    let rust_embeddings = align_rows(&rust, SpecialTokens::NONE)?;

    assert_embeddings_close(&rust_embeddings, ref_embeddings, ESM3_EMBED_COSINE_FLOOR)?;
    println!("ESM3 sm-open embedding parity OK (cosine floor {ESM3_EMBED_COSINE_FLOOR})");
    Ok(())
}
