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
    use ferritin_plms::esm3::pretrained::ESM3Models;
    use ferritin_plms::loader::LoadOptions;

    let device = Device::Cpu;
    let (source, filename, config) = ESM3Models::SmOpen.model_info();
    let vb = source.var_builder(filename, &LoadOptions::new(device.clone()))?;
    let model = ESM3::load(vb, config)?;

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
    // Skips (loudly) while no esm3_parity fixture is committed — see
    // PARITY_COVERAGE and ferritin-100.20.
    let Some(fixture) = ParityFixture::load_or_skip("esm3_parity", &device)? else {
        return Ok(());
    };
    let ref_embeddings = fixture.tensor("embeddings")?;

    let runner = ESM3Runner::from_pretrained(ESM3Models::SmOpen, device)?;
    let rust = runner.embed_sequence(SHORT_SEQ)?; // (1, L+2, 1536)
    let rust_embeddings = align_rows(&rust, SpecialTokens::NONE)?;

    assert_embeddings_close(&rust_embeddings, ref_embeddings, ESM3_EMBED_COSINE_FLOOR)?;
    println!("ESM3 sm-open embedding parity OK (cosine floor {ESM3_EMBED_COSINE_FLOOR})");
    Ok(())
}

/// The structure encoder loads from the released checkpoint and emits tokens.
///
/// This replaces a test that asserted the encoder *refuses* to load. It did,
/// for good reason: the port targeted a network the checkpoint does not
/// contain. Every one of those differences is now modelled, so the refusal is
/// gone and this pins the behaviour that replaced it (ferritin-100.22).
///
/// What this does and does not establish: it proves the encoder loads and emits
/// in-range, deterministic tokens. It does **not** establish numerical parity —
/// a subtly wrong geometry passes here, which is not hypothetical: it did, until
/// `test_esm3_structure_tokens_match_python_reference` caught two dropped FFN
/// bias tensors (ferritin-100.27). That test is the one that pins the numbers.
#[test]
#[ignore = "downloads esm3_structure_encoder_v0.pth (~60 MB)"]
fn test_esm3_structure_encoder_loads_and_encodes() -> anyhow::Result<()> {
    use candle_core::Tensor;
    use ferritin_plms::esm3::pretrained::StructureEncoderRunner;

    let runner = StructureEncoderRunner::from_pretrained(Device::Cpu)?;

    // An ideal alpha-helix: rise 1.5 A, 100 degrees per residue.
    let l = 32usize;
    let (r, rise, turn) = (2.3f64, 1.5f64, 100f64.to_radians());
    let mut v: Vec<f32> = Vec::with_capacity(l * 9);
    for i in 0..l {
        for off in [-1.0f64, 0.0, 1.0] {
            let t = (i as f64 + off * 0.35) * turn;
            let z = (i as f64 + off * 0.35) * rise;
            v.extend_from_slice(&[(r * t.cos()) as f32, (r * t.sin()) as f32, z as f32]);
        }
    }
    let coords = Tensor::from_vec(v, (1, l, 3, 3), &Device::Cpu)?;

    let tokens = runner.encode(&coords)?;
    assert_eq!(tokens.dims(), &[1, l], "one structure token per residue");

    let ids = tokens.flatten_all()?.to_vec1::<u32>()?;
    assert!(
        ids.iter().all(|&i| i < 4096),
        "tokens must index the 4096-entry codebook; got max {:?}",
        ids.iter().max()
    );

    // A helix is locally periodic, so its interior should not tokenize as 32
    // distinct states — that would mean the geometry is not reaching the
    // codebook.
    let distinct: std::collections::HashSet<u32> = ids[4..l - 4].iter().copied().collect();
    assert!(
        distinct.len() < ids[4..l - 4].len(),
        "a regular helix should reuse structure tokens; got {distinct:?}"
    );

    let again = runner.encode(&coords)?.flatten_all()?.to_vec1::<u32>()?;
    assert_eq!(ids, again, "encoding must be deterministic");
    Ok(())
}

/// Structure token parity against the reference `StructureTokenEncoder`
/// (ferritin-100.27).
///
/// The port merged in PR #201 had no numerical check at all — every test was
/// self-consistency (tensors resolve, tokens in range, encoding deterministic,
/// a helix reuses tokens), and a subtly wrong geometry passes all of it. The
/// defects that would survive were specific: a transposed rotation convention
/// in the gathered frames, an off-by-one in the relative-position bin shift,
/// the wrong neighbour taken as the query node, or the distance and rotation
/// per-head scales applied in the wrong order.
///
/// Structure tokens are discrete codebook indices, so this asserts **exact
/// equality** rather than a tolerance. Any of the above changes which codebook
/// entry a residue lands on; none of them produce "nearly the right id".
///
/// The backbone coordinates come from the fixture rather than being rebuilt
/// here, so the two sides cannot disagree about the *input*. It is a real
/// 93-residue protein (1BC8 chain C) rather than the ideal alpha-helix the
/// neighbouring test uses: a helix is locally periodic and exercises a narrow,
/// forgiving slice of the codebook.
#[test]
#[ignore = "downloads esm3_structure_encoder_v0.pth (~60 MB)"]
fn test_esm3_structure_tokens_match_python_reference() -> anyhow::Result<()> {
    use ferritin_plms::esm3::pretrained::StructureEncoderRunner;
    use support::parity::ParityFixture;

    let device = Device::Cpu;
    let Some(fixture) = ParityFixture::load_or_skip("esm3_structure_parity", &device)? else {
        return Ok(());
    };

    let coords = fixture.tensor("backbone_coords")?; // (L, 3, 3)
    let expected: Vec<u32> = fixture
        .tensor("structure_tokens")?
        .to_dtype(candle_core::DType::U32)?
        .to_vec1()?;

    let runner = StructureEncoderRunner::from_pretrained(device.clone())?;
    let tokens = runner.encode(&coords.unsqueeze(0)?)?;
    assert_eq!(
        tokens.dims(),
        &[1, expected.len()],
        "one structure token per residue"
    );
    let actual: Vec<u32> = tokens.flatten_all()?.to_vec1()?;

    if actual != expected {
        let disagreements: Vec<String> = actual
            .iter()
            .zip(expected.iter())
            .enumerate()
            .filter(|(_, (a, e))| a != e)
            .take(8)
            .map(|(i, (a, e))| format!("residue {i}: got {a}, reference {e}"))
            .collect();
        anyhow::bail!(
            "{} of {} structure tokens disagree with the reference encoder:\n  {}",
            actual
                .iter()
                .zip(expected.iter())
                .filter(|(a, e)| a != e)
                .count(),
            expected.len(),
            disagreements.join("\n  ")
        );
    }
    Ok(())
}
