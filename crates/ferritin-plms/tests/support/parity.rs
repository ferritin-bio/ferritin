//! Shared golden-fixture parity harness for the PLM Candle ports.
//!
//! Every parity test compares a Rust forward pass against a Python reference
//! tensor stored in `tests/fixtures/<name>.safetensors`. Before this module
//! each test hand-rolled its own fixture loader and comparison logic
//! (`test_plm_ligandmpnn.rs` had a private `kl_divergence`, `test_plm_amplify`
//! and `test_plm_esm2` each repeated a load-safetensors-and-max-diff block).
//! This module centralises:
//!
//!   * [`ParityFixture`] — resolve and load a fixture, with a clear
//!     "run scripts/generate_X_fixtures.py" error when it is missing.
//!   * [`assert_logits_close`] — max-absolute-difference, reports the offending
//!     `(position, vocab_index)` on failure.
//!   * [`assert_distribution_close`] — per-position KL divergence, reports the
//!     max KL and the position at which it occurred.
//!   * [`assert_embeddings_close`] — per-residue cosine similarity, for models
//!     whose meaningful output is an embedding rather than a distribution
//!     (ESMC, ESM3).
//!
//! ## Special-token alignment
//!
//! The reference tensors are **not** all laid out the same way, because the
//! tokenizers do not agree on which special tokens they prepend/append:
//!
//!   * **AMPLIFY** and **ESM2** wrap the sequence in BOS + `L` residues + EOS.
//!     Their fixture generators (`generate_amplify_fixtures.py`,
//!     `generate_esm2_fixtures.py`) strip BOS/EOS and save only the `L`
//!     interior rows, so the reference is `(L, V)` with **no** special tokens.
//!   * **ProteinMPNN / LigandMPNN** have no special tokens at all; the
//!     reference `log_probs` is `(L, 21)` one row per residue.
//!   * **ESMC** and **ESM3** wrap the sequence in BOS + `L` + EOS and their
//!     embedding references keep those two rows, so the reference is
//!     `(L + 2, D)`.
//!
//! A silent off-by-one in that alignment would make a genuinely broken port
//! look correct, so the Rust side must be trimmed to match the reference
//! explicitly. Use [`SpecialTokens`] + [`align_rows`] to describe how many
//! leading/trailing rows to drop from the Rust output rather than open-coding a
//! `narrow` at each call site.
#![allow(dead_code)]

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;

/// A loaded parity fixture: the safetensors key→tensor map plus the fixture
/// name, used for good error messages.
#[derive(Debug)]
pub struct ParityFixture {
    name: String,
    tensors: HashMap<String, Tensor>,
}

impl ParityFixture {
    /// Load `tests/fixtures/<name>.safetensors` into memory.
    ///
    /// On absence, returns an error naming the generator script to run, rather
    /// than a bare "file not found". `name` is the fixture stem without the
    /// `.safetensors` extension (e.g. `"esm2_parity"`); the generator script is
    /// inferred as `generate_<stem>_fixtures.py` where `<stem>` drops a trailing
    /// `_parity`. When that inference is wrong (e.g. the ProteinMPNN fixture is
    /// `1BC8_log_probs` but the script is `generate_proteinmpnn_fixtures.py`),
    /// use [`ParityFixture::load_with_generator`].
    pub fn load(name: &str, device: &Device) -> Result<Self> {
        Self::load_with_generator(name, script_stem(name), device)
    }

    /// Like [`load`](Self::load) but names the generator script explicitly, for
    /// fixtures whose file stem does not match their `generate_<X>_fixtures.py`
    /// script. `generator` is the `<X>` in that filename.
    pub fn load_with_generator(name: &str, generator: &str, device: &Device) -> Result<Self> {
        let path = fixture_path(name);
        if !path.exists() {
            bail!(
                "Parity fixture not found at {}.\n\
                 Generate it with: python scripts/generate_{}_fixtures.py \
                 --output crates/ferritin-plms/tests/fixtures/",
                path.display(),
                generator,
            );
        }
        let tensors = candle_core::safetensors::load(&path, device)?;
        Ok(Self {
            name: name.to_string(),
            tensors,
        })
    }

    /// Fetch a tensor by key, erroring (with the fixture name and the keys that
    /// *are* present) when it is absent.
    pub fn tensor(&self, key: &str) -> Result<&Tensor> {
        self.tensors.get(key).ok_or_else(|| {
            let mut keys: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
            keys.sort_unstable();
            anyhow::anyhow!(
                "fixture '{}' has no tensor '{}'; present keys: [{}]",
                self.name,
                key,
                keys.join(", ")
            )
        })
    }

    /// The fixture stem, for diagnostics.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Whether the HF-download-dependent parity tests should actually run.
///
/// These tests need the committed fixture (cheap, in-repo) **and** the model
/// weights, which are fetched from Hugging Face on first run and cached. That
/// download is far too heavy for the PR path, so the tests are off unless
/// `FERRITIN_HF_TESTS` is set to `1`/`true`. The nightly workflow sets it; a
/// developer with the weights cached can run them the same way locally:
///
/// ```bash
/// FERRITIN_HF_TESTS=1 cargo test -p ferritin-plms
/// ```
///
/// A gated-off test returns early (reported as passed) rather than being
/// `#[ignore]`d, so `cargo test` exercises the surrounding wiring without the
/// download and the nightly `--include-ignored` run does the real comparison.
pub fn hf_tests_enabled() -> bool {
    matches!(
        std::env::var("FERRITIN_HF_TESTS").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Resolve a fixture stem to an absolute path under `tests/fixtures/`.
pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.safetensors"))
}

/// Map a fixture stem to its generator-script stem for error messages.
/// `esm2_parity` -> `esm2`, `amplify_parity` -> `amplify`, etc.
fn script_stem(name: &str) -> &str {
    name.strip_suffix("_parity").unwrap_or(name)
}

/// Describes how many special-token rows the **Rust** output carries relative
/// to the reference tensor, so [`align_rows`] can trim the Rust side to match.
#[derive(Clone, Copy, Debug, Default)]
pub struct SpecialTokens {
    /// Leading special-token rows in the Rust output not present in the
    /// reference (e.g. BOS = 1 for ESM2/AMPLIFY interior-only references).
    pub leading: usize,
    /// Trailing special-token rows in the Rust output not present in the
    /// reference (e.g. EOS = 1).
    pub trailing: usize,
}

impl SpecialTokens {
    /// No special tokens to strip; Rust and reference rows already align.
    /// (ProteinMPNN/LigandMPNN, and any reference that keeps BOS/EOS.)
    pub const NONE: Self = Self {
        leading: 0,
        trailing: 0,
    };
    /// Strip one leading BOS and one trailing EOS from the Rust output — the
    /// AMPLIFY / ESM2 interior-only reference layout.
    pub const BOS_EOS: Self = Self {
        leading: 1,
        trailing: 1,
    };
}

/// Reduce a Rust model output to a 2D `(rows, features)` tensor aligned to the
/// reference, dropping `special.leading`/`special.trailing` rows.
///
/// Accepts either a batched `(1, L, F)` tensor (the common runner output) or an
/// already-2D `(L, F)` tensor. The batch dimension, if present, must be 1.
pub fn align_rows(rust: &Tensor, special: SpecialTokens) -> Result<Tensor> {
    let t = match rust.dims() {
        [1, _, _] => rust.squeeze(0)?,
        [_, _] => rust.clone(),
        other => bail!(
            "align_rows expects (1, L, F) or (L, F); got shape {:?}",
            other
        ),
    };
    let (rows, _feats) = t.dims2()?;
    if special.leading + special.trailing >= rows {
        bail!(
            "align_rows: stripping {} leading + {} trailing rows leaves nothing of {} rows",
            special.leading,
            special.trailing,
            rows
        );
    }
    let kept = rows - special.leading - special.trailing;
    Ok(t.narrow(0, special.leading, kept)?)
}

/// Assert that two logit tensors agree to within `tol` in max-absolute
/// difference. Both must be 2D `(L, V)` and identically shaped.
///
/// On failure the error names the exact `(position, vocab_index)` at which the
/// largest disagreement occurred, not just "assertion failed".
pub fn assert_logits_close(rust: &Tensor, reference: &Tensor, tol: f32) -> Result<()> {
    let rust_rows = rows2(rust, "rust logits")?;
    let ref_rows = rows2(reference, "reference logits")?;
    check_same_shape(&rust_rows, &ref_rows, "logits")?;

    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);
    for (pos, (rrow, prow)) in rust_rows.iter().zip(ref_rows.iter()).enumerate() {
        for (vi, (&r, &p)) in rrow.iter().zip(prow.iter()).enumerate() {
            let d = (r - p).abs();
            if d > worst {
                worst = d;
                worst_at = (pos, vi);
            }
        }
    }
    if worst > tol {
        let (pos, vi) = worst_at;
        bail!(
            "logit parity broke at (position {pos}, vocab_index {vi}): \
             |rust {} - ref {}| = {worst:.6e} exceeds tol {tol:.1e}",
            rust_rows[pos][vi],
            ref_rows[pos][vi],
        );
    }
    Ok(())
}

/// Assert that two probability distributions agree to within `kl_threshold` in
/// per-position KL divergence. Both must be 2D `(L, V)` where each row is a
/// probability vector (non-negative, ideally summing to 1).
///
/// KL(P‖Q) is computed with P = reference, Q = rust. On failure the error names
/// the position of the largest KL. Lifts the implementation that previously
/// lived in `test_plm_ligandmpnn.rs`.
pub fn assert_distribution_close(
    rust: &Tensor,
    reference: &Tensor,
    kl_threshold: f64,
) -> Result<()> {
    let rust_rows = rows2(rust, "rust distribution")?;
    let ref_rows = rows2(reference, "reference distribution")?;
    check_same_shape(&rust_rows, &ref_rows, "distribution")?;

    let mut max_kl = 0.0f64;
    let mut max_at = 0usize;
    for (pos, (q, p)) in rust_rows.iter().zip(ref_rows.iter()).enumerate() {
        let kl = kl_divergence(p, q);
        if kl > max_kl {
            max_kl = kl;
            max_at = pos;
        }
    }
    if max_kl > kl_threshold {
        bail!(
            "distribution parity broke at position {max_at}: \
             KL = {max_kl:.6} exceeds threshold {kl_threshold}"
        );
    }
    Ok(())
}

/// Assert that per-residue embeddings agree, i.e. every row's cosine similarity
/// to the reference is at least `cosine_floor`. Both must be 2D `(L, D)` and
/// identically shaped.
///
/// On failure the error names the position with the lowest cosine similarity.
pub fn assert_embeddings_close(rust: &Tensor, reference: &Tensor, cosine_floor: f32) -> Result<()> {
    let rust_rows = rows2(rust, "rust embeddings")?;
    let ref_rows = rows2(reference, "reference embeddings")?;
    check_same_shape(&rust_rows, &ref_rows, "embeddings")?;

    let mut min_cos = f32::INFINITY;
    let mut min_at = 0usize;
    for (pos, (r, p)) in rust_rows.iter().zip(ref_rows.iter()).enumerate() {
        let cos = cosine_similarity(r, p);
        if cos < min_cos {
            min_cos = cos;
            min_at = pos;
        }
    }
    if min_cos < cosine_floor {
        bail!(
            "embedding parity broke at position {min_at}: \
             cosine similarity {min_cos:.6} is below floor {cosine_floor}"
        );
    }
    Ok(())
}

// ── internals ──────────────────────────────────────────────────────────────

/// KL(P‖Q) = Σ P_i · ln(P_i / Q_i). Both slices must be probability vectors.
fn kl_divergence(p: &[f32], q: &[f32]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(pi, _)| **pi > 1e-9)
        .map(|(pi, qi)| {
            let qi = qi.max(1e-9_f32);
            (*pi as f64) * ((*pi as f64).ln() - (qi as f64).ln())
        })
        .sum()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Convert a 2D tensor to row-major `Vec<Vec<f32>>`, with a labelled error.
fn rows2(t: &Tensor, what: &str) -> Result<Vec<Vec<f32>>> {
    let dims = t.dims();
    if dims.len() != 2 {
        bail!("{what} must be a 2D (L, F) tensor; got shape {dims:?}");
    }
    Ok(t.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?)
}

fn check_same_shape(rust: &[Vec<f32>], reference: &[Vec<f32>], what: &str) -> Result<()> {
    if rust.len() != reference.len() {
        bail!(
            "{what}: sequence length mismatch (rust {} rows vs reference {} rows)",
            rust.len(),
            reference.len()
        );
    }
    let rust_f = rust.first().map(Vec::len).unwrap_or(0);
    let ref_f = reference.first().map(Vec::len).unwrap_or(0);
    if rust_f != ref_f {
        bail!("{what}: feature-dim mismatch (rust {rust_f} vs reference {ref_f})");
    }
    Ok(())
}
