//! Unified trait for protein language model runners.
//!
//! [`PlmRunner`] provides a common interface for sequence embedding across
//! ESM2, AMPLIFY, ESMC and ESM3, so downstream code can be generic over the
//! runner type (benchmarking, ensembles, parity harnesses).
//!
//! # The special-token contract
//!
//! This is the part that matters for correctness. Every runner's tokenizer
//! wraps the amino-acid sequence in special tokens, so [`PlmRunner::embed`]
//! returns *more* rows than the sequence has residues. The old doc comment
//! said `(1, L, d_model)` where `L` includes "any BOS/EOS tokens" — and "any"
//! was doing far too much work: a caller comparing two runners' embeddings had
//! no way to know whether their rows were aligned, so a mismatch showed up as
//! silently misaligned residues rather than an error.
//!
//! Each runner now declares its layout via [`PlmRunner::special_tokens`], and
//! the provided [`PlmRunner::embed_residues`] strips it:
//!
//! | method | rows | use it when |
//! |---|---|---|
//! | [`embed`][PlmRunner::embed] | `L + leading + trailing` | you want the raw model output, specials included |
//! | [`embed_residues`][PlmRunner::embed_residues] | exactly `sequence.len()` | you are indexing by residue, or comparing across models |
//!
//! All four current runners are `BOS_EOS` (one leading, one trailing), but
//! that is a fact about these checkpoints, not a guarantee — declare the
//! layout in the impl rather than assuming it at the call site.
//!
//! ```no_run
//! # use ferritin_plms::plm_runner::PlmRunner;
//! # fn compare(a: &dyn PlmRunner, b: &dyn PlmRunner, seq: &str) -> anyhow::Result<()> {
//! // Row i of each corresponds to residue i, whatever the tokenizers do.
//! let x = a.embed_residues(seq)?;
//! let y = b.embed_residues(seq)?;
//! assert_eq!(x.dim(1)?, y.dim(1)?);
//! # Ok(())
//! # }
//! ```

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

// ── SpecialTokenLayout ────────────────────────────────────────────────────────

/// How many special tokens a runner's tokenizer wraps a sequence in.
///
/// Named counts rather than an enum of model names, so porting a new model
/// means declaring its shape rather than editing a `match` somewhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenLayout {
    /// Tokens prepended before the first residue (e.g. BOS / `<cls>`).
    pub leading: usize,
    /// Tokens appended after the last residue (e.g. EOS).
    pub trailing: usize,
}

impl SpecialTokenLayout {
    /// No special tokens: model output rows map 1:1 onto residues.
    pub const NONE: Self = Self {
        leading: 0,
        trailing: 0,
    };

    /// One leading BOS and one trailing EOS — what every currently ported
    /// model does.
    pub const BOS_EOS: Self = Self {
        leading: 1,
        trailing: 1,
    };

    /// Total number of non-residue rows.
    pub const fn total(&self) -> usize {
        self.leading + self.trailing
    }
}

// ── ModelMetadata ─────────────────────────────────────────────────────────────

/// Architecture facts a caller needs to size downstream layers.
///
/// Without this, callers had to hardcode `d_model` per model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelMetadata {
    /// Hidden width of the embeddings returned by [`PlmRunner::embed`].
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Token vocabulary size — the last dimension of [`PlmRunner::logits`].
    pub vocab_size: usize,
    /// Maximum supported sequence length, when the architecture has one.
    /// `None` for models with relative or rotary positions and no hard cap.
    pub max_positions: Option<usize>,
}

// ── PlmRunner ─────────────────────────────────────────────────────────────────

/// Trait implemented by all PLM runner types.
///
/// Structure-prediction and inverse-folding runners (`ESMFold2Runner`,
/// `ProteinMPNNRunner`) deliberately stay outside this trait — they are not
/// embedding models.
pub trait PlmRunner {
    /// Run a forward pass on `sequence` and return raw per-residue embeddings.
    ///
    /// Shape: `(1, L + special, d_model)`, where `special` is
    /// [`special_tokens().total()`][PlmRunner::special_tokens]. Rows for
    /// special tokens are **included**; use
    /// [`embed_residues`][Self::embed_residues] to get one row per residue.
    fn embed(&self, sequence: &str) -> Result<Tensor>;

    /// Model name / identifier string (e.g. "esm2", "amplify", "esmc").
    fn model_name(&self) -> &str;

    /// The special tokens this runner's tokenizer adds around a sequence.
    fn special_tokens(&self) -> SpecialTokenLayout;

    /// Architecture dimensions, for sizing downstream layers.
    fn metadata(&self) -> ModelMetadata;

    /// Device the model's weights live on.
    fn device(&self) -> &Device;

    /// Per-residue embeddings with special-token rows removed.
    ///
    /// Shape: `(1, sequence.len(), d_model)` — guaranteed, and therefore
    /// comparable row-for-row across models.
    ///
    /// Provided in terms of [`embed`][Self::embed] and
    /// [`special_tokens`][Self::special_tokens], so an implementor declares
    /// its layout rather than reimplementing the strip. This is what
    /// `ESMFold2Runner::fold_protein` used to hand-roll as
    /// `.narrow(1, 1, l)` against ESMC specifically.
    fn embed_residues(&self, sequence: &str) -> Result<Tensor> {
        let raw = self.embed(sequence)?;
        let layout = self.special_tokens();
        let rows = raw.dim(1)?;
        let expected = sequence.len() + layout.total();
        if rows != expected {
            bail!(
                "{}: embed() returned {rows} rows for a {}-residue sequence, but its \
                 declared special-token layout ({} leading, {} trailing) implies {expected}. \
                 The runner's SpecialTokenLayout does not match its tokenizer.",
                self.model_name(),
                sequence.len(),
                layout.leading,
                layout.trailing,
            );
        }
        Ok(raw.narrow(1, layout.leading, sequence.len())?)
    }

    /// Masked-LM logits over the token vocabulary.
    ///
    /// Shape: `(1, L + special, vocab_size)` — the raw form, aligned with
    /// [`embed`][Self::embed] rather than
    /// [`embed_residues`][Self::embed_residues].
    ///
    /// Defaults to an error: not every runner has a masked-LM head, and
    /// silently returning something else would be worse than refusing.
    fn logits(&self, _sequence: &str) -> Result<Tensor> {
        bail!("{} does not expose masked-LM logits", self.model_name())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Emits `sequence.len() + layout.total()` rows, so `embed_residues` has
    /// something consistent to strip.
    struct MockRunner {
        layout: SpecialTokenLayout,
        /// When set, `embed` lies about its row count to exercise the guard.
        rows_override: Option<usize>,
    }

    impl MockRunner {
        fn new(layout: SpecialTokenLayout) -> Self {
            Self {
                layout,
                rows_override: None,
            }
        }
    }

    impl PlmRunner for MockRunner {
        fn embed(&self, sequence: &str) -> Result<Tensor> {
            let rows = self
                .rows_override
                .unwrap_or(sequence.len() + self.layout.total());
            Tensor::zeros((1usize, rows, 16usize), DType::F32, &Device::Cpu)
                .map_err(anyhow::Error::from)
        }

        fn model_name(&self) -> &str {
            "mock"
        }

        fn special_tokens(&self) -> SpecialTokenLayout {
            self.layout
        }

        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                d_model: 16,
                n_layers: 2,
                vocab_size: 33,
                max_positions: Some(1024),
            }
        }

        fn device(&self) -> &Device {
            &Device::Cpu
        }
    }

    #[test]
    fn test_special_token_layout_totals() {
        assert_eq!(SpecialTokenLayout::NONE.total(), 0);
        assert_eq!(SpecialTokenLayout::BOS_EOS.total(), 2);
    }

    #[test]
    fn test_mock_runner_model_name() {
        assert_eq!(
            MockRunner::new(SpecialTokenLayout::BOS_EOS).model_name(),
            "mock"
        );
    }

    /// `embed` keeps the special-token rows.
    #[test]
    fn test_embed_includes_special_tokens() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        let t = runner.embed("ACDE").unwrap();
        assert_eq!(t.dims(), &[1, 6, 16], "4 residues + BOS + EOS");
    }

    /// `embed_residues` strips them, giving exactly one row per residue.
    #[test]
    fn test_embed_residues_strips_special_tokens() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        let t = runner.embed_residues("ACDE").unwrap();
        assert_eq!(t.dims(), &[1, 4, 16]);
    }

    /// A runner with no special tokens is passed through unchanged.
    #[test]
    fn test_embed_residues_noop_for_no_special_tokens() {
        let runner = MockRunner::new(SpecialTokenLayout::NONE);
        let t = runner.embed_residues("ACDE").unwrap();
        assert_eq!(t.dims(), &[1, 4, 16]);
    }

    /// Asymmetric layouts strip from the correct end.
    #[test]
    fn test_embed_residues_handles_leading_only() {
        let runner = MockRunner::new(SpecialTokenLayout {
            leading: 1,
            trailing: 0,
        });
        let t = runner.embed_residues("ACDE").unwrap();
        assert_eq!(t.dims(), &[1, 4, 16]);
    }

    /// A layout that disagrees with the tokenizer is reported, not silently
    /// turned into misaligned residues — the whole point of the contract.
    #[test]
    fn test_embed_residues_rejects_layout_mismatch() {
        let runner = MockRunner {
            layout: SpecialTokenLayout::BOS_EOS,
            rows_override: Some(4), // claims BOS_EOS but returns bare residues
        };
        let err = runner
            .embed_residues("ACDE")
            .map(|_| ())
            .expect_err("a layout that does not match the row count must error");
        assert!(
            err.to_string().contains("does not match its tokenizer"),
            "error should name the contract violation; got: {err}"
        );
    }

    #[test]
    fn test_metadata_is_reported() {
        let md = MockRunner::new(SpecialTokenLayout::BOS_EOS).metadata();
        assert_eq!(md.d_model, 16);
        assert_eq!(md.vocab_size, 33);
        assert_eq!(md.max_positions, Some(1024));
    }

    /// Runners without a masked-LM head refuse rather than returning something
    /// that is not logits.
    #[test]
    fn test_logits_defaults_to_unsupported() {
        let err = MockRunner::new(SpecialTokenLayout::BOS_EOS)
            .logits("ACDE")
            .map(|_| ())
            .expect_err("the default logits() must refuse");
        assert!(err.to_string().contains("does not expose masked-LM logits"));
    }
}
