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
//! # Batching
//!
//! [`embed_batch`][PlmRunner::embed_batch] runs several sequences through one
//! forward pass, and [`embed_residues_batch`][PlmRunner::embed_residues_batch]
//! is its residue-aligned form. The default `embed_batch` just loops and
//! zero-pads, so it is correct for every runner; a runner overrides it only
//! once its architecture honours a padding mask, since without one the pad
//! tokens change the *real* residues' embeddings with no error raised.
//!
//! The four ported architectures want that mask in three different shapes, so
//! [`pad_token_batch`] builds one `TokenBatch` and each runner converts:
//! ESM-2 passes the `1 = real` mask straight through, AMPLIFY converts it with
//! [`additive_padding_mask`], and ESM-C and ESM-3 use it as a `where_cond`
//! predicate.
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

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, Tensor};

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

    /// One trailing EOS and no BOS — T5's layout, which ProtT5 uses.
    ///
    /// Distinct from [`BOS_EOS`][Self::BOS_EOS] by one leading row, which is
    /// exactly enough to misalign every residue if assumed rather than read.
    pub const EOS_ONLY: Self = Self {
        leading: 0,
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

// ── Batching ──────────────────────────────────────────────────────────────────

/// A right-padded batch of token ids and the padding mask that goes with it.
///
/// Built once by [`pad_token_batch`], then converted into whatever shape a
/// given model's forward pass wants. The four ported architectures disagree
/// about that — ESM-2 takes a `1 = real token` mask, AMPLIFY takes an additive
/// pre-softmax bias, ESM-C and ESM-3 take a `where_cond` predicate — and
/// deriving all three from one mask is what keeps that disagreement from
/// becoming three subtly different notions of "padded".
#[derive(Debug, Clone)]
pub struct TokenBatch {
    /// `(batch, padded_len)` token ids, dtype `U32`.
    pub ids: Tensor,
    /// `(batch, padded_len)`, dtype `U8`: 1 at a real token, 0 at padding.
    pub mask: Tensor,
    /// Unpadded token count of each row, in input order.
    pub lengths: Vec<usize>,
}

impl TokenBatch {
    /// The length every row was padded to — the longest row.
    pub fn padded_len(&self) -> usize {
        self.lengths.iter().copied().max().unwrap_or(0)
    }

    /// The padding mask cast to `dtype`, for models that want it as a float.
    pub fn mask_as(&self, dtype: DType) -> Result<Tensor> {
        Ok(self.mask.to_dtype(dtype)?)
    }
}

/// Right-pad token rows to a common length and build the matching mask.
///
/// `pad_id` must be the tokenizer's actual pad token, not an arbitrary unused
/// id. It is not cosmetic: [`ESMC::forward`][crate::esmc::models::esmc::ESMC::forward]
/// reconstructs its own key-padding mask as `tokens.ne(pad_id)` when none is
/// supplied, so padding with a different id would leave those positions
/// looking like real residues to the one model that infers its own mask.
pub fn pad_token_batch(rows: &[Vec<u32>], pad_id: u32, device: &Device) -> Result<TokenBatch> {
    if rows.is_empty() {
        bail!("pad_token_batch: empty batch — there is nothing to embed");
    }
    let lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
    let padded_len = lengths.iter().copied().max().unwrap_or(0);

    let mut ids = Vec::with_capacity(rows.len() * padded_len);
    let mut mask = Vec::with_capacity(rows.len() * padded_len);
    for row in rows {
        ids.extend_from_slice(row);
        ids.resize(ids.len() + padded_len - row.len(), pad_id);
        mask.extend(std::iter::repeat_n(1u8, row.len()));
        mask.extend(std::iter::repeat_n(0u8, padded_len - row.len()));
    }

    Ok(TokenBatch {
        ids: Tensor::from_vec(ids, (rows.len(), padded_len), device)?,
        mask: Tensor::from_vec(mask, (rows.len(), padded_len), device)?,
        lengths,
    })
}

/// Zero every row of `embeddings` that sits at a padded position.
///
/// `embeddings` is `(batch, seq_len, d)`, `mask` is `(batch, seq_len)` with
/// `1` at real tokens. A padded query still produces a full output row — the
/// key-side mask only stops *other* positions attending to it — and that row
/// is arbitrary rather than meaningfully zero. Zeroing it makes
/// [`PlmRunner::embed_batch`] deterministic regardless of what each
/// architecture happens to compute there, so pooling over the sequence axis
/// does not silently mix garbage into the result.
pub fn zero_padded_rows(embeddings: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let keep = mask.to_dtype(embeddings.dtype())?.unsqueeze(2)?;
    Ok(embeddings.broadcast_mul(&keep)?)
}

/// The additive value used for padded key positions, chosen per dtype.
///
/// It has to be large enough that `exp(score + fill - max)` underflows to zero
/// and small enough to stay finite in the tensor's dtype. F16 tops out at
/// ±65504, so the usual `-1e9` (or `f32::MIN`) would saturate to `-inf` there;
/// `-1e4` is still ~10^4 below any realistic scaled dot product, so softmax
/// returns exactly 0 for those positions in every supported dtype.
pub fn attention_mask_fill_value(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => -1e4,
        _ => -1e9,
    }
}

/// Turn a `(batch, seq_len)` padding mask (1 = real token, 0 = pad) into an
/// additive `(batch, seq_len)` bias in `dtype`.
///
/// The bias is `0` at real tokens and [`attention_mask_fill_value`] at pads,
/// and is meant to be added to the attention **scores before softmax**.
/// Multiplying the probabilities after softmax is not equivalent: it leaves the
/// surviving probabilities un-renormalised and still lets pad positions steal
/// mass.
///
/// The two ported architectures that take an additive mask want it at
/// different ranks — ESM-2 as `(batch, 1, 1, seq)`, AMPLIFY as `(batch, seq)`
/// which it broadcasts itself — so this returns the 2-D core and each model
/// reshapes.
pub fn additive_padding_mask(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    let (batch, seq_len) = mask.dims2()?;
    let mask = mask.to_dtype(dtype)?;
    let ones = Tensor::ones((batch, seq_len), dtype, mask.device())?;
    // (1 - mask) * fill → 0 where real, `fill` where pad.
    Ok((ones - &mask)?.affine(attention_mask_fill_value(dtype), 0.0)?)
}

/// Right-pad a set of `(1, rows_i, d)` tensors with zeros to a common row
/// count and concatenate them into `(B, rows_max, d)`.
pub fn stack_padded(rows: &[Tensor]) -> Result<Tensor> {
    let first = rows
        .first()
        .ok_or_else(|| anyhow!("stack_padded: empty batch — there is nothing to stack"))?;
    let width = first.dim(2)?;
    let max_rows = rows
        .iter()
        .map(|t| t.dim(1))
        .collect::<candle_core::Result<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0);

    let mut padded = Vec::with_capacity(rows.len());
    for (i, t) in rows.iter().enumerate() {
        if t.dim(0)? != 1 {
            bail!(
                "stack_padded: element {i} has batch dimension {}, expected 1",
                t.dim(0)?
            );
        }
        if t.dim(2)? != width {
            bail!(
                "stack_padded: element {i} is {} wide but element 0 is {width} wide",
                t.dim(2)?
            );
        }
        let n = t.dim(1)?;
        padded.push(if n == max_rows {
            t.clone()
        } else {
            let pad = Tensor::zeros((1, max_rows - n, width), t.dtype(), t.device())?;
            Tensor::cat(&[t, &pad], 1)?
        });
    }
    Ok(Tensor::cat(&padded, 0)?)
}

// ── PlmRunner ─────────────────────────────────────────────────────────────────

/// Trait implemented by all PLM runner types.
///
/// Structure and inverse-folding runners (`ProteinMPNNRunner`,
/// `StructureEncoderRunner`) deliberately stay outside this trait — they
/// consume or emit structure, not sequence embeddings.
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

    /// How many residues `sequence` encodes.
    ///
    /// One per byte for the standard amino-acid alphabets, which is the
    /// default. SaProt is the exception: it tokenizes each residue as an
    /// (amino acid, 3Di structural state) pair over a 20x20 product alphabet,
    /// so `"MdAaLp"` is three residues, not six (ferritin-goh.3).
    ///
    /// [`embed_residues`][Self::embed_residues] uses this rather than
    /// `sequence.len()`, so a model whose alphabet is not one byte per residue
    /// still gets exactly one row per residue.
    fn residue_count(&self, sequence: &str) -> usize {
        sequence.len()
    }

    /// Per-residue embeddings with special-token rows removed.
    ///
    /// Shape: `(1, sequence.len(), d_model)` — guaranteed, and therefore
    /// comparable row-for-row across models.
    ///
    /// Provided in terms of [`embed`][Self::embed] and
    /// [`special_tokens`][Self::special_tokens], so an implementor declares
    /// its layout rather than reimplementing the strip — which callers used to
    /// hand-roll as `.narrow(1, 1, l)` against one model's layout specifically.
    fn embed_residues(&self, sequence: &str) -> Result<Tensor> {
        let raw = self.embed(sequence)?;
        let layout = self.special_tokens();
        let residues = self.residue_count(sequence);
        let rows = raw.dim(1)?;
        let expected = residues + layout.total();
        if rows != expected {
            bail!(
                "{}: embed() returned {rows} rows for a {residues}-residue sequence, but its \
                 declared special-token layout ({} leading, {} trailing) implies {expected}. \
                 The runner's SpecialTokenLayout does not match its tokenizer.",
                self.model_name(),
                layout.leading,
                layout.trailing,
            );
        }
        Ok(raw.narrow(1, layout.leading, residues)?)
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

    /// Embed several sequences in **one** forward pass.
    ///
    /// Shape: `(B, T_max, d_model)`, where `T_max` is the longest
    /// `residue_count + special_tokens().total()` in `sequences`. Rows are
    /// right-padded and **every padded row is exactly zero**, so the result is
    /// the same whatever each architecture happens to compute at a pad
    /// position.
    ///
    /// Row `i`'s real content occupies
    /// `0 .. residue_count(sequences[i]) + special_tokens().total()`, which
    /// means the *trailing* special token sits at a different column in each
    /// row. Use [`embed_residues_batch`][Self::embed_residues_batch] when you
    /// want column `j` to be residue `j` for every row.
    ///
    /// # Default implementation
    ///
    /// Loops over [`embed`][Self::embed] and zero-pads. Correct for any
    /// runner, but it gives up the throughput this method exists for: `B`
    /// kernel launches instead of one. Override it with a real batched forward
    /// pass — which requires the model to honour a padding mask, because
    /// without one the padded keys leak into attention and change the *real*
    /// residues' embeddings, a silent quality regression rather than an error.
    fn embed_batch(&self, sequences: &[&str]) -> Result<Tensor> {
        if sequences.is_empty() {
            bail!(
                "{}: embed_batch called with an empty batch",
                self.model_name()
            );
        }
        let rows = sequences
            .iter()
            .map(|s| self.embed(s))
            .collect::<Result<Vec<_>>>()?;
        stack_padded(&rows)
    }

    /// Per-residue embeddings for a batch, special-token rows removed.
    ///
    /// Shape: `(B, R_max, d_model)` where `R_max` is the largest
    /// [`residue_count`][Self::residue_count]. Row `i`, column `j` is residue
    /// `j` of `sequences[i]`; columns past that sequence's length are exactly
    /// zero. This is the batched analogue of
    /// [`embed_residues`][Self::embed_residues] and the form to pool over.
    ///
    /// Provided in terms of [`embed_batch`][Self::embed_batch], and it
    /// re-checks the row count: an override that pads to the wrong length, or
    /// declares a `SpecialTokenLayout` its tokenizer does not follow, fails
    /// here rather than returning misaligned residues.
    fn embed_residues_batch(&self, sequences: &[&str]) -> Result<Tensor> {
        if sequences.is_empty() {
            bail!(
                "{}: embed_residues_batch called with an empty batch",
                self.model_name()
            );
        }
        let raw = self.embed_batch(sequences)?;
        let layout = self.special_tokens();
        let counts: Vec<usize> = sequences.iter().map(|s| self.residue_count(s)).collect();
        let max_residues = counts.iter().copied().max().unwrap_or(0);

        let batch = raw.dim(0)?;
        if batch != sequences.len() {
            bail!(
                "{}: embed_batch returned {batch} rows for {} sequences",
                self.model_name(),
                sequences.len()
            );
        }
        let rows = raw.dim(1)?;
        let expected = max_residues + layout.total();
        if rows != expected {
            bail!(
                "{}: embed_batch returned {rows} columns for a batch whose longest sequence has \
                 {max_residues} residues, but its declared special-token layout ({} leading, {} \
                 trailing) implies {expected}.",
                self.model_name(),
                layout.leading,
                layout.trailing,
            );
        }

        let per_row = counts
            .iter()
            .enumerate()
            .map(|(i, &n)| Ok(raw.narrow(0, i, 1)?.narrow(1, layout.leading, n)?))
            .collect::<Result<Vec<_>>>()?;
        stack_padded(&per_row)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Emits `sequence.len() + layout.total()` rows, so `embed_residues` has
    /// something consistent to strip.
    ///
    /// Every value in a row is `sequence.len() as f32`, which is never 0 for a
    /// non-empty sequence — so the batching tests can tell a model-produced row
    /// from a zero pad row, and can tell which sequence a row came from.
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
            Ok(
                (Tensor::ones((1usize, rows, 16usize), DType::F32, &Device::Cpu)?
                    * sequence.len() as f64)?,
            )
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

    // ── Batching ──────────────────────────────────────────────────────────

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// The default `embed_batch` pads to the longest row and reports the shape
    /// the doc comment promises.
    #[test]
    fn test_embed_batch_pads_to_longest_sequence() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        let t = runner.embed_batch(&["ACDEFG", "AC"]).unwrap();
        assert_eq!(t.dims(), &[2, 8, 16], "6 residues + BOS + EOS");
    }

    /// Padded rows are exactly zero, and the real rows still carry the model's
    /// own values — the guarantee that lets a caller pool over the sequence
    /// axis without a length vector.
    #[test]
    fn test_embed_batch_zero_pads_short_rows() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        let t = runner.embed_batch(&["ACDEFG", "AC"]).unwrap();
        let short = t.narrow(0, 1, 1).unwrap();
        // "AC" occupies 2 + 2 = 4 of the 8 columns.
        assert!(
            values(&short.narrow(1, 0, 4).unwrap())
                .iter()
                .all(|&v| v == 2.0),
            "real rows should keep the model's output"
        );
        assert!(
            values(&short.narrow(1, 4, 4).unwrap())
                .iter()
                .all(|&v| v == 0.0),
            "padded rows should be exactly zero"
        );
    }

    /// `embed_residues_batch` puts residue `j` at column `j` in **every** row,
    /// which `embed_batch` does not: its trailing special token sits at a
    /// different column per row.
    #[test]
    fn test_embed_residues_batch_aligns_columns_across_rows() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        let t = runner.embed_residues_batch(&["ACDEFG", "AC"]).unwrap();
        assert_eq!(t.dims(), &[2, 6, 16]);

        let short = t.narrow(0, 1, 1).unwrap();
        assert!(
            values(&short.narrow(1, 0, 2).unwrap())
                .iter()
                .all(|&v| v == 2.0),
            "the two real residues of \"AC\" should survive the strip"
        );
        assert!(
            values(&short.narrow(1, 2, 4).unwrap())
                .iter()
                .all(|&v| v == 0.0),
            "columns past a sequence's length should be zero, not its EOS row"
        );
    }

    /// A batch of one is the same thing `embed` returns.
    #[test]
    fn test_embed_batch_of_one_matches_embed() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        let single = runner.embed("ACDE").unwrap();
        let batched = runner.embed_batch(&["ACDE"]).unwrap();
        assert_eq!(single.dims(), batched.dims());
        assert_eq!(values(&single), values(&batched));
    }

    /// An empty batch is a caller bug, not a zero-row tensor.
    #[test]
    fn test_embed_batch_rejects_empty_input() {
        let runner = MockRunner::new(SpecialTokenLayout::BOS_EOS);
        assert!(runner.embed_batch(&[]).is_err());
        assert!(runner.embed_residues_batch(&[]).is_err());
    }

    /// The same guard `embed_residues` has, at batch rank: a runner whose
    /// `embed_batch` disagrees with its declared layout fails loudly.
    #[test]
    fn test_embed_residues_batch_rejects_layout_mismatch() {
        let runner = MockRunner {
            layout: SpecialTokenLayout::BOS_EOS,
            rows_override: Some(4), // claims BOS_EOS but returns bare residues
        };
        let err = runner.embed_residues_batch(&["ACDE"]).unwrap_err();
        assert!(
            err.to_string().contains("special-token layout"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_pad_token_batch_shapes_ids_and_mask() {
        let batch = pad_token_batch(&[vec![0, 5, 7, 2], vec![0, 6, 2]], 1, &Device::Cpu).unwrap();
        assert_eq!(batch.lengths, vec![4, 3]);
        assert_eq!(batch.padded_len(), 4);
        assert_eq!(
            batch.ids.to_vec2::<u32>().unwrap(),
            vec![vec![0, 5, 7, 2], vec![0, 6, 2, 1]],
            "the short row should be right-padded with the pad id"
        );
        assert_eq!(
            batch.mask.to_vec2::<u8>().unwrap(),
            vec![vec![1, 1, 1, 1], vec![1, 1, 1, 0]]
        );
    }

    #[test]
    fn test_pad_token_batch_rejects_empty() {
        assert!(pad_token_batch(&[], 1, &Device::Cpu).is_err());
    }

    #[test]
    fn test_zero_padded_rows_clears_only_pads() {
        let device = Device::Cpu;
        let x = Tensor::ones((1usize, 3usize, 2usize), DType::F32, &device).unwrap();
        let mask = Tensor::new(&[[1u8, 1u8, 0u8]], &device).unwrap();
        let out = zero_padded_rows(&x, &mask).unwrap();
        assert_eq!(values(&out), vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);
    }

    /// The additive form: 0 at real tokens, strongly negative at pads.
    #[test]
    fn test_additive_padding_mask_values() {
        let mask = Tensor::new(&[[1f32, 1f32, 0f32]], &Device::Cpu).unwrap();
        let bias = additive_padding_mask(&mask, DType::F32).unwrap();
        assert_eq!(bias.dims(), &[1, 3]);
        let v = values(&bias);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], 0.0);
        assert!(v[2] <= -1e4, "pad position bias was {}", v[2]);
    }

    /// The masked-score fill value must stay finite in every supported dtype;
    /// F16 saturates past ±65504, which would turn the bias into `-inf`.
    #[test]
    fn test_attention_mask_fill_value_is_dtype_safe() {
        for dtype in [DType::F32, DType::F64, DType::BF16, DType::F16] {
            let v = attention_mask_fill_value(dtype);
            assert!(
                v <= -1e4,
                "fill value {v} for {dtype:?} is not negative enough"
            );
        }
        assert!(
            attention_mask_fill_value(DType::F16) > -65504.0,
            "F16 fill value must stay inside the F16 range"
        );
    }

    #[test]
    fn test_stack_padded_rejects_width_mismatch() {
        let device = Device::Cpu;
        let a = Tensor::zeros((1usize, 2usize, 4usize), DType::F32, &device).unwrap();
        let b = Tensor::zeros((1usize, 2usize, 8usize), DType::F32, &device).unwrap();
        assert!(stack_padded(&[a, b]).is_err());
    }
}
