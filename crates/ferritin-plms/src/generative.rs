//! Shared interface for autoregressive protein language models.
//!
//! Generative models deliberately do not implement [`PlmRunner`][crate::PlmRunner]:
//! their primary contract is sequence likelihood, not residue embeddings. This
//! trait keeps that boundary explicit while the embedding conformance suite is
//! still specialized to masked/encoder-only models.

use crate::plm_runner::ModelMetadata;
use anyhow::{Result, bail};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::ops::log_softmax;

/// Interface for deterministic, autoregressive sequence scoring.
pub trait GenerativeModel {
    /// Stable model identifier used in diagnostics.
    fn model_name(&self) -> &str;

    /// Architecture dimensions for the loaded model.
    fn metadata(&self) -> ModelMetadata;

    /// Device on which the model's weights live.
    fn device(&self) -> &candle_core::Device;

    /// Encode a protein sequence, including the model's terminal tokens.
    fn token_ids(&self, sequence: &str) -> Result<Vec<u32>>;

    /// Return causal-LM logits for the encoded sequence.
    ///
    /// The result is shaped `(1, tokens, vocab_size)`. Row `i` predicts token
    /// `i + 1`, matching the shift used by the reference PyTorch model.
    fn logits(&self, sequence: &str) -> Result<Tensor>;

    /// Sum the log-probability assigned to the protein residues in `sequence`.
    ///
    /// Model-specific terminal tokens are included in the forward pass so the
    /// context matches the reference implementation, but the returned score
    /// excludes the leading terminal and the trailing terminal target.
    fn log_likelihood(&self, sequence: &str) -> Result<f32> {
        let ids = self.token_ids(sequence)?;
        if ids.len() < 3 {
            bail!(
                "{}: sequence must contain at least one residue",
                self.model_name()
            );
        }

        let logits = self.logits(sequence)?.to_dtype(candle_core::DType::F32)?;
        if logits.dims() != [1, ids.len(), self.metadata().vocab_size] {
            bail!(
                "{}: logits shape {:?} does not match {} encoded tokens and vocab size {}",
                self.model_name(),
                logits.dims(),
                ids.len(),
                self.metadata().vocab_size
            );
        }

        // The first row predicts the first residue, and the last row predicts
        // the terminal token. The latter is intentionally excluded from the
        // protein score, just as the upstream ProGen likelihood script does.
        let residue_count = ids.len() - 2;
        let shifted = logits.i((0, 0..residue_count, ..))?;
        let log_probs = log_softmax(&shifted, D::Minus1)?;
        let targets = Tensor::from_vec(
            ids[1..=residue_count].to_vec(),
            (residue_count, 1),
            logits.device(),
        )?;
        let selected = log_probs.gather(&targets, D::Minus1)?.squeeze(D::Minus1)?;
        Ok(selected.sum_all()?.to_scalar::<f32>()?)
    }

    /// Mean per-residue log-probability, useful for comparing proteins of
    /// different lengths without changing the canonical summed score.
    fn mean_log_likelihood(&self, sequence: &str) -> Result<f32> {
        let n = sequence.chars().count();
        if n == 0 {
            bail!(
                "{}: sequence must contain at least one residue",
                self.model_name()
            );
        }
        Ok(self.log_likelihood(sequence)? / n as f32)
    }
}
