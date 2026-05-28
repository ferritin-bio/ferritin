//! Unified trait for protein language model runners.
//!
//! `PlmRunner` provides a common interface for sequence embedding across
//! ESM2, AMPLIFY, and ESMC. Downstream code can be generic over the runner
//! type (e.g., for benchmarking or ensemble inference).

use anyhow::Result;
use candle_core::Tensor;

/// Trait implemented by all PLM runner types.
pub trait PlmRunner {
    /// Run a forward pass on `sequence` and return per-residue embeddings.
    ///
    /// Shape: `(1, L, d_model)` where `L` includes any BOS/EOS tokens.
    fn embed(&self, sequence: &str) -> Result<Tensor>;

    /// Model name / identifier string (e.g. "esm2", "amplify", "esmc").
    fn model_name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    struct MockRunner;

    impl PlmRunner for MockRunner {
        fn embed(&self, _sequence: &str) -> Result<Tensor> {
            Tensor::zeros((1usize, 3usize, 16usize), DType::F32, &Device::Cpu)
                .map_err(anyhow::Error::from)
        }

        fn model_name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_mock_runner_model_name() {
        let runner = MockRunner;
        assert_eq!(runner.model_name(), "mock");
    }

    #[test]
    fn test_mock_runner_embed_shape() {
        let runner = MockRunner;
        let tensor = runner.embed("ACDE").unwrap();
        assert_eq!(tensor.dims(), &[1, 3, 16]);
    }
}
