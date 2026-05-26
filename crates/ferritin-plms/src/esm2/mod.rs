//! ESM-2 protein language model from Facebook/Meta Research.
//!
//! This module implements **ESM-2** (`EsmForMaskedLM`), a BERT-style masked language model
//! trained on UniRef50. Weights are publicly available on HuggingFace without any license gate.
//!
//! HuggingFace repos (facebook/esm2_t{6,12,30,33,36,48}_UR50D):
//! - <https://huggingface.co/facebook/esm2_t6_8M_UR50D>
//! - <https://huggingface.co/facebook/esm2_t12_35M_UR50D>
//! - <https://huggingface.co/facebook/esm2_t30_150M_UR50D>
//! - <https://huggingface.co/facebook/esm2_t33_650M_UR50D>
//!
//! Features: sequence embeddings, per-residue logits, contact head predictions.
//! Loads weights from HuggingFace safetensors format via `ESM2Runner`.
//!
//! **Note**: Architecturally distinct from `esmc` (EvolutionaryScale ESM-C/ESM-3), which is
//! multimodal and gated. ESM-2 is sequence-only and publicly accessible.

pub mod esm2;
pub mod esm2_runner;
