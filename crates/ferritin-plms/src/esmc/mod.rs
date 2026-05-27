//! ESM-C / ESM-3 model family from EvolutionaryScale.
//!
//! This module ports the [EvolutionaryScale Python SDK](https://github.com/evolutionaryscale/esm)
//! for the **ESM-C** (esmc-300m, esmc-600m) and **ESM-3** (esm3-sm-open-v1) model families.
//!
//! These are **multimodal** models (sequence + structure + function tracks) with geometric
//! attention and codebook quantization. Weights are gated-access on HuggingFace and require
//! accepting the EvolutionaryScale license before downloading.
//!
//! HuggingFace repos:
//! - <https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12>
//! - <https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12>
//!
//! **Note**: This is architecturally distinct from `esm2` (Facebook/Meta ESM-2), which is a
//! sequence-only BERT-style encoder available publicly. The two modules share the ESM name and
//! amino acid alphabet but are otherwise unrelated.
//!
//! # Status
//! The core `esmc` model and its layers compile and load weights. The `sdk/`, `pretrained.rs`,
//! and parts of `utils/` are aspirational WIP stubs (commented out of this module).

pub mod layers;
pub mod models;
pub mod pretrained;
// pub mod sdk;         // WIP: uses undefined abc crate and protein_chain/complex types
pub mod tokenization;
pub mod utils;
