//! ESM-3 multimodal protein language model from EvolutionaryScale.
//!
//! ESM3 is a generative model conditioned on sequence, structure, and function tracks.
//! It extends ESMC with a VQ-VAE structure encoder/decoder, multi-track tokenization
//! (SS8, SASA, function annotations), and a unified generative trunk.
//!
//! Reference implementation: <https://github.com/evolutionaryscale/esm>
//! HuggingFace weights (gated): <https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1>
//!
//! # Status
//! Scaffold only — sub-modules are stubs pending implementation of ferritin-4kf,
//! ferritin-bz0, ferritin-iln, ferritin-zcf, and ferritin-fdz.

pub mod layers;
pub mod models;
pub mod pretrained;
pub mod tokenization;
pub mod utils;
