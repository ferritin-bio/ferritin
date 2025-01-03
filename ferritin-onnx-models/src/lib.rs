//! Ferritin Onnx Models
//!
//! This crate provides easy access to various ONNX models for protein and ligand prediction.
//! The models are downloaded from HuggingFace and run using ONNX Runtime.
//! Currently supports ESM2 and LigandMPNN models.
//!
pub mod models;
pub mod utilities;

// pub use models::amplify::{AMPLIFYModels, AMPLIFY};
pub use models::esm2::{ESM2Models, ESM2};
pub use models::ligandmpnn::{LigandMPNN, ModelType};
pub use utilities::{ndarray_to_tensor_f32, tensor_to_ndarray_f32, tensor_to_ndarray_i64};
