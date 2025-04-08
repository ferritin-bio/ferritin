//! # Ferritin-Structure-Mesh
//!
//! A library for visualizing protein structures from ferritin-core using various rendering backends.
//!
//! This crate provides visualization capabilities for protein structure data models defined in
//! ferritin-core. It can use Bevy's rendering engine and/or Rerun for visualization depending on
//! which features are enabled.
//!
//! ## Features
//! - 3D visualization of protein structures
//! - Configurable coloring schemes
//! - Support for multiple visualization styles
//! - Optional integrations with Bevy and Rerun
//!
pub mod colors;
pub mod conversions;
pub mod plugin;
pub mod structure;
pub use colors::ColorScheme;
pub use conversions::ToRerun;
pub use plugin::{StructurePlugin, StructureSettings};
pub use structure::{RenderOptions, Structure};
