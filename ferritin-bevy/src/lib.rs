//! # Ferritin-Bevy
//!
//! A plugin for visualizing protein structures from ferritin-core using the Bevy game engine.
//!
//! This crate provides visualization capabilities for protein structure data models defined in
//! ferritin-core. It leverages Bevy's powerful rendering engine to create interactive 3D
//! visualizations of protein structures.
//!
//! ## Features
//! - 3D visualization of protein structures
//! - Configurable coloring schemes
//! - Interactive camera controls
//! - Support for multiple visualization styles
//!
pub mod colors;
pub mod plugin;
pub mod structure;
pub use colors::ColorScheme;
pub use plugin::{StructurePlugin, StructureSettings};
pub use structure::{RenderOptions, Structure};
