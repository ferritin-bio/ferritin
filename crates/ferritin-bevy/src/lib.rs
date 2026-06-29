//! Bevy-based protein structure rendering for ferritin.

pub mod colors;
pub mod plugin;
pub mod structure;

pub use colors::ColorScheme;
pub use plugin::{LoadProteinEvent, StructurePlugin, StructureSettings};
pub use structure::{RenderOptions, Structure};

#[cfg(feature = "rerun")]
pub mod conversions;
#[cfg(feature = "rerun")]
pub use conversions::ToRerun;
