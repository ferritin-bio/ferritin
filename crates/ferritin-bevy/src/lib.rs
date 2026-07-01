//! Bevy-based protein structure rendering for ferritin.

pub mod colors;
pub mod mvs_executor;
pub mod plugin;
pub mod selection;
pub mod structure;

pub use colors::ColorScheme;
pub use mvs_executor::{
    LoadMvsEvent, MvsEntity, MvsError, MvsLabel, MvsPlugin, MvsStateResource, OrbitCamera,
};
pub use plugin::{LoadProteinEvent, StructurePlugin, StructureSettings};
pub use structure::{RenderOptions, Structure};

#[cfg(feature = "rerun")]
pub mod conversions;
#[cfg(feature = "rerun")]
pub use conversions::ToRerun;
