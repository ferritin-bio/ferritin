//! Trajectory layer (Layer 2): multi-model structure access.
//!
//! This module provides the [`Trajectory`] trait for accessing ordered
//! collections of [`Model`] frames, plus two implementations:
//!
//! - [`ArrayTrajectory`]: eager — stores all frames as `Vec<Model>`.
//! - [`ModelCoordsTrajectory`]: lazy — stores one topology + `Coordinates`.
//!
//! All implementations are object-safe (`dyn Trajectory` works).

use crate::model::Model;
use std::borrow::Cow;

pub mod array_trajectory;
pub mod coordinates;
pub mod model_coords_trajectory;

pub use array_trajectory::ArrayTrajectory;
pub use coordinates::{Coordinates, Frame, UnitCell};
pub use model_coords_trajectory::ModelCoordsTrajectory;

/// Multi-model structure access. Object-safe: usable as `dyn Trajectory`.
///
/// Implementors expose an ordered sequence of [`Model`] frames. Frames may be
/// returned as borrowed references (`Cow::Borrowed`) when already in memory, or
/// as owned values (`Cow::Owned`) when constructed lazily.
pub trait Trajectory {
    /// Total number of frames in the trajectory.
    fn frame_count(&self) -> usize;

    /// A representative frame (typically the first), useful when callers need
    /// topology access without specifying a frame index.
    fn representative(&self) -> &Model;

    /// Returns the frame at `index`, either borrowed or constructed on demand.
    ///
    /// # Panics
    /// Panics if `index >= frame_count()`.
    fn frame(&self, index: usize) -> Cow<'_, Model>;
}

// Compile-time object-safety check — must compile:
fn _assert_object_safe(_: &dyn Trajectory) {}
