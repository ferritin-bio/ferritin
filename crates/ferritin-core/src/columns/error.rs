//! Error type with stable numeric codes for FFI and wasm-bindgen consumers.

use crate::model::ModelError;
use std::fmt;

/// Category of a [`BoundaryError`].
///
/// The discriminants are part of the boundary contract: foreign callers may
/// switch on [`BoundaryError::code`] and the values will not be renumbered.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryErrorKind {
    /// The model or hierarchy violates a structural invariant.
    InvalidModel = 1,
    /// A handle or row index is outside the exported table.
    IndexOutOfRange = 2,
    /// A frame was paired with a topology snapshot of a different hierarchy.
    StaleTopology = 3,
    /// A count or byte length does not fit the boundary's `u32` index type.
    CapacityExceeded = 4,
}

/// An error raised while exporting columns across the WASM/FFI/GPU boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryError {
    kind: BoundaryErrorKind,
    message: String,
}

impl BoundaryError {
    pub(crate) fn new(kind: BoundaryErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn out_of_range(what: &str, index: u32, len: u32) -> Self {
        Self::new(
            BoundaryErrorKind::IndexOutOfRange,
            format!("{what} {index} is out of range (len={len})"),
        )
    }

    /// The error category.
    pub fn kind(&self) -> BoundaryErrorKind {
        self.kind
    }

    /// Stable numeric code for the error category (see [`BoundaryErrorKind`]).
    pub fn code(&self) -> u32 {
        self.kind as u32
    }

    /// Human-readable description.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BoundaryError {}

impl From<ModelError> for BoundaryError {
    fn from(err: ModelError) -> Self {
        Self::new(BoundaryErrorKind::InvalidModel, err.to_string())
    }
}

/// Convert a count to the boundary's `u32` index type.
pub(crate) fn to_u32(what: &str, value: usize) -> Result<u32, BoundaryError> {
    u32::try_from(value).map_err(|_| {
        BoundaryError::new(
            BoundaryErrorKind::CapacityExceeded,
            format!("{what} ({value}) exceeds u32::MAX"),
        )
    })
}
