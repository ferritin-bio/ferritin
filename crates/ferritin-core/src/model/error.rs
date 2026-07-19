//! Validation errors for molecular model construction.

use std::fmt;

/// An invariant violation encountered while constructing a molecular model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelError {
    message: String,
}

impl ModelError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ModelError {}
