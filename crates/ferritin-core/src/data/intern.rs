//! String interning for efficient storage of repeated string values.
//!
//! This module is only compiled when the `intern` feature is enabled.
//! It provides an [`Interner`] that maps strings to lightweight [`InternedId`]
//! handles, avoiding repeated allocation for identical strings.

#![cfg(feature = "intern")]

use std::collections::HashMap;
use std::sync::Arc;

/// A lightweight handle to an interned string.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct InternedId(u32);

/// String interner — maps unique strings to monotonically assigned [`InternedId`]s.
///
/// Interning the same string twice returns the same [`InternedId`], so identity
/// comparisons between ids are equivalent to string equality.
///
/// # Example
/// ```
/// # #[cfg(feature = "intern")]
/// # {
/// use ferritin_core::data::intern::Interner;
/// let mut interner = Interner::new();
/// let id_a = interner.intern("ALA");
/// let id_b = interner.intern("ALA");
/// assert_eq!(id_a, id_b);
/// # }
/// ```
#[derive(Debug, Default)]
pub struct Interner {
    strings: Vec<Arc<str>>,
    lookup: HashMap<Arc<str>, u32>,
}

impl Interner {
    /// Create a new, empty `Interner`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `s`, returning a stable [`InternedId`].
    ///
    /// If `s` has already been interned, the existing id is returned without
    /// allocating. Otherwise a new id is assigned.
    pub fn intern(&mut self, s: &str) -> InternedId {
        if let Some(&id) = self.lookup.get(s) {
            return InternedId(id);
        }
        let id = self.strings.len() as u32;
        let arc: Arc<str> = Arc::from(s);
        self.strings.push(Arc::clone(&arc));
        self.lookup.insert(arc, id);
        InternedId(id)
    }

    /// Retrieve the string for a previously obtained [`InternedId`].
    ///
    /// # Panics
    /// Panics if `id` was not produced by this `Interner`.
    pub fn get(&self, id: InternedId) -> &str {
        &self.strings[id.0 as usize]
    }

    /// Returns the number of unique strings stored.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Returns `true` if no strings have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

#[cfg(all(test, feature = "intern"))]
mod tests {
    use super::*;

    #[test]
    fn test_interner_basic() {
        let mut interner = Interner::new();
        let id_ala = interner.intern("ALA");
        let id_gly = interner.intern("GLY");
        assert_eq!(interner.get(id_ala), "ALA");
        assert_eq!(interner.get(id_gly), "GLY");
        assert_ne!(id_ala, id_gly);
    }

    #[test]
    fn test_interner_deduplication() {
        let mut interner = Interner::new();
        let id1 = interner.intern("SER");
        let id2 = interner.intern("SER");
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_interner_len() {
        let mut interner = Interner::new();
        assert_eq!(interner.len(), 0);
        assert!(interner.is_empty());

        interner.intern("A");
        assert_eq!(interner.len(), 1);

        interner.intern("B");
        assert_eq!(interner.len(), 2);

        // Duplicates do not increase len
        interner.intern("A");
        assert_eq!(interner.len(), 2);
    }
}
