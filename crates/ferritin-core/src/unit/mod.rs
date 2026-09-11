//! Zero-copy view layer for subset selection on [`Model`][crate::model::Model].
//!
//! A [`Unit`] represents a selection of atoms within a
//! [`Model`][crate::model::Model] without copying the underlying data. It stores
//! an [`OrderedSet`][crate::data::OrderedSet] of atom indices and provides
//! lazy iterators for accessing coordinates and performing set operations.

mod unit;

pub use unit::Unit;
