pub mod intern;
pub mod ordered_set;
pub mod segmentation;

pub use ordered_set::OrderedSet;
pub use segmentation::Segmentation;

#[cfg(feature = "intern")]
pub use intern::{InternedId, Interner};
