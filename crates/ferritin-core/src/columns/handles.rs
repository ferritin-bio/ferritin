//! Stable numeric handles for atoms, residues, and chains.

macro_rules! define_handle {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        ///
        /// A `#[repr(transparent)]` `u32` dense row index, so it crosses an FFI or
        /// wasm-bindgen boundary as a plain integer. It is meaningful only for the
        /// [`TopologyColumns`](super::TopologyColumns) snapshot it came from (or any
        /// other snapshot of the same `Arc<AtomicHierarchy>`).
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            /// The handle as a `usize` row index.
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl From<$name> for u32 {
            fn from(handle: $name) -> u32 {
                handle.0
            }
        }
    };
}

define_handle!(
    /// Handle to one atom row.
    AtomHandle
);
define_handle!(
    /// Handle to one residue row.
    ResidueHandle
);
define_handle!(
    /// Handle to one chain row.
    ChainHandle
);
