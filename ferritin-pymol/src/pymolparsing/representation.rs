//! This module provides a bitmask representation for various types of 3D molecular visualizations.
//!
//! Note: this should probably be moved to it's own crate.
//!
//! It defines a set of flags that can be combined to represent different visual elements
//! in a molecular structure, such as cylinders, spheres, surfaces, and more.
//!
//! The `RepBitmask` struct allows for efficient storage and manipulation of these flags,
//! while the `RepType` enum provides a way to refer to individual representation types.
//!
//! # Examples
//!
//! ```
//! use pseutils::pymolparsing::representation::RepBitmask;
//!
//! let mut reps = RepBitmask::new();
//! reps.insert(RepBitmask::CYL | RepBitmask::SPHERE);
//!
//! assert!(reps.contains(RepBitmask::CYL));
//! assert!(reps.contains(RepBitmask::SPHERE));
//! ```
use bitflags::bitflags;
use serde::{Deserialize, Serialize};

/// First, define an enum for all representation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepType {
    Cyl,
    Sphere,
    Surface,
    Label,
    NonbondedSphere,
    Cartoon,
    Ribbon,
    Line,
    Mesh,
    Dot,
    Dash,
    Nonbonded,
    Cell,
    CGO,
    Callback,
    Extent,
    Slice,
    Angle,
    Dihedral,
    Ellipsoid,
    Volume,
}

bitflags! {
     #[derive(Debug)]
    pub struct RepBitmask: u32 {
        const CYL             = 1 << 0;
        const SPHERE          = 1 << 1;
        const SURFACE         = 1 << 2;
        const LABEL           = 1 << 3;
        const NONBONDED_SPHERE = 1 << 4;
        const CARTOON         = 1 << 5;
        const RIBBON          = 1 << 6;
        const LINE            = 1 << 7;
        const MESH            = 1 << 8;
        const DOT             = 1 << 9;
        const DASH            = 1 << 10;
        const NONBONDED       = 1 << 11;
        const CELL            = 1 << 12;
        const CGO             = 1 << 13;
        const CALLBACK        = 1 << 14;
        const EXTENT          = 1 << 15;
        const SLICE           = 1 << 16;
        const ANGLE           = 1 << 17;
        const DIHEDRAL        = 1 << 18;
        const ELLIPSOID       = 1 << 19;
        const VOLUME          = 1 << 20;

        const REPS_ATOM_MASK = Self::CYL.bits() | Self::SPHERE.bits() | Self::SURFACE.bits() |
            Self::LABEL.bits() | Self::NONBONDED_SPHERE.bits() | Self::CARTOON.bits() | Self::RIBBON.bits() |
            Self::LINE.bits() | Self::MESH.bits() | Self::DOT.bits() | Self::NONBONDED.bits() | Self::ELLIPSOID.bits();

        const REPS_OBJECT_MASK = Self::SURFACE.bits() | Self::MESH.bits() | Self::DOT.bits() |
            Self::CELL.bits() | Self::CGO.bits() | Self::CALLBACK.bits() | Self::EXTENT.bits() | Self::SLICE.bits() |
            Self::ANGLE.bits() | Self::DIHEDRAL.bits() | Self::VOLUME.bits() | Self::DASH.bits();
    }
}

// You can also add methods to RepBitmask if needed
impl RepBitmask {
    pub fn new() -> Self {
        RepBitmask::empty()
    }
    // Function to convert an integer to RepBitmask
    pub fn from_int(value: u32) -> RepBitmask {
        RepBitmask::from_bits_truncate(value)
    }
}

// Custom Serde deserialization
impl<'de> Deserialize<'de> for RepBitmask {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(RepBitmask::from_bits_truncate(bits))
    }
}

// Custom Serde serialization
impl Serialize for RepBitmask {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u32(self.bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rep_bitmask() {
        let mut reps = RepBitmask::new();
        assert!(reps.is_empty());

        reps.insert(RepBitmask::CYL | RepBitmask::SPHERE);
        assert!(reps.contains(RepBitmask::CYL));
        assert!(reps.contains(RepBitmask::SPHERE));
        assert!(!reps.contains(RepBitmask::SURFACE));

        reps.remove(RepBitmask::CYL);
        assert!(!reps.contains(RepBitmask::CYL));
        assert!(reps.contains(RepBitmask::SPHERE));

        assert!(RepBitmask::REPS_ATOM_MASK.contains(RepBitmask::SPHERE));
        assert!(RepBitmask::REPS_OBJECT_MASK.contains(RepBitmask::SURFACE));
    }

    #[test]
    fn test_bitmask_from_int() {
        let int_value = 5; // This represents CYL | SURFACE (1 | 4)
        let bitmask = RepBitmask::from_int(int_value);

        // Check that the correct flags are set
        assert!(bitmask.contains(RepBitmask::CYL));
        assert!(bitmask.contains(RepBitmask::SURFACE));
    }

    #[test]
    fn test_bitmask_from_int_02() {
        // I see this value in the example.pse
        // Binary: 111110110001111111
        let int_value = 2060287;
        let bitmask = RepBitmask::from_int(int_value);

        // Check that the correct flags are set
        assert!(bitmask.contains(RepBitmask::CYL));
        assert!(bitmask.contains(RepBitmask::SURFACE));

        assert!(bitmask.contains(RepBitmask::SPHERE));
        assert!(bitmask.contains(RepBitmask::SURFACE));
        assert!(bitmask.contains(RepBitmask::LABEL));
        assert!(bitmask.contains(RepBitmask::NONBONDED_SPHERE));
        assert!(bitmask.contains(RepBitmask::CARTOON));
        assert!(bitmask.contains(RepBitmask::RIBBON));
        assert!(bitmask.contains(RepBitmask::LINE));
        assert!(bitmask.contains(RepBitmask::MESH));
        assert!(bitmask.contains(RepBitmask::DOT));
        assert!(bitmask.contains(RepBitmask::DASH));
        assert!(bitmask.contains(RepBitmask::NONBONDED));
        assert!(bitmask.contains(RepBitmask::CGO));
        assert!(bitmask.contains(RepBitmask::CALLBACK));
        assert!(bitmask.contains(RepBitmask::SLICE));
        assert!(bitmask.contains(RepBitmask::ANGLE));
        assert!(bitmask.contains(RepBitmask::DIHEDRAL));
        assert!(bitmask.contains(RepBitmask::ELLIPSOID));
        assert!(bitmask.contains(RepBitmask::VOLUME));

        assert!(!bitmask.contains(RepBitmask::EXTENT));
        assert_eq!(bitmask.bits(), 2060287);
    }
}
