//! Per-model coordinate data that varies across trajectory frames.
//!
//! [`AtomicConformation`] holds everything that differs between frames: Cartesian
//! coordinates, occupancy, B-factors, and per-atom confidence scores. Topology
//! (connectivity, sequence, etc.) lives in [`super::hierarchy::AtomicHierarchy`].

use super::ModelError;

/// Per-model coordinate data — varies across trajectory frames.
///
/// Coordinates are stored as struct-of-arrays (SoA): `x[i]`, `y[i]`, `z[i]`
/// for atom `i`. This layout is cache-friendly for common operations that
/// process one coordinate component at a time (e.g., computing pairwise
/// distances, centroid calculations).
///
/// # Host-side only
///
/// Coordinates are plain `Vec<f32>` — no GPU tensors or device types.
/// ferritin-core is intentionally framework-agnostic.
#[derive(Clone, Debug)]
pub struct AtomicConformation {
    /// X coordinates (Å), one per atom.
    pub x: Vec<f32>,
    /// Y coordinates (Å), one per atom.
    pub y: Vec<f32>,
    /// Z coordinates (Å), one per atom.
    pub z: Vec<f32>,
    /// Occupancy for each atom (0.0–1.0), or `None` if not recorded.
    pub occupancy: Option<Vec<f32>>,
    /// Isotropic B-factor (temperature factor) for each atom, or `None`.
    /// For predicted structures use [`confidence`] instead to avoid semantic ambiguity.
    pub b_iso: Option<Vec<f32>>,
    /// Per-atom confidence score (e.g. pLDDT from AlphaFold / ESM3), or `None`.
    ///
    /// Kept separate from `b_iso` so callers can distinguish experimental
    /// B-factors from model confidence without inspecting metadata.
    pub confidence: Option<Vec<f32>>,
}

impl AtomicConformation {
    /// Validate all per-atom columns against `n_atoms`.
    pub fn validate(&self, n_atoms: usize) -> Result<(), ModelError> {
        for (name, len) in [
            ("x", self.x.len()),
            ("y", self.y.len()),
            ("z", self.z.len()),
        ] {
            if len != n_atoms {
                return Err(ModelError::new(format!(
                    "conformation.{name} has length {len}, expected {n_atoms}"
                )));
            }
        }
        for (name, values) in [
            ("occupancy", self.occupancy.as_ref()),
            ("b_iso", self.b_iso.as_ref()),
            ("confidence", self.confidence.as_ref()),
        ] {
            if let Some(values) = values
                && values.len() != n_atoms
            {
                return Err(ModelError::new(format!(
                    "conformation.{name} has length {}, expected {n_atoms}",
                    values.len()
                )));
            }
        }
        Ok(())
    }
    /// Number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.x.len()
    }

    /// Get `[x, y, z]` for atom `i`.
    ///
    /// # Panics
    /// Panics if `i >= n_atoms()`.
    pub fn coord(&self, i: usize) -> [f32; 3] {
        [self.x[i], self.y[i], self.z[i]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformation_coord() {
        let conf = AtomicConformation {
            x: vec![1.0, 4.0, 7.0],
            y: vec![2.0, 5.0, 8.0],
            z: vec![3.0, 6.0, 9.0],
            occupancy: Some(vec![1.0, 0.5, 1.0]),
            b_iso: None,
            confidence: Some(vec![90.0, 85.0, 92.0]),
        };

        assert_eq!(conf.n_atoms(), 3);
        assert_eq!(conf.coord(0), [1.0, 2.0, 3.0]);
        assert_eq!(conf.coord(1), [4.0, 5.0, 6.0]);
        assert_eq!(conf.coord(2), [7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_conformation_optional_fields() {
        let conf = AtomicConformation {
            x: vec![0.0],
            y: vec![0.0],
            z: vec![0.0],
            occupancy: None,
            b_iso: None,
            confidence: None,
        };
        assert_eq!(conf.n_atoms(), 1);
        assert_eq!(conf.coord(0), [0.0, 0.0, 0.0]);
        assert!(conf.occupancy.is_none());
        assert!(conf.b_iso.is_none());
        assert!(conf.confidence.is_none());
    }

    #[test]
    fn test_conformation_validation_rejects_mismatched_columns() {
        let conf = AtomicConformation {
            x: vec![0.0],
            y: vec![],
            z: vec![0.0],
            occupancy: None,
            b_iso: None,
            confidence: None,
        };
        assert_eq!(
            conf.validate(1).unwrap_err().to_string(),
            "conformation.y has length 0, expected 1"
        );
    }
}
