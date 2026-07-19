//! Coordinate data for trajectory frames.
//!
//! [`Frame`] holds a single snapshot of coordinates (topology-free), and
//! [`Coordinates`] is an ordered collection of frames for trajectory storage.

use crate::model::ModelError;

/// Unit cell parameters for periodic boundary condition systems.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitCell {
    /// Length of the a-axis in Å.
    pub a: f32,
    /// Length of the b-axis in Å.
    pub b: f32,
    /// Length of the c-axis in Å.
    pub c: f32,
    /// Angle between b and c axes in degrees.
    pub alpha: f32,
    /// Angle between a and c axes in degrees.
    pub beta: f32,
    /// Angle between a and b axes in degrees.
    pub gamma: f32,
}

/// Single frame of coordinates — topology-free.
///
/// Stores Cartesian coordinates as parallel arrays (SoA layout). Optional
/// fields are `None` when not relevant (e.g. non-periodic systems have no
/// `cell`; single-structure files have no `time`).
#[derive(Clone, Debug)]
pub struct Frame {
    /// X coordinates (Å), one per atom.
    pub x: Vec<f32>,
    /// Y coordinates (Å), one per atom.
    pub y: Vec<f32>,
    /// Z coordinates (Å), one per atom.
    pub z: Vec<f32>,
    /// Periodic box parameters, or `None` for non-periodic systems.
    pub cell: Option<UnitCell>,
    /// Simulation time in picoseconds, or `None` if not applicable.
    pub time: Option<f64>,
}

/// Ordered collection of coordinate frames.
///
/// Frames are indexed by position. Use [`Coordinates::frame`] for random access.
#[derive(Clone, Debug)]
pub struct Coordinates {
    frames: Vec<Frame>,
}

impl Coordinates {
    /// Construct a validated coordinate collection.
    pub fn try_new(frames: Vec<Frame>) -> Result<Self, ModelError> {
        let expected = frames.first().map_or(0, |frame| frame.x.len());
        for (frame_index, frame) in frames.iter().enumerate() {
            for (axis, len) in [
                ("x", frame.x.len()),
                ("y", frame.y.len()),
                ("z", frame.z.len()),
            ] {
                if len != expected {
                    return Err(ModelError::new(format!(
                        "coordinates frame {frame_index}.{axis} has length {len}, expected {expected}"
                    )));
                }
            }
        }
        Ok(Self { frames })
    }

    /// Construct from a `Vec<Frame>`.
    ///
    /// # Panics
    /// Panics if coordinate column lengths differ. Use [`Coordinates::try_new`]
    /// to report invalid input without panicking.
    pub fn new(frames: Vec<Frame>) -> Self {
        Self::try_new(frames).expect("invalid coordinate frames")
    }

    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if there are no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Returns a reference to the frame at `index`.
    ///
    /// # Panics
    /// Panics if `index >= len()`.
    pub fn frame(&self, index: usize) -> &Frame {
        &self.frames[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinates_len() {
        let frames: Vec<Frame> = (0..3)
            .map(|i| Frame {
                x: vec![i as f32],
                y: vec![0.0],
                z: vec![0.0],
                cell: None,
                time: None,
            })
            .collect();
        let coords = Coordinates::new(frames);
        assert_eq!(coords.len(), 3);
        assert!(!coords.is_empty());
    }

    #[test]
    fn test_frame_fields() {
        let cell = UnitCell {
            a: 10.0,
            b: 20.0,
            c: 30.0,
            alpha: 90.0,
            beta: 90.0,
            gamma: 120.0,
        };
        let frame = Frame {
            x: vec![1.0, 2.0],
            y: vec![3.0, 4.0],
            z: vec![5.0, 6.0],
            cell: Some(cell.clone()),
            time: Some(0.5),
        };

        assert_eq!(frame.x, vec![1.0, 2.0]);
        assert_eq!(frame.y, vec![3.0, 4.0]);
        assert_eq!(frame.z, vec![5.0, 6.0]);
        assert_eq!(frame.time, Some(0.5));
        let c = frame.cell.as_ref().unwrap();
        assert_eq!(c.a, 10.0);
        assert_eq!(c.b, 20.0);
        assert_eq!(c.c, 30.0);
        assert_eq!(c.alpha, 90.0);
        assert_eq!(c.beta, 90.0);
        assert_eq!(c.gamma, 120.0);
    }

    #[test]
    fn test_coordinates_validation_rejects_frame_size_mismatch() {
        let frames = vec![
            Frame {
                x: vec![0.0],
                y: vec![0.0],
                z: vec![0.0],
                cell: None,
                time: None,
            },
            Frame {
                x: vec![0.0, 1.0],
                y: vec![0.0, 1.0],
                z: vec![0.0, 1.0],
                cell: None,
                time: None,
            },
        ];
        assert!(Coordinates::try_new(frames).is_err());
    }
}
