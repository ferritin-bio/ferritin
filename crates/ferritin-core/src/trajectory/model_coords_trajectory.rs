//! Lazy trajectory: one topology + many coordinate frames.
//!
//! [`ModelCoordsTrajectory`] constructs a [`Model`] on demand from a stored
//! topology and a frame-specific coordinate snapshot. This is more memory-
//! efficient than [`super::ArrayTrajectory`] when storing many frames with the
//! same connectivity.

use super::{Coordinates, Trajectory};
use crate::model::{AtomicConformation, Model, ModelError};
use std::borrow::Cow;
use std::sync::Arc;

/// Lazy trajectory: one topology + many coordinate frames.
///
/// Stores a single [`Model`] as the topology reference and a [`Coordinates`]
/// collection. `frame()` constructs a fresh `Model` on each call — it is
/// `Cow::Owned` — avoiding the need to materialise all frames simultaneously.
pub struct ModelCoordsTrajectory {
    topology: Model,
    coords: Coordinates,
}

impl ModelCoordsTrajectory {
    /// Construct a validated lazy trajectory.
    pub fn try_new(topology: Model, coords: Coordinates) -> Result<Self, ModelError> {
        topology.hierarchy.validate()?;
        topology.conformation.validate(topology.n_atoms())?;
        for index in 0..coords.len() {
            let frame = coords.frame(index);
            if frame.x.len() != topology.n_atoms() {
                return Err(ModelError::new(format!(
                    "coordinates frame {index} has {} atoms, expected {}",
                    frame.x.len(),
                    topology.n_atoms()
                )));
            }
        }
        Ok(Self { topology, coords })
    }

    /// Construct from a topology model and coordinate frames.
    pub fn new(topology: Model, coords: Coordinates) -> Self {
        Self::try_new(topology, coords).expect("invalid model-coordinate trajectory")
    }
}

impl Trajectory for ModelCoordsTrajectory {
    fn frame_count(&self) -> usize {
        self.coords.len()
    }

    fn representative(&self) -> &Model {
        &self.topology
    }

    fn frame(&self, index: usize) -> Cow<'_, Model> {
        let f = self.coords.frame(index);
        let conformation = AtomicConformation {
            x: f.x.clone(),
            y: f.y.clone(),
            z: f.z.clone(),
            occupancy: None,
            b_iso: None,
            confidence: None,
        };
        Cow::Owned(Model::new(
            Arc::clone(&self.topology.hierarchy),
            conformation,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::coordinates::{Coordinates, Frame};
    use super::*;
    use crate::data::Segmentation;
    use crate::info::elements::Element;
    use crate::model::tables::{AtomsTable, ChainsTable, ResidueGroup, ResiduesTable};
    use crate::model::{AtomicConformation, AtomicHierarchy, Bonds};
    use std::borrow::Cow;

    fn make_simple_hierarchy(n_residues: usize) -> Arc<AtomicHierarchy> {
        let n_atoms = n_residues;
        let atoms = AtomsTable {
            atom_name: (0..n_atoms).map(|_| "CA".to_string()).collect(),
            auth_atom_name: (0..n_atoms).map(|_| "CA".to_string()).collect(),
            element: vec![Element::C; n_atoms],
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };
        let residues = ResiduesTable {
            comp_id: (0..n_residues).map(|i| format!("R{}", i)).collect(),
            auth_comp_id: (0..n_residues).map(|i| format!("R{}", i)).collect(),
            label_seq_id: (0..n_residues as i32).collect(),
            auth_seq_id: (1..=n_residues as i32).collect(),
            ins_code: vec![None; n_residues],
            group: vec![ResidueGroup::Polymer; n_residues],
        };
        let chains = ChainsTable {
            label_asym_id: vec!["A".into()],
            auth_asym_id: vec!["A".into()],
            entity_id: vec!["1".into()],
        };
        let atom_offsets: Vec<u32> = (0..=n_residues as u32).collect();
        let atom_to_residue = Segmentation::from_offsets(atom_offsets);
        let residue_to_chain = Segmentation::from_offsets(vec![0, n_residues as u32]);
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);
        Arc::new(AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        })
    }

    fn make_conformation(n: usize) -> AtomicConformation {
        AtomicConformation {
            x: (0..n).map(|i| i as f32).collect(),
            y: (0..n).map(|_| 0.0).collect(),
            z: (0..n).map(|_| 0.0).collect(),
            occupancy: None,
            b_iso: None,
            confidence: None,
        }
    }

    fn make_coords(n_frames: usize, n_atoms: usize) -> Coordinates {
        let frames: Vec<Frame> = (0..n_frames)
            .map(|i| Frame {
                x: (0..n_atoms).map(|a| (i * n_atoms + a) as f32).collect(),
                y: vec![0.0; n_atoms],
                z: vec![0.0; n_atoms],
                cell: None,
                time: Some(i as f64 * 0.001),
            })
            .collect();
        Coordinates::new(frames)
    }

    #[test]
    fn test_model_coords_frame_count() {
        let h = make_simple_hierarchy(3);
        let topology = Model::new(Arc::clone(&h), make_conformation(3));
        let coords = make_coords(5, 3);
        let traj = ModelCoordsTrajectory::new(topology, coords);
        assert_eq!(traj.frame_count(), 5);
    }

    #[test]
    fn test_model_coords_lazy() {
        let h = make_simple_hierarchy(3);
        let topology = Model::new(Arc::clone(&h), make_conformation(3));
        let coords = make_coords(4, 3);
        let traj = ModelCoordsTrajectory::new(topology, coords);
        for i in 0..4 {
            let f = traj.frame(i);
            assert!(
                matches!(f, Cow::Owned(_)),
                "frame({}) should be Cow::Owned",
                i
            );
        }
    }

    #[test]
    fn test_model_coords_shared_hierarchy() {
        let h = make_simple_hierarchy(2);
        let topology = Model::new(Arc::clone(&h), make_conformation(2));
        let coords = make_coords(3, 2);
        let traj = ModelCoordsTrajectory::new(topology, coords);

        // The Arc in each generated frame should be identical to the topology's Arc
        for i in 0..3 {
            let f = traj.frame(i);
            assert!(
                Arc::ptr_eq(&traj.topology.hierarchy, &f.hierarchy),
                "frame({}) hierarchy must share the same Arc as topology",
                i
            );
        }
    }
}
