//! Eager trajectory: all frames held in memory as `Vec<Model>`.

use super::Trajectory;
use crate::model::Model;
use std::borrow::Cow;
use std::sync::Arc;

/// Error type for [`ArrayTrajectory`] construction.
#[derive(Debug, Clone)]
pub enum TrajectoryError {
    /// Trajectory must contain at least one frame.
    Empty,
    /// All frames must share the same `Arc<AtomicHierarchy>`.
    MixedTopologies,
    /// A frame violates model topology or conformation invariants.
    InvalidModel,
}

impl std::fmt::Display for TrajectoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrajectoryError::Empty => write!(f, "trajectory must contain at least one frame"),
            TrajectoryError::MixedTopologies => {
                write!(f, "all frames must share the same Arc<AtomicHierarchy>")
            }
            TrajectoryError::InvalidModel => write!(f, "trajectory contains an invalid model"),
        }
    }
}

impl std::error::Error for TrajectoryError {}

/// Eager trajectory: all models held in memory.
///
/// All frames **must** share the same `Arc<AtomicHierarchy>` (verified on
/// construction). This invariant ensures that topology does not vary across
/// frames, which is required for correct trajectory analysis.
pub struct ArrayTrajectory {
    models: Vec<Model>,
}

impl ArrayTrajectory {
    /// Construct from a `Vec<Model>`.
    ///
    /// Returns `Err(TrajectoryError::Empty)` if `models` is empty.
    /// Returns `Err(TrajectoryError::MixedTopologies)` if any two models have
    /// different `Arc<AtomicHierarchy>` pointers.
    pub fn new(models: Vec<Model>) -> Result<Self, TrajectoryError> {
        if models.is_empty() {
            return Err(TrajectoryError::Empty);
        }
        for model in &models {
            model
                .hierarchy
                .validate()
                .map_err(|_| TrajectoryError::InvalidModel)?;
            model
                .conformation
                .validate(model.n_atoms())
                .map_err(|_| TrajectoryError::InvalidModel)?;
        }
        let first = Arc::as_ptr(&models[0].hierarchy);
        for m in &models[1..] {
            if Arc::as_ptr(&m.hierarchy) != first {
                return Err(TrajectoryError::MixedTopologies);
            }
        }
        Ok(Self { models })
    }

    /// Returns a slice of all models in the trajectory.
    pub fn models(&self) -> &[Model] {
        &self.models
    }
}

impl Trajectory for ArrayTrajectory {
    fn frame_count(&self) -> usize {
        self.models.len()
    }

    fn representative(&self) -> &Model {
        &self.models[0]
    }

    fn frame(&self, index: usize) -> Cow<'_, Model> {
        Cow::Borrowed(&self.models[index])
    }
}

#[cfg(test)]
mod tests {
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

    fn make_conformation(n: usize, offset: f32) -> AtomicConformation {
        AtomicConformation {
            x: (0..n).map(|i| i as f32 + offset).collect(),
            y: (0..n).map(|i| i as f32 * 10.0).collect(),
            z: vec![0.0; n],
            occupancy: None,
            b_iso: None,
            confidence: None,
        }
    }

    #[test]
    fn test_array_trajectory_frame_count() {
        let hierarchy = make_simple_hierarchy(3);
        let models: Vec<Model> = (0..3)
            .map(|i| {
                Model::new(
                    Arc::clone(&hierarchy),
                    make_conformation(3, i as f32 * 100.0),
                )
            })
            .collect();
        let traj = ArrayTrajectory::new(models).unwrap();
        assert_eq!(traj.frame_count(), 3);
    }

    #[test]
    fn test_array_trajectory_borrowed() {
        let hierarchy = make_simple_hierarchy(2);
        let models: Vec<Model> = (0..3)
            .map(|i| {
                Model::new(
                    Arc::clone(&hierarchy),
                    make_conformation(2, i as f32 * 10.0),
                )
            })
            .collect();
        let traj = ArrayTrajectory::new(models).unwrap();
        for i in 0..3 {
            let f = traj.frame(i);
            assert!(
                matches!(f, Cow::Borrowed(_)),
                "frame({}) should be Cow::Borrowed",
                i
            );
        }
    }

    #[test]
    fn test_array_trajectory_arc_shared() {
        let hierarchy = make_simple_hierarchy(4);
        let models: Vec<Model> = (0..3)
            .map(|i| Model::new(Arc::clone(&hierarchy), make_conformation(4, i as f32 * 5.0)))
            .collect();
        let traj = ArrayTrajectory::new(models).unwrap();
        let h0 = &traj.frame(0).hierarchy;
        let h2 = &traj.frame(2).hierarchy;
        assert!(
            Arc::ptr_eq(h0, h2),
            "frame(0) and frame(2) must share the same Arc<AtomicHierarchy>"
        );
    }

    #[test]
    fn test_array_trajectory_empty_error() {
        let result = ArrayTrajectory::new(vec![]);
        assert!(matches!(result, Err(TrajectoryError::Empty)));
    }

    #[test]
    fn test_array_trajectory_mixed_topologies_error() {
        let h1 = make_simple_hierarchy(3);
        let h2 = make_simple_hierarchy(3); // different Arc allocation
        let models = vec![
            Model::new(Arc::clone(&h1), make_conformation(3, 0.0)),
            Model::new(Arc::clone(&h2), make_conformation(3, 1.0)),
        ];
        let result = ArrayTrajectory::new(models);
        assert!(matches!(result, Err(TrajectoryError::MixedTopologies)));
    }

    #[test]
    fn test_array_trajectory_object_safe() {
        let hierarchy = make_simple_hierarchy(2);
        let models: Vec<Model> = (0..2)
            .map(|i| Model::new(Arc::clone(&hierarchy), make_conformation(2, i as f32)))
            .collect();
        let traj = ArrayTrajectory::new(models).unwrap();
        let boxed: Box<dyn Trajectory> = Box::new(traj);
        assert_eq!(boxed.frame_count(), 2);
    }
}
