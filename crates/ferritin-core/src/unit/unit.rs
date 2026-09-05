//! Zero-copy view into a subset of a [`Model`].

use crate::data::OrderedSet;
use crate::model::{Model, ModelError, SymmetryOperator};

/// Zero-copy view into a subset of a Model.
///
/// `Unit` holds a reference to a `Model` and an `OrderedSet` of selected atom indices.
/// All operations are lazy — iterators do not allocate, and set operations produce
/// new `Unit`s without copying coordinate data.
#[derive(Clone)]
pub struct Unit<'a> {
    model: &'a Model,
    atoms: OrderedSet,
    operator: SymmetryOperator,
}

impl<'a> Unit<'a> {
    /// Create a `Unit` from a boolean mask.
    ///
    /// The mask must have the same length as `model.n_atoms()`.
    /// Only atoms where `mask[i] == true` are included.
    ///
    /// # Panics
    /// Panics if `mask.len() != model.n_atoms()`.
    pub fn from_mask(model: &'a Model, mask: &[bool]) -> Self {
        assert_eq!(
            mask.len(),
            model.n_atoms(),
            "mask length must equal number of atoms"
        );
        let indices: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i as u32) } else { None })
            .collect();

        let atoms = if indices.is_empty() {
            OrderedSet::interval(0, 0)
        } else {
            OrderedSet::from_sorted(indices)
        };

        Self {
            model,
            atoms,
            operator: SymmetryOperator::identity(),
        }
    }

    /// Create a `Unit` containing all atoms in the model.
    pub fn all(model: &'a Model) -> Self {
        let n = model.n_atoms() as u32;
        Self {
            model,
            atoms: OrderedSet::interval(0, n),
            operator: SymmetryOperator::identity(),
        }
    }

    /// Create a `Unit` from pre-computed indices.
    pub fn from_indices(model: &'a Model, indices: OrderedSet) -> Self {
        Self::try_from_indices(model, indices).expect("unit atom index is out of range")
    }

    /// Create a `Unit` from validated pre-computed indices.
    pub fn try_from_indices(model: &'a Model, indices: OrderedSet) -> Result<Self, ModelError> {
        if let Some(index) = indices
            .iter()
            .find(|&index| index as usize >= model.n_atoms())
        {
            return Err(ModelError::new(format!(
                "unit atom index {index} is out of range for {} atoms",
                model.n_atoms()
            )));
        }
        Ok(Self {
            model,
            atoms: indices,
            operator: SymmetryOperator::identity(),
        })
    }

    /// Create a validated unit carrying an affine symmetry operator.
    pub fn try_from_indices_with_operator(
        model: &'a Model,
        indices: OrderedSet,
        operator: SymmetryOperator,
    ) -> Result<Self, ModelError> {
        let mut unit = Self::try_from_indices(model, indices)?;
        unit.operator = operator;
        Ok(unit)
    }

    /// Number of selected atoms.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Returns `true` if no atoms are selected.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &Model {
        self.model
    }

    pub fn operator(&self) -> &SymmetryOperator {
        &self.operator
    }

    /// Lazy iterator over selected atom indices.
    pub fn atom_indices(&self) -> impl Iterator<Item = u32> + '_ {
        self.atoms.iter()
    }

    /// Lazy iterator over coordinates of selected atoms.
    pub fn coords(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.atoms
            .iter()
            .map(|i| self.operator.apply(self.model.coord(i as usize)))
    }

    /// Union of two units (must reference the same model).
    ///
    /// # Panics
    /// Panics if the units reference different models.
    pub fn union(&self, other: &Unit<'a>) -> Self {
        assert!(
            std::ptr::eq(self.model, other.model),
            "cannot union Units from different Models"
        );
        assert_eq!(
            self.operator, other.operator,
            "cannot union Units with different operators"
        );
        Self {
            model: self.model,
            atoms: self.atoms.union(&other.atoms),
            operator: self.operator.clone(),
        }
    }

    /// Intersection of two units (must reference the same model).
    ///
    /// # Panics
    /// Panics if the units reference different models.
    pub fn intersection(&self, other: &Unit<'a>) -> Self {
        assert!(
            std::ptr::eq(self.model, other.model),
            "cannot intersect Units from different Models"
        );
        assert_eq!(
            self.operator, other.operator,
            "cannot intersect Units with different operators"
        );
        Self {
            model: self.model,
            atoms: self.atoms.intersection(&other.atoms),
            operator: self.operator.clone(),
        }
    }

    /// Difference of two units: atoms in `self` but not in `other`.
    ///
    /// # Panics
    /// Panics if the units reference different models.
    pub fn difference(&self, other: &Unit<'a>) -> Self {
        assert!(
            std::ptr::eq(self.model, other.model),
            "cannot difference Units from different Models"
        );
        assert_eq!(
            self.operator, other.operator,
            "cannot difference Units with different operators"
        );
        Self {
            model: self.model,
            atoms: self.atoms.difference(&other.atoms),
            operator: self.operator.clone(),
        }
    }

    /// Select atoms belonging to a specific chain.
    ///
    /// Returns `None` if the chain doesn't exist.
    pub fn chain(model: &'a Model, chain_id: &str) -> Option<Self> {
        let chain_idx = model
            .hierarchy
            .chains
            .label_asym_id
            .iter()
            .position(|c| c == chain_id)?;

        let res_range = model.hierarchy.residue_to_chain.segment(chain_idx);
        if res_range.is_empty() {
            return Some(Self {
                model,
                atoms: OrderedSet::interval(0, 0),
                operator: SymmetryOperator::identity(),
            });
        }
        let atom_start = model
            .hierarchy
            .atom_to_residue
            .segment(res_range.start)
            .start;
        let atom_end = model
            .hierarchy
            .atom_to_residue
            .segment(res_range.end - 1)
            .end;

        Some(Self {
            model,
            atoms: OrderedSet::interval(atom_start as u32, atom_end as u32),
            operator: SymmetryOperator::identity(),
        })
    }

    /// Select backbone atoms (N, CA, C, O).
    pub fn backbone(model: &'a Model) -> Self {
        let backbone_names = ["N", "CA", "C", "O"];
        let indices: Vec<u32> = model
            .hierarchy
            .atoms
            .atom_name
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                if backbone_names.contains(&name.as_str()) {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect();

        let atoms = if indices.is_empty() {
            OrderedSet::interval(0, 0)
        } else {
            OrderedSet::from_sorted(indices)
        };

        Self {
            model,
            atoms,
            operator: SymmetryOperator::identity(),
        }
    }
}

impl Model {
    /// Create a filtered view of this model using a boolean mask.
    ///
    /// # Panics
    /// Panics if `mask.len() != self.n_atoms()`.
    pub fn filter(&self, mask: &[bool]) -> Unit<'_> {
        Unit::from_mask(self, mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Segmentation;
    use crate::info::elements::Element;
    use crate::model::bonds::Bonds;
    use crate::model::conformation::AtomicConformation;
    use crate::model::hierarchy::AtomicHierarchy;
    use crate::model::tables::{AtomsTable, ChainsTable, ResidueGroup, ResiduesTable};
    use std::sync::Arc;

    fn make_test_model(n_atoms: usize) -> Model {
        let atoms = AtomsTable {
            atom_name: (0..n_atoms).map(|i| format!("A{}", i)).collect(),
            auth_atom_name: (0..n_atoms).map(|i| format!("A{}", i)).collect(),
            element: vec![Element::C; n_atoms],
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };
        let residues = ResiduesTable {
            comp_id: vec!["ALA".into()],
            auth_comp_id: vec!["ALA".into()],
            label_seq_id: vec![0],
            auth_seq_id: vec![1],
            ins_code: vec![None],
            group: vec![ResidueGroup::Polymer],
        };
        let chains = ChainsTable {
            label_asym_id: vec!["A".into()],
            auth_asym_id: vec!["A".into()],
            entity_id: vec!["1".into()],
        };
        let atom_to_residue = Segmentation::from_offsets(vec![0, n_atoms as u32]);
        let residue_to_chain = Segmentation::from_offsets(vec![0, 1]);
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);

        let hierarchy = Arc::new(AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        });

        let conformation = AtomicConformation {
            x: (0..n_atoms).map(|i| i as f32).collect(),
            y: (0..n_atoms).map(|i| i as f32 * 10.0).collect(),
            z: (0..n_atoms).map(|_| 0.0).collect(),
            occupancy: None,
            b_iso: None,
            confidence: None,
        };

        Model::new(hierarchy, conformation)
    }

    fn make_backbone_model() -> Model {
        let atom_names = vec!["N".into(), "CA".into(), "C".into(), "O".into(), "CB".into()];
        let n_atoms = atom_names.len();
        let atoms = AtomsTable {
            auth_atom_name: atom_names.clone(),
            atom_name: atom_names,
            element: vec![Element::C; n_atoms],
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };
        let residues = ResiduesTable {
            comp_id: vec!["ALA".into()],
            auth_comp_id: vec!["ALA".into()],
            label_seq_id: vec![0],
            auth_seq_id: vec![1],
            ins_code: vec![None],
            group: vec![ResidueGroup::Polymer],
        };
        let chains = ChainsTable {
            label_asym_id: vec!["A".into()],
            auth_asym_id: vec!["A".into()],
            entity_id: vec!["1".into()],
        };
        let atom_to_residue = Segmentation::from_offsets(vec![0, n_atoms as u32]);
        let residue_to_chain = Segmentation::from_offsets(vec![0, 1]);
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);

        let hierarchy = Arc::new(AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        });

        let conformation = AtomicConformation {
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0],
            y: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            z: vec![0.0, 0.0, 0.0, 0.0, 0.0],
            occupancy: None,
            b_iso: None,
            confidence: None,
        };

        Model::new(hierarchy, conformation)
    }

    #[test]
    fn test_unit_from_mask_no_copy() {
        let model = make_test_model(10);
        let mask: Vec<bool> = (0..10).map(|i| i % 2 == 0).collect();

        let unit = Unit::from_mask(&model, &mask);

        assert_eq!(unit.len(), 5);
        assert!(std::ptr::eq(unit.model(), &model));

        let indices: Vec<u32> = unit.atom_indices().collect();
        assert_eq!(indices, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_unit_all() {
        let model = make_test_model(5);
        let unit = Unit::all(&model);

        assert_eq!(unit.len(), 5);
        let indices: Vec<u32> = unit.atom_indices().collect();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_unit_from_indices_rejects_out_of_range_index() {
        let model = make_test_model(5);
        let result = Unit::try_from_indices(&model, OrderedSet::from_sorted(vec![1, 5]));
        assert_eq!(
            result.err().unwrap().to_string(),
            "unit atom index 5 is out of range for 5 atoms"
        );
    }

    #[test]
    fn test_unit_iteration_lazy() {
        let model = make_test_model(100);
        let mask: Vec<bool> = (0..100).map(|i| i < 3).collect();

        let unit = Unit::from_mask(&model, &mask);

        let coords: Vec<[f32; 3]> = unit.coords().collect();
        assert_eq!(coords.len(), 3);
        assert_eq!(coords[0], [0.0, 0.0, 0.0]);
        assert_eq!(coords[1], [1.0, 10.0, 0.0]);
        assert_eq!(coords[2], [2.0, 20.0, 0.0]);
    }

    #[test]
    fn test_unit_set_operations() {
        let model = make_test_model(10);

        let mask_a: Vec<bool> = (0..10).map(|i| i < 5).collect();
        let mask_b: Vec<bool> = (0..10).map(|i| (3..8).contains(&i)).collect();

        let unit_a = Unit::from_mask(&model, &mask_a);
        let unit_b = Unit::from_mask(&model, &mask_b);

        let union = unit_a.union(&unit_b);
        let union_indices: Vec<u32> = union.atom_indices().collect();
        assert_eq!(union_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);

        let intersection = unit_a.intersection(&unit_b);
        let int_indices: Vec<u32> = intersection.atom_indices().collect();
        assert_eq!(int_indices, vec![3, 4]);

        let difference = unit_a.difference(&unit_b);
        let diff_indices: Vec<u32> = difference.atom_indices().collect();
        assert_eq!(diff_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_unit_empty() {
        let model = make_test_model(5);
        let mask = vec![false; 5];

        let unit = Unit::from_mask(&model, &mask);
        assert!(unit.is_empty());
        assert_eq!(unit.len(), 0);
    }

    #[test]
    fn test_model_filter() {
        let model = make_test_model(5);
        let mask = vec![true, false, true, false, true];

        let unit = model.filter(&mask);
        assert_eq!(unit.len(), 3);
    }

    #[test]
    fn test_unit_chain() {
        let model = make_test_model(5);

        let chain_a = Unit::chain(&model, "A");
        assert!(chain_a.is_some());
        assert_eq!(chain_a.unwrap().len(), 5);

        let chain_b = Unit::chain(&model, "B");
        assert!(chain_b.is_none());
    }

    #[test]
    fn test_unit_backbone() {
        let model = make_backbone_model();

        let backbone = Unit::backbone(&model);
        assert_eq!(backbone.len(), 4);

        let indices: Vec<u32> = backbone.atom_indices().collect();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "mask length must equal number of atoms")]
    fn test_unit_from_mask_wrong_length() {
        let model = make_test_model(5);
        let mask = vec![true, false, true];
        Unit::from_mask(&model, &mask);
    }
}
