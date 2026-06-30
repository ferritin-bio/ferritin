//! The [`Model`] struct: one frame of a (possibly multi-frame) structure.
//!
//! A `Model` pairs a shared [`AtomicHierarchy`] (topology) with a frame-specific
//! [`AtomicConformation`] (coordinates). Multiple frames of a trajectory share
//! the same `Arc<AtomicHierarchy>`, avoiding redundant copies of the sequence,
//! connectivity, and chain/residue data.
//!
//! # Canonical iteration order
//!
//! Residues are stored in **canonical order**: chain order → `auth_seq_id` →
//! `ins_code`.  The hierarchy's `atom_to_residue` and `residue_to_chain`
//! segmentations must be built with residues in this order.  Callers must ensure
//! the input data is sorted before constructing `AtomicHierarchy`; no runtime
//! re-sorting is performed here.

use std::sync::Arc;
use super::hierarchy::AtomicHierarchy;
use super::conformation::AtomicConformation;
use super::tables::ResidueGroup;
use crate::info::elements::Element;
use crate::views::{ModelAtomView, ModelChainView, ModelResidueView};

/// One frame of a (possibly multi-frame) molecular structure.
///
/// # Shared topology
///
/// `hierarchy` is wrapped in `Arc` so multiple `Model`s (e.g., NMR ensemble
/// frames or MD trajectory frames) can share a single topology without cloning.
///
/// # Canonical iteration order
///
/// **INVARIANT**: residues within each chain iterate in ascending `auth_seq_id`
/// order, with ties broken by `ins_code` (`None` < `Some('A')` < `Some('B')`
/// …).  This invariant is established when the [`AtomicHierarchy`] is
/// constructed and is relied upon by iterators such as [`Model::protein_residues`]
/// and [`Model::ligand_residues`].
#[derive(Clone)]
pub struct Model {
    /// Shared topology (connectivity, sequence, chain/residue data).
    pub hierarchy: Arc<AtomicHierarchy>,
    /// Frame-specific coordinate data.
    pub conformation: AtomicConformation,
}

impl Model {
    /// Construct a new `Model` from a shared topology and per-frame coordinates.
    pub fn new(hierarchy: Arc<AtomicHierarchy>, conformation: AtomicConformation) -> Self {
        Self { hierarchy, conformation }
    }

    /// Get `[x, y, z]` for atom `i`.
    pub fn coord(&self, i: usize) -> [f32; 3] {
        self.conformation.coord(i)
    }

    /// Direct access to x coordinates (SoA layout).
    pub fn x(&self) -> &[f32] {
        &self.conformation.x
    }

    /// Direct access to y coordinates (SoA layout).
    pub fn y(&self) -> &[f32] {
        &self.conformation.y
    }

    /// Direct access to z coordinates (SoA layout).
    pub fn z(&self) -> &[f32] {
        &self.conformation.z
    }

    /// Returns coordinates as array-of-structs `[[x,y,z], ...]`.
    ///
    /// This is a compatibility shim for callers expecting the old AoS layout.
    /// Contains the blast radius for the AoS→SoA transition.
    pub fn coords_as_slice(&self) -> Vec<[f32; 3]> {
        let n = self.n_atoms();
        let (x, y, z) = (&self.conformation.x, &self.conformation.y, &self.conformation.z);
        (0..n).map(|i| [x[i], y[i], z[i]]).collect()
    }

    /// Total number of atoms.
    pub fn n_atoms(&self) -> usize {
        self.hierarchy.n_atoms()
    }

    /// Total number of residues.
    pub fn n_residues(&self) -> usize {
        self.hierarchy.n_residues()
    }

    /// Total number of chains.
    pub fn n_chains(&self) -> usize {
        self.hierarchy.n_chains()
    }

    /// Iterator over residue indices where `group == ResidueGroup::Polymer`.
    ///
    /// Residues are yielded in canonical order (see type-level doc).
    pub fn protein_residues(&self) -> impl Iterator<Item = usize> + '_ {
        let groups = &self.hierarchy.residues.group;
        (0..self.n_residues()).filter(move |&i| groups[i] == ResidueGroup::Polymer)
    }

    /// Iterator over residue indices where `group == ResidueGroup::NonPolymer`.
    ///
    /// Residues are yielded in canonical order (see type-level doc).
    pub fn ligand_residues(&self) -> impl Iterator<Item = usize> + '_ {
        let groups = &self.hierarchy.residues.group;
        (0..self.n_residues()).filter(move |&i| groups[i] == ResidueGroup::NonPolymer)
    }

    /// Iterator over all chains as [`ModelChainView`].
    pub fn chains(&self) -> impl Iterator<Item = ModelChainView<'_>> {
        (0..self.n_chains()).map(move |i| ModelChainView::new(self, i))
    }

    /// Iterator over all residues as [`ModelResidueView`].
    pub fn residues(&self) -> impl Iterator<Item = ModelResidueView<'_>> {
        (0..self.n_residues()).map(move |i| ModelResidueView::new(self, i))
    }

    /// Iterator over amino-acid residues only, as [`ModelResidueView`].
    pub fn residues_aminoacid(&self) -> impl Iterator<Item = ModelResidueView<'_>> {
        self.residues().filter(|r| r.is_amino_acid())
    }

    /// Iterator over all atoms as [`ModelAtomView`].
    pub fn atoms(&self) -> impl Iterator<Item = ModelAtomView<'_>> {
        (0..self.n_atoms()).map(move |i| ModelAtomView::new(self, i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Segmentation;
    use crate::model::bonds::Bonds;
    use crate::model::tables::{AtomsTable, ChainsTable, ResidueGroup, ResiduesTable};

    /// Build a minimal 1-chain, 3-residue, 3-atom hierarchy.
    fn make_simple_hierarchy(n_residues: usize, groups: Vec<ResidueGroup>) -> Arc<AtomicHierarchy> {
        let n_atoms = n_residues; // 1 atom per residue for simplicity
        let atoms = AtomsTable {
            atom_name: (0..n_atoms).map(|_| "CA".to_string()).collect(),
            element: vec![Element::C; n_atoms],
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };
        let residues = ResiduesTable {
            comp_id: (0..n_residues).map(|i| format!("R{}", i)).collect(),
            label_seq_id: (0..n_residues as i32).collect(),
            auth_seq_id: (1..=n_residues as i32).collect(),
            ins_code: vec![None; n_residues],
            group: groups,
        };
        let chains = ChainsTable {
            label_asym_id: vec!["A".into()],
            auth_asym_id: vec!["A".into()],
            entity_id: vec!["1".into()],
        };
        // 1 atom per residue
        let atom_offsets: Vec<u32> = (0..=n_residues as u32).collect();
        let atom_to_residue = Segmentation::from_offsets(atom_offsets);
        // all residues in 1 chain
        let residue_to_chain = Segmentation::from_offsets(vec![0, n_residues as u32]);
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);

        Arc::new(AtomicHierarchy { atoms, residues, chains, atom_to_residue, residue_to_chain, bonds })
    }

    fn make_conformation(n: usize) -> AtomicConformation {
        AtomicConformation {
            x: (0..n).map(|i| i as f32).collect(),
            y: (0..n).map(|i| i as f32 * 10.0).collect(),
            z: (0..n).map(|_| 0.0).collect(),
            occupancy: None,
            b_iso: None,
            confidence: None,
        }
    }

    #[test]
    fn test_model_arc_shared() {
        // Two Models sharing the same Arc<AtomicHierarchy>
        let hierarchy = make_simple_hierarchy(3, vec![ResidueGroup::Polymer; 3]);

        let conf1 = make_conformation(3);
        let conf2 = AtomicConformation {
            x: vec![100.0, 200.0, 300.0],
            y: vec![0.0, 0.0, 0.0],
            z: vec![0.0, 0.0, 0.0],
            occupancy: None,
            b_iso: None,
            confidence: None,
        };

        let model1 = Model::new(Arc::clone(&hierarchy), conf1);
        let model2 = Model::new(Arc::clone(&hierarchy), conf2);

        // Both models point to the exact same AtomicHierarchy allocation
        assert!(Arc::ptr_eq(&model1.hierarchy, &model2.hierarchy),
            "Both models must share the same Arc<AtomicHierarchy>");

        // But coordinates are independent
        assert_eq!(model1.coord(0), [0.0, 0.0, 0.0]);
        assert_eq!(model2.coord(0), [100.0, 0.0, 0.0]);

        // Topology counts are identical (same Arc)
        assert_eq!(model1.n_atoms(), model2.n_atoms());
        assert_eq!(model1.n_residues(), model2.n_residues());
    }

    #[test]
    fn test_model_iteration_order() {
        // INVARIANT: residues iterate chain-by-chain, in auth_seq_id order within each chain.
        //
        // We build a 2-chain hierarchy where:
        //   Chain 0: residues with auth_seq_id [3, 1, 2] stored in canonical order [1, 2, 3]
        //   Chain 1: residues with auth_seq_id [10, 5] stored in canonical order [5, 10]
        //
        // The test verifies that protein_residues() yields residue indices 0,1,2,3,4
        // which correspond to auth_seq_id [1, 2, 3, 5, 10] — chain-then-auth_seq_id order.
        let n_atoms = 5;
        let atoms = AtomsTable {
            atom_name: vec!["CA".into(); n_atoms],
            element: vec![Element::C; n_atoms],
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };
        // Residues stored in canonical order: chain 0 (auth_seq_id 1,2,3) then chain 1 (auth_seq_id 5,10)
        let residues = ResiduesTable {
            comp_id: vec!["ALA".into(), "GLY".into(), "SER".into(), "HOH".into(), "ATP".into()],
            label_seq_id: vec![0, 1, 2, 3, 4],
            auth_seq_id: vec![1, 2, 3, 5, 10], // canonical ascending order per chain
            ins_code: vec![None; 5],
            group: vec![
                ResidueGroup::Polymer,
                ResidueGroup::Polymer,
                ResidueGroup::Polymer,
                ResidueGroup::Polymer,
                ResidueGroup::Polymer,
            ],
        };
        let chains = ChainsTable {
            label_asym_id: vec!["A".into(), "B".into()],
            auth_asym_id:  vec!["A".into(), "B".into()],
            entity_id:     vec!["1".into(), "2".into()],
        };
        let atom_to_residue = Segmentation::from_offsets(vec![0, 1, 2, 3, 4, 5]);
        let residue_to_chain = Segmentation::from_offsets(vec![0, 3, 5]);
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);

        let hierarchy = Arc::new(AtomicHierarchy {
            atoms, residues, chains, atom_to_residue, residue_to_chain, bonds,
        });
        let model = Model::new(hierarchy, make_conformation(n_atoms));

        // Verify iteration is in canonical order: residue index 0,1,2 (chain 0) then 3,4 (chain 1)
        let res_indices: Vec<usize> = model.protein_residues().collect();
        assert_eq!(res_indices, vec![0, 1, 2, 3, 4],
            "Residues must iterate in canonical order: chain 0 first, then chain 1, \
             each in ascending auth_seq_id order");

        // Verify the auth_seq_ids are in expected canonical order
        let auth_seq_ids: Vec<i32> = res_indices
            .iter()
            .map(|&r| model.hierarchy.residues.auth_seq_id[r])
            .collect();
        assert_eq!(auth_seq_ids, vec![1, 2, 3, 5, 10],
            "auth_seq_ids must be in ascending order within each chain");

        // Verify chain assignment is correct
        let chain_indices: Vec<usize> = res_indices
            .iter()
            .map(|&r| model.hierarchy.chain_of_residue(r))
            .collect();
        assert_eq!(chain_indices, vec![0, 0, 0, 1, 1],
            "Residues 0-2 in chain 0, residues 3-4 in chain 1");
    }

    #[test]
    fn test_model_protein_ligand_split() {
        // Build a model with a mix of Polymer and NonPolymer residues.
        // Residue layout: [Polymer, Polymer, Polymer, NonPolymer, NonPolymer]
        let groups = vec![
            ResidueGroup::Polymer,
            ResidueGroup::Polymer,
            ResidueGroup::Polymer,
            ResidueGroup::NonPolymer,
            ResidueGroup::NonPolymer,
        ];
        let hierarchy = {
            let n_residues = 5;
            let n_atoms = 5;
            let atoms = AtomsTable {
                atom_name: vec!["CA".into(); n_atoms],
                element: vec![Element::C; n_atoms],
                alt_loc: vec![None; n_atoms],
                formal_charge: vec![None; n_atoms],
            };
            let residues = ResiduesTable {
                comp_id: vec!["ALA".into(), "GLY".into(), "SER".into(), "HOH".into(), "ATP".into()],
                label_seq_id: vec![0, 1, 2, 3, 4],
                auth_seq_id: vec![1, 2, 3, 401, 402],
                ins_code: vec![None; n_residues],
                group: groups,
            };
            let chains = ChainsTable {
                label_asym_id: vec!["A".into()],
                auth_asym_id: vec!["A".into()],
                entity_id: vec!["1".into()],
            };
            let atom_to_residue = Segmentation::from_offsets(vec![0, 1, 2, 3, 4, 5]);
            let residue_to_chain = Segmentation::from_offsets(vec![0, 5]);
            let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);
            Arc::new(AtomicHierarchy { atoms, residues, chains, atom_to_residue, residue_to_chain, bonds })
        };

        let model = Model::new(hierarchy, make_conformation(5));

        let protein: Vec<usize> = model.protein_residues().collect();
        let ligands: Vec<usize> = model.ligand_residues().collect();

        assert_eq!(protein, vec![0, 1, 2], "Polymer residues should be 0, 1, 2");
        assert_eq!(ligands, vec![3, 4], "NonPolymer residues should be 3, 4");

        // Verify no overlap and full coverage
        assert_eq!(protein.len() + ligands.len(), 5);
        for p in &protein {
            assert!(!ligands.contains(p), "No residue should be in both groups");
        }
    }

    #[test]
    fn test_model_coords_as_slice() {
        let hierarchy = make_simple_hierarchy(3, vec![ResidueGroup::Polymer; 3]);
        let conf = AtomicConformation {
            x: vec![1.0, 4.0, 7.0],
            y: vec![2.0, 5.0, 8.0],
            z: vec![3.0, 6.0, 9.0],
            occupancy: None,
            b_iso: None,
            confidence: None,
        };
        let model = Model::new(hierarchy, conf);

        let coords = model.coords_as_slice();
        assert_eq!(coords.len(), 3);
        assert_eq!(coords[0], [1.0, 2.0, 3.0]);
        assert_eq!(coords[1], [4.0, 5.0, 6.0]);
        assert_eq!(coords[2], [7.0, 8.0, 9.0]);

        assert_eq!(model.x(), &[1.0, 4.0, 7.0]);
        assert_eq!(model.y(), &[2.0, 5.0, 8.0]);
        assert_eq!(model.z(), &[3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_model_soa_aos_consistency() {
        let hierarchy = make_simple_hierarchy(5, vec![ResidueGroup::Polymer; 5]);
        let conf = make_conformation(5);
        let model = Model::new(hierarchy, conf);

        let aos = model.coords_as_slice();
        let (x, y, z) = (model.x(), model.y(), model.z());

        for i in 0..5 {
            assert_eq!(aos[i][0], x[i], "x mismatch at {}", i);
            assert_eq!(aos[i][1], y[i], "y mismatch at {}", i);
            assert_eq!(aos[i][2], z[i], "z mismatch at {}", i);
        }
    }

    #[test]
    fn test_model_chains_iterator() {
        let hierarchy = {
            let n_atoms = 5;
            let atoms = AtomsTable {
                atom_name: vec!["CA".into(); n_atoms],
                element: vec![Element::C; n_atoms],
                alt_loc: vec![None; n_atoms],
                formal_charge: vec![None; n_atoms],
            };
            let residues = ResiduesTable {
                comp_id: vec!["ALA".into(), "GLY".into(), "SER".into(), "HOH".into(), "ATP".into()],
                label_seq_id: vec![0, 1, 2, 3, 4],
                auth_seq_id: vec![1, 2, 3, 5, 10],
                ins_code: vec![None; 5],
                group: vec![ResidueGroup::Polymer; 5],
            };
            let chains = ChainsTable {
                label_asym_id: vec!["A".into(), "B".into()],
                auth_asym_id: vec!["A".into(), "B".into()],
                entity_id: vec!["1".into(), "2".into()],
            };
            let atom_to_residue = Segmentation::from_offsets(vec![0, 1, 2, 3, 4, 5]);
            let residue_to_chain = Segmentation::from_offsets(vec![0, 3, 5]);
            let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);
            Arc::new(AtomicHierarchy { atoms, residues, chains, atom_to_residue, residue_to_chain, bonds })
        };
        let model = Model::new(hierarchy, make_conformation(5));

        let chains: Vec<_> = model.chains().collect();
        assert_eq!(chains.len(), 2, "should have 2 chains");
        assert_eq!(chains[0].chain_id(), "A");
        assert_eq!(chains[1].chain_id(), "B");
        assert_eq!(chains[0].residue_count(), 3);
        assert_eq!(chains[1].residue_count(), 2);
    }

    #[test]
    fn test_model_residues_iterator() {
        let hierarchy = make_simple_hierarchy(
            3,
            vec![ResidueGroup::Polymer, ResidueGroup::Polymer, ResidueGroup::NonPolymer],
        );
        let model = Model::new(hierarchy, make_conformation(3));

        let residues: Vec<_> = model.residues().collect();
        assert_eq!(residues.len(), 3);
        assert_eq!(residues[0].residue_id(), 1);
        assert_eq!(residues[2].residue_id(), 3);
        assert_eq!(residues[0].chain_id(), "A");
        assert!(residues[2].atom_count() > 0);
    }

    #[test]
    fn test_model_atoms_iterator() {
        let hierarchy = make_simple_hierarchy(3, vec![ResidueGroup::Polymer; 3]);
        let conf = AtomicConformation {
            x: vec![1.0, 4.0, 7.0],
            y: vec![2.0, 5.0, 8.0],
            z: vec![3.0, 6.0, 9.0],
            occupancy: None,
            b_iso: None,
            confidence: None,
        };
        let model = Model::new(hierarchy, conf);

        let atoms: Vec<_> = model.atoms().collect();
        assert_eq!(atoms.len(), 3);
        assert_eq!(atoms[0].coords(), [1.0, 2.0, 3.0]);
        assert_eq!(atoms[1].coords(), [4.0, 5.0, 6.0]);
        assert_eq!(atoms[0].atom_name(), "CA");
        assert_eq!(atoms[0].chain_id(), "A");
    }

    #[test]
    fn test_model_chain_residue_atom_traversal() {
        // Build a 2-chain, 4-residue, 8-atom model (2 atoms per residue)
        let n_atoms = 8;
        let n_residues = 4;
        let atoms = AtomsTable {
            atom_name: (0..n_atoms).map(|i| if i % 2 == 0 { "N".into() } else { "CA".into() }).collect(),
            element: vec![Element::C; n_atoms],
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };
        let residues = ResiduesTable {
            comp_id: vec!["ALA".into(), "GLY".into(), "HOH".into(), "ATP".into()],
            label_seq_id: vec![0, 1, 2, 3],
            auth_seq_id: vec![1, 2, 10, 20],
            ins_code: vec![None; n_residues],
            group: vec![
                ResidueGroup::Polymer, ResidueGroup::Polymer,
                ResidueGroup::NonPolymer, ResidueGroup::NonPolymer,
            ],
        };
        let chains = ChainsTable {
            label_asym_id: vec!["A".into(), "B".into()],
            auth_asym_id: vec!["A".into(), "B".into()],
            entity_id: vec!["1".into(), "2".into()],
        };
        let atom_to_residue = Segmentation::from_offsets(vec![0, 2, 4, 6, 8]);
        let residue_to_chain = Segmentation::from_offsets(vec![0, 2, 4]);
        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);
        let hierarchy = Arc::new(AtomicHierarchy { atoms, residues, chains, atom_to_residue, residue_to_chain, bonds });
        let model = Model::new(hierarchy, make_conformation(n_atoms));

        // Traverse chain -> residue -> atom
        let chain_a = model.chains().next().unwrap();
        assert_eq!(chain_a.chain_id(), "A");
        let res_in_a: Vec<_> = chain_a.iter_residues().collect();
        assert_eq!(res_in_a.len(), 2);
        assert_eq!(res_in_a[0].residue_name(), "ALA");
        assert!(res_in_a[0].is_amino_acid());

        let atoms_in_r0: Vec<_> = res_in_a[0].iter_atoms().collect();
        assert_eq!(atoms_in_r0.len(), 2);
        assert_eq!(atoms_in_r0[0].atom_name(), "N");
        assert_eq!(atoms_in_r0[1].atom_name(), "CA");

        let ca = res_in_a[0].find_atom_by_name("CA");
        assert!(ca.is_some());
        assert_eq!(ca.unwrap().atom_name(), "CA");
    }
}
