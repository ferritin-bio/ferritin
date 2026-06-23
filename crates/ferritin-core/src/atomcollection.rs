//! AtomCollection
//!
//! An AtomCollection is primarily a group of atoms with some atomic properties like coordinates, element type
//! and residue information. Additional data like bonds can be added post-instantiation.
//! The data for residues within this collection can be iterated through. Other useful queries like inter-atomic
//! distances are supported.
use std::sync::Arc;
use super::bonds::{Bond, BondOrder};
use super::info::constants::get_bonds_canonical20;
use super::views::chain::ChainView;
use super::views::residue::ResidueView;
use crate::data::Segmentation;
use crate::info::elements::Element;
use crate::model::{AtomicConformation, AtomicHierarchy, Bonds, Model};
use crate::model::tables::{AtomsTable, ChainsTable, ResidueGroup, ResiduesTable};
use itertools::{Itertools, izip};

/// Atom Collection
///
/// The core data structure of ferritin-core.
///
/// it strives to be simple, high performance, and extensible using
/// traits.
#[derive(Clone)]
pub struct AtomCollection {
    size: usize,
    coords: Vec<[f32; 3]>,
    res_ids: Vec<i32>,
    res_names: Vec<String>,
    is_hetero: Vec<bool>,
    elements: Vec<Element>,
    atom_names: Vec<String>,
    chain_ids: Vec<String>,
    bonds: Option<Vec<Bond>>,
    residue_start_indices: Option<Vec<usize>>,
    chain_start_indices: Option<Vec<usize>>,
}

impl AtomCollection {
    pub fn new(
        size: usize,
        coords: Vec<[f32; 3]>,
        res_ids: Vec<i32>,
        res_names: Vec<String>,
        is_hetero: Vec<bool>,
        elements: Vec<Element>,
        atom_names: Vec<String>,
        chain_ids: Vec<String>,
        bonds: Option<Vec<Bond>>,
    ) -> Self {
        let mut ac = AtomCollection {
            size,
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            atom_names,
            chain_ids,
            bonds,
            residue_start_indices: None,
            chain_start_indices: None,
        };
        ac.calculate_chain_indices();
        ac
    }
    // Calculate and cache chain start indices
    pub fn calculate_chain_indices(&mut self) {
        if self.chain_start_indices.is_none() {
            if self.residue_start_indices.is_none() {
                let residue_starts = self.get_residue_starts();
                self.residue_start_indices = Some(residue_starts);
            }

            // Get chain starts as residue indices
            let residue_starts = self.residue_start_indices.as_ref().unwrap();
            let chain_starts: Vec<usize> = self
                .get_chain_starts()
                .iter()
                .map(|&atom_idx| {
                    // Find the residue index that contains this atom
                    let residue_idx = residue_starts
                        .iter()
                        .enumerate()
                        .filter(|&(_, &res_start)| res_start <= atom_idx)
                        .last()
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    residue_idx
                })
                .collect();

            self.chain_start_indices = Some(chain_starts);
        }
    }
    pub fn connect_via_residue_names(&mut self) {
        if self.bonds.is_some() {
            println!("Bonds already in place. Not overwriting.");
            return;
        }
        let aa_bond_info = get_bonds_canonical20();
        let residue_starts = self.get_residue_starts();
        let n_atoms = self.size;
        let mut bonds = Vec::new();
        for res_i in 0..residue_starts.len() - 1 {
            let curr_start_i = residue_starts[res_i];
            let next_start_i = residue_starts[res_i + 1];
            if let Some(bond_dict_for_res) =
                aa_bond_info.get(&self.res_names[curr_start_i].as_str())
            {
                for &(atom_name1, atom_name2, bond_type) in bond_dict_for_res {
                    let atom_indices1: Vec<usize> = (curr_start_i..next_start_i)
                        .filter(|&i| self.atom_names[i] == atom_name1)
                        .collect();
                    let atom_indices2: Vec<usize> = (curr_start_i..next_start_i)
                        .filter(|&i| self.atom_names[i] == atom_name2)
                        .collect();
                    for &i in &atom_indices1 {
                        for &j in &atom_indices2 {
                            bonds.push(Bond::new(i as i32, j as i32, bond_type));
                        }
                    }
                }
            }
        }
        // Backbone C→N peptide bonds between consecutive residues on the same chain
        for res_i in 0..residue_starts.len() - 1 {
            let curr_start_i = residue_starts[res_i];
            let next_start_i = residue_starts[res_i + 1];
            // Skip if these residues are on different chains
            if self.chain_ids[curr_start_i] != self.chain_ids[next_start_i] {
                continue;
            }
            // Skip hetero residues (ligands, solvent)
            if self.is_hetero[curr_start_i] || self.is_hetero[next_start_i] {
                continue;
            }
            let next_end_i = residue_starts
                .get(res_i + 2)
                .copied()
                .unwrap_or(n_atoms);
            let c_idx = (curr_start_i..next_start_i).find(|&i| self.atom_names[i] == "C");
            let n_idx = (next_start_i..next_end_i).find(|&i| self.atom_names[i] == "N");
            if let (Some(c), Some(n)) = (c_idx, n_idx) {
                bonds.push(Bond::new(c as i32, n as i32, BondOrder::Single));
            }
        }
        self.bonds = Some(bonds);
    }
    pub fn get_size(&self) -> usize {
        self.size
    }
    pub fn get_atom_name(&self, idx: usize) -> &String {
        &self.atom_names[idx]
    }
    pub fn get_bonds(&self) -> Option<&Vec<Bond>> {
        self.bonds.as_ref()
    }
    pub fn get_chain_id(&self, idx: usize) -> &String {
        &self.chain_ids[idx]
    }
    pub fn get_coord(&self, idx: usize) -> &[f32; 3] {
        &self.coords[idx]
    }
    pub fn get_coords(&self) -> &Vec<[f32; 3]> {
        self.coords.as_ref()
    }
    pub fn get_element(&self, idx: usize) -> &Element {
        &self.elements[idx]
    }
    pub fn get_elements(&self) -> &Vec<Element> {
        self.elements.as_ref()
    }
    pub fn get_is_hetero(&self, idx: usize) -> bool {
        self.is_hetero[idx]
    }
    pub fn get_resnames(&self) -> &Vec<String> {
        self.res_names.as_ref()
    }
    pub fn get_res_id(&self, idx: usize) -> &i32 {
        &self.res_ids[idx]
    }
    pub fn get_resids(&self) -> &Vec<i32> {
        self.res_ids.as_ref()
    }
    pub fn get_res_name(&self, idx: usize) -> &String {
        &self.res_names[idx]
    }
    /// A new residue starts, either when the chain ID, residue ID,
    /// insertion code or residue name changes from one to the next atom.
    fn get_residue_starts(&self) -> Vec<usize> {
        let mut starts = vec![0];

        starts.extend(
            izip!(&self.res_ids, &self.res_names, &self.chain_ids)
                .tuple_windows()
                .enumerate()
                .filter_map(
                    |(i, ((res_id1, name1, chain1), (res_id2, name2, chain2)))| {
                        if res_id1 != res_id2 || name1 != name2 || chain1 != chain2 {
                            Some(i + 1)
                        } else {
                            None
                        }
                    },
                ),
        );
        starts
    }
    pub fn get_residue_start_indices(&self) -> Option<&Vec<usize>> {
        self.residue_start_indices.as_ref()
    }
    /// A new chain starts when the chain ID changes from one atom to the next.
    fn get_chain_starts(&self) -> Vec<usize> {
        let mut starts = vec![0];
        starts.extend(
            self.chain_ids
                .iter()
                .tuple_windows()
                .enumerate()
                .filter_map(
                    |(i, (chain1, chain2))| {
                        if chain1 != chain2 { Some(i + 1) } else { None }
                    },
                ),
        );
        starts
    }

    /// Filter atoms using a boolean mask, returning a new `AtomCollection`.
    ///
    /// Bonds are remapped — only bonds where both endpoints survive the mask are kept,
    /// with indices adjusted to the new compact numbering.
    ///
    /// # Panics
    /// Panics if `mask.len() != self.size`.
    pub fn filter(&self, mask: &[bool]) -> AtomCollection {
        assert_eq!(mask.len(), self.size, "mask length must equal atom count");

        // Build old-index → new-index map in one pass.
        let mut next = 0usize;
        let remap: Vec<Option<usize>> = mask
            .iter()
            .map(|&keep| {
                if keep {
                    let idx = next;
                    next += 1;
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        let selected: Vec<usize> = remap.iter().enumerate().filter_map(|(i, r)| r.map(|_| i)).collect();

        let coords: Vec<[f32; 3]> = selected.iter().map(|&i| self.coords[i]).collect();
        let res_ids: Vec<i32> = selected.iter().map(|&i| self.res_ids[i]).collect();
        let res_names: Vec<String> = selected.iter().map(|&i| self.res_names[i].clone()).collect();
        let is_hetero: Vec<bool> = selected.iter().map(|&i| self.is_hetero[i]).collect();
        let elements: Vec<Element> = selected.iter().map(|&i| self.elements[i].clone()).collect();
        let atom_names: Vec<String> = selected.iter().map(|&i| self.atom_names[i].clone()).collect();
        let chain_ids: Vec<String> = selected.iter().map(|&i| self.chain_ids[i].clone()).collect();

        let bonds: Option<Vec<Bond>> = self.bonds.as_ref().map(|bonds| {
            bonds
                .iter()
                .filter_map(|b| {
                    let (a, b_idx) = b.get_atom_indices();
                    match (remap[a as usize], remap[b_idx as usize]) {
                        (Some(new_a), Some(new_b)) => {
                            Some(Bond::new(new_a as i32, new_b as i32, b.get_order()))
                        }
                        _ => None,
                    }
                })
                .collect()
        });

        AtomCollection::new(next, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, bonds)
    }

    /// Build a boolean mask selecting atoms belonging to `chain_id`.
    pub fn select_chain(&self, chain_id: &str) -> Vec<bool> {
        self.chain_ids.iter().map(|c| c == chain_id).collect()
    }

    /// Build a boolean mask selecting hetero atoms (ligands, solvent, etc.).
    pub fn select_hetero(&self) -> Vec<bool> {
        self.is_hetero.clone()
    }

    /// Build a boolean mask selecting backbone atoms (N, CA, C, O).
    pub fn select_backbone(&self) -> Vec<bool> {
        const BACKBONE: [&str; 4] = ["N", "CA", "C", "O"];
        self.atom_names.iter().map(|n| BACKBONE.contains(&n.as_str())).collect()
    }

    /// Build a boolean mask selecting atoms with `atom_name`.
    pub fn select_atom_name(&self, atom_name: &str) -> Vec<bool> {
        self.atom_names.iter().map(|n| n == atom_name).collect()
    }

    /// Build a boolean mask selecting residues with `res_name` (e.g. `"ALA"`).
    pub fn select_residue_name(&self, res_name: &str) -> Vec<bool> {
        self.res_names.iter().map(|n| n == res_name).collect()
    }

    pub fn iter_coords_and_elements(&self) -> impl Iterator<Item = (&[f32; 3], &Element)> {
        izip!(&self.coords, &self.elements)
    }

    pub fn iter_chains(&self) -> impl Iterator<Item = ChainView<'_>> {
        // Make sure indices are calculated
        let chain_starts = match &self.chain_start_indices {
            Some(indices) => indices.clone(),
            None => Vec::new(),
        };

        (0..chain_starts.len()).map(move |i| {
            let start_residue_idx = chain_starts[i];
            let end_residue_idx = if i + 1 < chain_starts.len() {
                chain_starts[i + 1]
            } else {
                // If it's the last chain, go to the end of the structure
                match &self.residue_start_indices {
                    Some(indices) => indices.len(),
                    None => self.size,
                }
            };

            ChainView {
                data: self,
                start_residue_idx,
                end_residue_idx,
            }
        })
    }
    pub fn iter_residues(&self) -> impl Iterator<Item = ResidueView<'_>> {
        let residue_starts = self.get_residue_starts();
        let atom_size = self.get_size();
        // Create a copy of the last element if it exists
        // Generate pairs for all residues
        let last_atom_idx = residue_starts.last().copied();
        (0..residue_starts.len().saturating_sub(1))
            .map(move |i| ResidueView::new(self, residue_starts[i], residue_starts[i + 1]))
            .chain(
                last_atom_idx
                    .map(|idx| ResidueView::new(self, idx, atom_size))
                    .into_iter(),
            )
    }
    /// Iterates over amino acid residues in the collection
    ///
    /// Returns a filtered iterator that only includes standard amino acid residues
    pub fn iter_residues_aminoacid(&self) -> impl Iterator<Item = ResidueView<'_>> {
        self.iter_residues()
            .filter(|residue| residue.is_amino_acid())
    }

    /// Convert this AtomCollection to a Model representation.
    ///
    /// Lifts the flat per-atom vectors into the hierarchical Model structure
    /// (AtomicHierarchy + AtomicConformation).
    pub fn to_model(&self) -> Model {
        let n_atoms = self.size;

        let atoms = AtomsTable {
            atom_name: self.atom_names.clone(),
            element: self.elements.iter().map(|e| e.symbol().to_string()).collect(),
            alt_loc: vec![None; n_atoms],
            formal_charge: vec![None; n_atoms],
        };

        let residue_starts = self.get_residue_starts();
        let n_residues = residue_starts.len();
        let mut comp_id = Vec::with_capacity(n_residues);
        let mut label_seq_id = Vec::with_capacity(n_residues);
        let mut auth_seq_id = Vec::with_capacity(n_residues);
        let mut ins_code = Vec::with_capacity(n_residues);
        let mut group = Vec::with_capacity(n_residues);

        for &start in &residue_starts {
            comp_id.push(self.res_names[start].clone());
            let res_id = self.res_ids[start];
            label_seq_id.push(res_id);
            auth_seq_id.push(res_id);
            ins_code.push(None);
            group.push(if self.is_hetero[start] {
                ResidueGroup::NonPolymer
            } else {
                ResidueGroup::Polymer
            });
        }

        let residues = ResiduesTable {
            comp_id,
            label_seq_id,
            auth_seq_id,
            ins_code,
            group,
        };

        let chain_starts = self.get_chain_starts();
        let n_chains = chain_starts.len();
        let mut label_asym_id = Vec::with_capacity(n_chains);
        let mut auth_asym_id = Vec::with_capacity(n_chains);
        let mut entity_id = Vec::with_capacity(n_chains);

        for &start in &chain_starts {
            let chain_id = self.chain_ids[start].clone();
            label_asym_id.push(chain_id.clone());
            auth_asym_id.push(chain_id.clone());
            entity_id.push(chain_id);
        }

        let chains = ChainsTable {
            label_asym_id,
            auth_asym_id,
            entity_id,
        };

        let mut atom_offsets: Vec<u32> = residue_starts.iter().map(|&s| s as u32).collect();
        atom_offsets.push(n_atoms as u32);
        let atom_to_residue = Segmentation::from_offsets(atom_offsets);

        let residue_offsets: Vec<u32> = chain_starts
            .iter()
            .map(|&atom_start| {
                residue_starts
                    .iter()
                    .position(|&r| r == atom_start)
                    .unwrap_or(0) as u32
            })
            .chain(std::iter::once(n_residues as u32))
            .collect();
        let residue_to_chain = Segmentation::from_offsets(residue_offsets);

        let bonds = Bonds::from_unsorted(vec![], vec![], vec![], n_atoms);

        let hierarchy = Arc::new(AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue,
            residue_to_chain,
            bonds,
        });

        let x: Vec<f32> = self.coords.iter().map(|c| c[0]).collect();
        let y: Vec<f32> = self.coords.iter().map(|c| c[1]).collect();
        let z: Vec<f32> = self.coords.iter().map(|c| c[2]).collect();

        let conformation = AtomicConformation {
            x,
            y,
            z,
            occupancy: None,
            b_iso: None,
            confidence: None,
        };

        Model::new(hierarchy, conformation)
    }
}

impl From<&Model> for AtomCollection {
    /// Convert a Model to an AtomCollection.
    ///
    /// Regenerates per-atom vectors from the hierarchy and conformation.
    fn from(model: &Model) -> Self {
        let n_atoms = model.n_atoms();
        let hierarchy = &model.hierarchy;
        let conformation = &model.conformation;

        let coords: Vec<[f32; 3]> = (0..n_atoms)
            .map(|i| [conformation.x[i], conformation.y[i], conformation.z[i]])
            .collect();

        let mut res_ids = Vec::with_capacity(n_atoms);
        let mut res_names = Vec::with_capacity(n_atoms);
        let mut is_hetero = Vec::with_capacity(n_atoms);
        let mut chain_ids = Vec::with_capacity(n_atoms);

        for atom_idx in 0..n_atoms {
            let res_idx = hierarchy.residue_of_atom(atom_idx);
            let chain_idx = hierarchy.chain_of_residue(res_idx);

            res_ids.push(hierarchy.residues.auth_seq_id[res_idx]);
            res_names.push(hierarchy.residues.comp_id[res_idx].clone());
            is_hetero.push(hierarchy.residues.group[res_idx] == ResidueGroup::NonPolymer);
            chain_ids.push(hierarchy.chains.auth_asym_id[chain_idx].clone());
        }

        let elements: Vec<Element> = hierarchy
            .atoms
            .element
            .iter()
            .map(|s| Element::from_symbol(s).unwrap_or(Element::C))
            .collect();

        let atom_names = hierarchy.atoms.atom_name.clone();

        AtomCollection::new(
            n_atoms,
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            atom_names,
            chain_ids,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_atom_collection() -> AtomCollection {
        let coords = vec![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ];
        let res_ids = vec![1, 1, 1, 2, 2];
        let res_names = vec![
            "ALA".into(), "ALA".into(), "ALA".into(),
            "GLY".into(), "GLY".into(),
        ];
        let is_hetero = vec![false, false, false, false, false];
        let elements = vec![Element::N, Element::C, Element::C, Element::N, Element::C];
        let atom_names = vec![
            "N".into(), "CA".into(), "C".into(),
            "N".into(), "CA".into(),
        ];
        let chain_ids = vec![
            "A".into(), "A".into(), "A".into(),
            "A".into(), "A".into(),
        ];

        AtomCollection::new(
            5,
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            atom_names,
            chain_ids,
            None,
        )
    }

    #[test]
    fn test_atomcollection_to_model_roundtrip() {
        let original = make_test_atom_collection();

        let model = original.to_model();
        let restored = AtomCollection::from(&model);

        assert_eq!(original.get_size(), restored.get_size());

        for i in 0..original.get_size() {
            assert_eq!(original.get_coord(i), restored.get_coord(i), "coord mismatch at {}", i);
            assert_eq!(original.get_res_id(i), restored.get_res_id(i), "res_id mismatch at {}", i);
            assert_eq!(original.get_res_name(i), restored.get_res_name(i), "res_name mismatch at {}", i);
            assert_eq!(original.get_atom_name(i), restored.get_atom_name(i), "atom_name mismatch at {}", i);
            assert_eq!(original.get_chain_id(i), restored.get_chain_id(i), "chain_id mismatch at {}", i);
            assert_eq!(original.get_is_hetero(i), restored.get_is_hetero(i), "is_hetero mismatch at {}", i);
        }
    }

    #[test]
    fn test_model_to_atomcollection_coords() {
        let ac = make_test_atom_collection();
        let model = ac.to_model();

        let model_coords = model.coords_as_slice();
        assert_eq!(model_coords.len(), ac.get_size());
        for i in 0..ac.get_size() {
            assert_eq!(model_coords[i], *ac.get_coord(i), "coord mismatch at {}", i);
        }

        assert_eq!(model.x(), &[1.0, 4.0, 7.0, 10.0, 13.0]);
        assert_eq!(model.y(), &[2.0, 5.0, 8.0, 11.0, 14.0]);
        assert_eq!(model.z(), &[3.0, 6.0, 9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_model_hierarchy_structure() {
        let ac = make_test_atom_collection();
        let model = ac.to_model();

        assert_eq!(model.n_atoms(), 5);
        assert_eq!(model.n_residues(), 2);
        assert_eq!(model.n_chains(), 1);

        assert_eq!(model.hierarchy.atoms_in_residue(0), 0..3);
        assert_eq!(model.hierarchy.atoms_in_residue(1), 3..5);

        assert_eq!(model.hierarchy.residues_in_chain(0), 0..2);
    }

    #[test]
    fn test_multi_chain_roundtrip() {
        let coords = vec![
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
        ];
        let res_ids = vec![1, 1, 10];
        let res_names = vec!["ALA".into(), "ALA".into(), "GLY".into()];
        let is_hetero = vec![false, false, true];
        let elements = vec![Element::N, Element::C, Element::N];
        let atom_names = vec!["N".into(), "CA".into(), "N".into()];
        let chain_ids = vec!["A".into(), "A".into(), "B".into()];

        let original = AtomCollection::new(
            3, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, None,
        );

        let model = original.to_model();
        assert_eq!(model.n_chains(), 2);
        assert_eq!(model.n_residues(), 2);

        let restored = AtomCollection::from(&model);
        assert_eq!(restored.get_chain_id(0), "A");
        assert_eq!(restored.get_chain_id(2), "B");
        assert!(restored.get_is_hetero(2));
        assert!(!restored.get_is_hetero(0));
    }

    fn make_two_chain_collection() -> AtomCollection {
        // Chain A: 3 atoms (N, CA, C), res 1 ALA
        // Chain B: 2 atoms (N, CA), res 2 GLY
        // Atom order: 0=A/N, 1=A/CA, 2=A/C, 3=B/N, 4=B/CA
        let coords = vec![[1.,0.,0.],[2.,0.,0.],[3.,0.,0.],[4.,0.,0.],[5.,0.,0.]];
        let res_ids = vec![1, 1, 1, 2, 2];
        let res_names: Vec<String> = ["ALA","ALA","ALA","GLY","GLY"].iter().map(|s| s.to_string()).collect();
        let is_hetero = vec![false, false, false, false, false];
        let elements = vec![Element::N, Element::C, Element::C, Element::N, Element::C];
        let atom_names: Vec<String> = ["N","CA","C","N","CA"].iter().map(|s| s.to_string()).collect();
        let chain_ids: Vec<String> = ["A","A","A","B","B"].iter().map(|s| s.to_string()).collect();
        let bonds = vec![
            Bond::new(0, 1, BondOrder::Single),
            Bond::new(1, 2, BondOrder::Single),
            Bond::new(3, 4, BondOrder::Single),
        ];
        AtomCollection::new(5, coords, res_ids, res_names, is_hetero, elements, atom_names, chain_ids, Some(bonds))
    }

    #[test]
    fn test_filter_keeps_selected_atoms() {
        let ac = make_two_chain_collection();
        let mask = vec![true, false, true, false, true];
        let filtered = ac.filter(&mask);
        assert_eq!(filtered.get_size(), 3);
        assert_eq!(filtered.get_coord(0), &[1., 0., 0.]);
        assert_eq!(filtered.get_coord(1), &[3., 0., 0.]);
        assert_eq!(filtered.get_coord(2), &[5., 0., 0.]);
    }

    #[test]
    fn test_filter_remaps_bonds() {
        let ac = make_two_chain_collection();
        // Keep atoms 0,1,2 (chain A). Bond 0-1 and 1-2 survive; bond 3-4 is dropped.
        let mask = vec![true, true, true, false, false];
        let filtered = ac.filter(&mask);
        let bonds = filtered.get_bonds().unwrap();
        assert_eq!(bonds.len(), 2);
        let indices: Vec<(i32, i32)> = bonds.iter().map(|b| b.get_atom_indices()).collect();
        assert!(indices.contains(&(0, 1)));
        assert!(indices.contains(&(1, 2)));
    }

    #[test]
    fn test_filter_drops_cross_boundary_bonds() {
        let ac = make_two_chain_collection();
        // Keep only atoms 0 and 4 — bond 0-1 and 3-4 are cut; bond 3-4's 3 is missing.
        let mask = vec![true, false, false, false, true];
        let filtered = ac.filter(&mask);
        assert_eq!(filtered.get_size(), 2);
        let bonds = filtered.get_bonds().unwrap();
        assert!(bonds.is_empty());
    }

    #[test]
    fn test_select_chain() {
        let ac = make_two_chain_collection();
        let mask = ac.select_chain("B");
        assert_eq!(mask, vec![false, false, false, true, true]);
        let filtered = ac.filter(&mask);
        assert_eq!(filtered.get_size(), 2);
    }

    #[test]
    fn test_select_backbone() {
        let ac = make_two_chain_collection();
        let mask = ac.select_backbone();
        // N, CA, C, N, CA — all 5 are backbone
        assert_eq!(mask, vec![true, true, true, true, true]);
    }

    #[test]
    fn test_select_atom_name() {
        let ac = make_two_chain_collection();
        let mask = ac.select_atom_name("CA");
        assert_eq!(mask, vec![false, true, false, false, true]);
    }

    #[test]
    fn test_select_residue_name() {
        let ac = make_two_chain_collection();
        let mask = ac.select_residue_name("ALA");
        assert_eq!(mask, vec![true, true, true, false, false]);
    }

    #[test]
    #[should_panic(expected = "mask length must equal atom count")]
    fn test_filter_wrong_mask_length() {
        let ac = make_test_atom_collection();
        ac.filter(&[true, false]);
    }
}
