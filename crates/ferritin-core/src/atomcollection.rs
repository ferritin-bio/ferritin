//! AtomCollection
//!
//! An AtomCollection is primarily a group of atoms with some atomic properties like coordinates, element type
//! and residue information. Additional data like bonds can be added post-instantiation.
//! The data for residues within this collection can be iterated through. Other useful queries like inter-atomic
//! distances are supported.
use super::bonds::Bond;
use super::info::constants::get_bonds_canonical20;
use super::views::chain::ChainView;
use super::views::residue::ResidueView;
use crate::info::elements::Element;
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
                    // Create all possible bond combinations
                    for &i in &atom_indices1 {
                        for &j in &atom_indices2 {
                            bonds.push(Bond::new(i as i32, j as i32, bond_type));
                        }
                    }
                }
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
}
