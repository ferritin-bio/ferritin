//! AtomCollection
//!
//! An AtomCollection is primarily a group of atoms with some atomic properties like coordinates, element type
//! and residue information. Additional data like bonds can be added post-instantiation.
//! The data for residues within this collection can be iterated through. Other useful queries like inter-atomic
//! distances are supported.
use super::bonds::{Bond, BondOrder};
use super::info::constants::get_bonds_canonical20;
use super::views::chain::ChainView;
use super::views::residue::ResidueView;
// use crate::residue::{ResidueAtoms, ResidueIter};
// use crate::selection::{AtomSelector, AtomView, Selection};
use itertools::{Itertools, izip};
use pdbtbx::Element;

/// Atom Collection
///
/// The core data structure of ferritin-core.
///
/// it strives to be simple, high performance, and extensible using
/// traits.
///
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
    residue_start_indices: Option<Vec<i32>>,
    chain_start_indices: Option<Vec<i32>>,
    // atom_type: Vec<String>,
    // // ... other fixed fields
    // dynamic_fields: HashMap<String, Vec<Box<dyn Any>>>,
    // //         self.add_annotation("chain_id", dtype="U4")
    // self.add_annotation("res_id", dtype=int)
    // self.add_annotation("ins_code", dtype="U1")  <- what is this?
    // self.add_annotation("res_name", dtype="U5")
    // self.add_annotation("hetero", dtype=bool)
    // self.add_annotation("atom_name", dtype="U6")
    // self.add_annotation("element", dtype="U2")
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
            // First ensure we have residue indices calculated
            if self.residue_start_indices.is_none() {
                let residue_starts = self.get_residue_starts();
                self.residue_start_indices =
                    Some(residue_starts.iter().map(|&idx| idx as i32).collect());
            }

            // Get chain starts as residue indices
            let residue_starts = self.residue_start_indices.as_ref().unwrap();
            let chain_starts: Vec<i32> = self
                .get_chain_starts()
                .iter()
                .map(|&atom_idx| {
                    // Find the residue index that contains this atom
                    let residue_idx = residue_starts
                        .iter()
                        .enumerate()
                        .filter(|&(_, &res_start)| res_start as usize <= atom_idx)
                        .last()
                        .map(|(i, _)| i as i32)
                        .unwrap_or(0);
                    residue_idx
                })
                .collect();

            self.chain_start_indices = Some(chain_starts);
        }
    }
    pub fn calculate_displacement(&self) {
        // Measure the displacement vector, i.e. the vector difference, from
        // one array of atom coordinates to another array of coordinates.
        unimplemented!()
    }
    pub fn calculate_distance(&self, _atoms: AtomCollection) {
        // def distance(atoms1, atoms2, box=None):
        // """
        // Measure the euclidian distance between atoms.

        // Parameters
        // ----------
        // atoms1, atoms2 : ndarray or Atom or AtomArray or AtomArrayStack
        //     The atoms to measure the distances between.
        //     The dimensions may vary.
        //     Alternatively, a ndarray containing the coordinates can be
        //     provided.
        //     Usual *NumPy* broadcasting rules apply.
        // box : ndarray, shape=(3,3) or shape=(m,3,3), optional
        //     If this parameter is set, periodic boundary conditions are
        //     taken into account (minimum-image convention), based on
        //     the box vectors given with this parameter.
        //     The shape *(m,3,3)* is only allowed, when the input coordinates
        //     comprise multiple models.

        // Returns
        // -------
        // dist : float or ndarray
        //     The atom distances.
        //     The shape is equal to the shape of the input `atoms` with the
        //     highest dimensionality minus the last axis.

        // See also
        // --------
        // index_distance
        // """
        // diff = displacement(atoms1, atoms2, box)
        // return np.sqrt(vector_dot(diff, diff))
        unimplemented!()
    }

    pub fn connect_via_residue_names(&mut self) {
        if self.bonds.is_some() {
            println!("Bonds already in place. Not overwriting.");
            return;
        }
        let aa_bond_info = get_bonds_canonical20();
        let residue_starts = self.get_residue_starts();
        // Iterate through residues
        let mut bonds = Vec::new();
        for res_i in 0..residue_starts.len() - 1 {
            let curr_start_i = residue_starts[res_i] as usize;
            let next_start_i = residue_starts[res_i + 1] as usize;
            if let Some(bond_dict_for_res) =
                aa_bond_info.get(&self.res_names[curr_start_i].as_str())
            {
                // Iterate through bonds in this residue
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
                            bonds.push(Bond::new(
                                i as i32,
                                j as i32,
                                BondOrder::match_bond(bond_type),
                            ));
                        }
                    }
                }
            }
        }
        self.bonds = Some(bonds);
    }
    pub fn connect_via_distance(&self) -> Vec<Bond> {
        // note: was intending to follow Biotite's algo
        unimplemented!()
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
    fn get_residue_starts(&self) -> Vec<i64> {
        let mut starts = vec![0];

        starts.extend(
            izip!(&self.res_ids, &self.res_names, &self.chain_ids)
                .tuple_windows()
                .enumerate()
                .filter_map(
                    |(i, ((res_id1, name1, chain1), (res_id2, name2, chain2)))| {
                        if res_id1 != res_id2 || name1 != name2 || chain1 != chain2 {
                            Some((i + 1) as i64)
                        } else {
                            None
                        }
                    },
                ),
        );
        starts
    }

    pub fn get_residue_start_indices(&self) -> Option<&Vec<i32>> {
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
            Some(indices) => indices.clone(), // Clone to avoid reference type issues
            None => {
                // This is suboptimal as it recalculates every time if not pre-calculated
                // Default to a single chain if none calculated
                vec![0]
            }
        };

        (0..chain_starts.len()).map(move |i| {
            let start_residue_idx = chain_starts[i] as usize;
            let end_residue_idx = if i + 1 < chain_starts.len() {
                chain_starts[i + 1] as usize
            } else {
                // If it's the last chain, go to the end of the structure
                match &self.residue_start_indices {
                    Some(indices) => indices.len(),
                    None => self.size, // Fallback if residue indices not calculated
                }
            };

            ChainView {
                data: self,
                start_residue_idx,
                end_residue_idx,
            }
        })
    }

    // Add to AtomCollection implementation
    pub fn iter_residues(&self) -> Box<dyn Iterator<Item = ResidueView<'_>> + '_> {
        let residue_starts = self.get_residue_starts();
        let atom_starts: Vec<usize> = residue_starts.iter().map(|&idx| idx as usize).collect();

        // Check if we have residues
        if atom_starts.is_empty() {
            return Box::new(std::iter::empty());
        }

        // Get the last atom index before moving atom_starts
        let last_idx = atom_starts.len() - 1;
        let last_atom_idx = atom_starts[last_idx];
        let atom_size = self.get_size();

        // Create iterators for all but the last residue
        let main_residues = (0..atom_starts.len() - 1)
            .map(move |i| ResidueView::new(self, atom_starts[i], atom_starts[i + 1]));

        // Handle the last residue separately - using the saved value
        let last_residue = std::iter::once(ResidueView::new(self, last_atom_idx, atom_size));

        // Chain the two iterators
        Box::new(main_residues.chain(last_residue))
    }

    /// Iterates over amino acid residues in the collection
    ///
    /// Returns a filtered iterator that only includes standard amino acid residues
    pub fn iter_residues_aminoacid(&self) -> impl Iterator<Item = ResidueView<'_>> + '_ {
        self.iter_residues()
            .filter(|residue| residue.is_amino_acid())
    }

    // pub fn iter_residues_aminoacid(&self) -> impl Iterator<Item = ResidueAtoms> {
    //     self.iter_residues_all()
    //         .filter(|residue| residue.is_amino_acid())
    // }
    // pub fn select(&self) -> AtomSelector {
    //     AtomSelector::new(self)
    // }
    // pub fn select_by_chain(&self, chain_id: &str) -> Selection {
    //     let indices: Vec<usize> = self
    //         .chain_ids
    //         .iter()
    //         .enumerate()
    //         .filter(|&(_, &ref chain)| chain == chain_id)
    //         .map(|(i, _)| i)
    //         .collect();
    //     Selection::new(indices)
    // }
    // pub fn select_by_residue(&self, res_name: &str) -> Selection {
    //     let indices: Vec<usize> = self
    //         .res_names
    //         .iter()
    //         .enumerate()
    //         .filter(|(_, name)| name.as_str() == res_name)
    //         .map(|(i, _)| i)
    //         .collect();
    //     Selection::new(indices)
    // }
    // pub fn view(&self, selection: Selection) -> AtomView {
    //     AtomView::new(self, selection)
    // }
}

#[cfg(test)]
mod tests {
    use crate::AtomCollection;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_residue_iterator() {
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        let ac = AtomCollection::from(&pdb);
        assert_eq!(ac.get_size(), 1413);
        // This includes Water Molecules
        let max_resid = ac.get_resids().iter().max().unwrap_or(&0);
        assert_eq!(*max_resid, 338);
        // this fn is only available in-crate
        // let residue_breaks = ac.get_residue_starts();
        // assert_eq!(residue_breaks, vec![1, 2, 3]);
    }

    #[test]
    fn test_chain_iterator() {
        let (prot_file, _temp) = TestFile::protein_04().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        let mut ac = AtomCollection::from(&pdb);

        // Calculate indices first
        ac.calculate_chain_indices();

        // Test chain iteration
        let chains: Vec<_> = ac.iter_chains().collect();
        assert_eq!(chains.len(), 2);
        // Check chain IDs
        assert_eq!(chains[0].chain_id(), "A");
        assert_eq!(chains[1].chain_id(), "B");

        // Check residue counts
        let chain_a_residue_count = chains[0].residue_count();
        let chain_b_residue_count = chains[1].residue_count();
        assert_eq!(chain_a_residue_count, 123);
        assert_eq!(chain_b_residue_count, 103);
        assert_eq!(
            chain_a_residue_count + chain_b_residue_count,
            ac.get_residue_start_indices().unwrap().len()
        );
    }
    #[test]
    fn test_atom_collection_iter_residues() {
        let (prot_file, _temp) = TestFile::protein_01().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        let ac = AtomCollection::from(&pdb);

        // Test residue iteration
        let residues: Vec<_> = ac.iter_residues().collect();

        // Verify residue count
        assert!(!residues.is_empty());

        // Check the first residue
        let first_residue = &residues[0];
        assert!(first_residue.atom_count() > 0);

        // Check that all atoms in a residue have the same residue ID and name
        let res_id = first_residue.residue_id();
        let res_name = first_residue.residue_name();
        assert_eq!(res_id, 0);
        assert_eq!(res_name, "MET");

        // Count of atoms in all residues should match total atom count
        let total_atoms_in_residues: usize = residues.iter().map(|r| r.atom_count()).sum();
        assert_eq!(total_atoms_in_residues, ac.get_size());
    }

    #[test]
    fn test_chain_iter_residues() {
        let (prot_file, _temp) = TestFile::protein_04().create_temp().unwrap();
        let (pdb, _) = pdbtbx::open(prot_file).unwrap();
        let mut ac = AtomCollection::from(&pdb);
        ac.calculate_chain_indices();

        // Get the first chain
        let chains: Vec<_> = ac.iter_chains().collect();
        let first_chain = &chains[0];

        // Test residue iteration within a chain
        let residues: Vec<_> = first_chain.iter_residues().collect();
        assert!(!residues.is_empty());

        // All residues should be in the same chain
        let chain_id = first_chain.chain_id();
        for residue in &residues {
            assert_eq!(residue.chain_id(), chain_id);
        }

        // The number of residues should match chain's residue count
        assert_eq!(residues.len(), first_chain.residue_count());
    }
}
