use super::bonds::{Bond, BondOrder};
use super::info::constants::{
    get_bonds_canonical20, is_amino_acid, is_carbohydrate, is_nucleotide,
};
use crate::core::residue::{ResidueAtoms, ResidueIter};
use crate::core::selection::{AtomSelector, AtomView, Selection};
use itertools::{izip, Itertools};
use pdbtbx::Element;
use std::iter::FromIterator;

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
        AtomCollection {
            size,
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            atom_names,
            chain_ids,
            bonds,
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
        //     Alternatively an ndarray containing the coordinates can be
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

        println!("Number of residues: {}", residue_starts.len());
        println!("First few residue names: {:?}", &self.res_names[..5]);
        println!("First few atom names: {:?}", &self.atom_names[..5]);

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
        // Update self.bonds
        println!("Updating bonds....");
        self.bonds = Some(bonds);
    }
    pub fn connect_via_distance(&self) -> Vec<Bond> {
        // connect_via_distances(atoms, distance_range=None, atom_mask=None,
        //                           inter_residue=True, default_bond_type=BondType.ANY,
        //                           periodic=False)

        //     Create a :class:`BondList` for a given atom array, based on
        //     pairwise atom distances.

        //     A :attr:`BondType.ANY`, bond is created for two atoms within the
        //     same residue, if the distance between them is within the expected
        //     bond distance range.
        //     Bonds between two adjacent residues are created for the atoms
        //     expected to connect these residues, i.e. ``'C'`` and ``'N'`` for
        //     peptides and ``"O3'"`` and ``'P'`` for nucleotides.

        //     Parameters
        //     ----------
        //     atoms : AtomArray
        //         The structure to create the :class:`BondList` for.
        //     distance_range : dict of tuple(str, str) -> tuple(float, float), optional
        //         Custom minimum and maximum bond distances.
        //         The dictionary keys are tuples of chemical elements representing
        //         the atoms to be potentially bonded.
        //         The order of elements within each tuple does not matter.
        //         The dictionary values are the minimum and maximum bond distance,
        //         respectively, for the given combination of elements.
        //         This parameter updates the default dictionary.
        //         Hence, the default bond distances for missing element pairs are
        //         still taken from the default dictionary.
        //         The default bond distances are taken from :footcite:`Allen1987`.
        //     inter_residue : bool, optional
        //         If true, connections between consecutive amino acids and
        //         nucleotides are also added.
        //     default_bond_type : BondType or int, optional
        //         By default, all created bonds have :attr:`BondType.ANY`.
        //         An alternative :class:`BondType` can be given in this parameter.
        //     periodic : bool, optional
        //         If set to true, bonds can also be detected in periodic
        //         boundary conditions.
        //         The `box` attribute of `atoms` is required in this case.

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
    pub fn get_elements(&self) -> &Vec<pdbtbx::Element> {
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
    pub(crate) fn get_residue_starts(&self) -> Vec<i64> {
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
    pub fn iter_coords_and_elements(&self) -> impl Iterator<Item = (&[f32; 3], &Element)> {
        izip!(&self.coords, &self.elements)
    }
    /// IterResiudees Will Iterate Through the AtomCollection one Residue at a time.
    ///
    /// This is the base for any onther residue filtration code.
    pub fn iter_residues_all(&self) -> ResidueIter {
        ResidueIter::new(self, self.get_residue_starts())
    }
    pub fn iter_residues_aminoacid(&self) -> impl Iterator<Item = ResidueAtoms> {
        self.iter_residues_all()
            .filter(|residue| is_amino_acid(&residue.res_name))
    }
    pub fn select(&self) -> AtomSelector {
        AtomSelector::new(self)
    }
    pub fn select_by_chain(&self, chain_id: &str) -> Selection {
        let indices: Vec<usize> = self
            .chain_ids
            .iter()
            .enumerate()
            .filter(|(_, &ref chain)| chain == chain_id)
            .map(|(i, _)| i)
            .collect();
        Selection::new(indices)
    }
    pub fn select_by_residue(&self, res_name: &str) -> Selection {
        let indices: Vec<usize> = self
            .res_names
            .iter()
            .enumerate()
            .filter(|(_, name)| name.as_str() == res_name)
            .map(|(i, _)| i)
            .collect();
        Selection::new(indices)
    }
    pub fn view(&self, selection: Selection) -> AtomView {
        AtomView::new(self, selection)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::test_utilities::get_atom_container;
    use crate::AtomCollection;
    use pdbtbx::Element;

    #[test]
    fn test_selection_api() {
        let ac: AtomCollection = get_atom_container();
        let selected_atoms = ac
            .select()
            .chain("A")
            .residue("GLY")
            .element(Element::C)
            .collect();
        assert_eq!(selected_atoms.size(), 22);

        // let carbon_coords: Vec<[f32; 3]> = selected_atoms
        //     .into_iter()
        //     .filter(|atom| *atom.element == Element::C)
        //     .map(|atom| *atom.coords)
        //     .collect();
    }

    #[test]
    fn test_residue_iterator() {
        let ac: AtomCollection = get_atom_container();
        assert_eq!(ac.get_size(), 1413);

        // This includes Water Molecules
        let max_resid = ac.get_resids().iter().max().unwrap_or(&0);
        assert_eq!(*max_resid, 338);

        // this fn is only available in-crate
        // let residue_breaks = ac.get_residue_starts();
        // assert_eq!(residue_breaks, vec![1, 2, 3]);

        // This is counting 294 - I expect
        // let residue_count = ac.iter_residues().count();
        // assert_eq!(residue_count, 154);
        //

        // Water count -> 139
        //

        for res in ac.iter_residues_all() {
            println!("{:?}", res.res_name)
        }
    }
}
