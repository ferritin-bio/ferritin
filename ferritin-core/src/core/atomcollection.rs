use super::constants::get_bonds_canonical20;
use crate::conversions;
use itertools::izip;
use itertools::Itertools;
use pdbtbx::Element;
use std::ops::BitAnd; // import

pub struct AtomSelector<'a> {
    collection: &'a AtomCollection,
    current_selection: Selection,
}

impl<'a> AtomSelector<'a> {
    // Chainable methods
    pub fn chain(mut self, chain_id: &str) -> Self {
        let chain_selection = self.collection.select_by_chain(chain_id);
        self.current_selection = &self.current_selection & &chain_selection;
        self
    }

    pub fn residue(mut self, res_name: &str) -> Self {
        let res_selection = self.collection.select_by_residue(res_name);
        self.current_selection = &self.current_selection & &res_selection;
        self
    }

    pub fn element(mut self, element: Element) -> Self {
        let element_selection = self
            .collection
            .elements
            .iter()
            .enumerate()
            .filter(|(_, &e)| e == element)
            .map(|(i, _)| i)
            .collect();
        self.current_selection = &self.current_selection & &Selection::new(element_selection);
        self
    }

    pub fn sphere(mut self, center: [f32; 3], radius: f32) -> Self {
        let sphere_selection = self
            .collection
            .coords
            .iter()
            .enumerate()
            .filter(|(_, &pos)| {
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                (dx * dx + dy * dy + dz * dz).sqrt() <= radius
            })
            .map(|(i, _)| i)
            .collect();
        self.current_selection = &self.current_selection & &Selection::new(sphere_selection);
        self
    }

    // Custom predicate selection
    pub fn filter<F>(mut self, predicate: F) -> Self
    where
        F: Fn(usize) -> bool,
    {
        let filtered = self
            .current_selection
            .indices
            .iter()
            .filter(|&&idx| predicate(idx))
            .copied()
            .collect();
        self.current_selection = Selection::new(filtered);
        self
    }

    // Finalize the selection and create a view
    pub fn collect(&self) -> AtomView {
        AtomView {
            collection: self.collection,
            selection: &self.current_selection,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Selection {
    indices: Vec<usize>,
}

impl Selection {
    fn new(indices: Vec<usize>) -> Self {
        Selection { indices }
    }

    // Combine selections using & operator
    fn and(&self, other: &Selection) -> Selection {
        let indices: Vec<usize> = self
            .indices
            .iter()
            .filter(|&&idx| other.indices.contains(&idx))
            .cloned()
            .collect();
        Selection::new(indices)
    }
}

impl BitAnd for &Selection {
    type Output = Selection;

    fn bitand(self, other: Self) -> Selection {
        self.and(other)
    }
}

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
    pub fn select(&self) -> AtomSelector {
        AtomSelector {
            collection: self,
            current_selection: Selection::new((0..self.size).collect()),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }
    pub fn bonds(&self) -> Option<&Vec<Bond>> {
        self.bonds.as_ref()
    }
    pub fn coords(&self) -> &Vec<[f32; 3]> {
        self.coords.as_ref()
    }
    pub fn iter_coords_and_elements(&self) -> impl Iterator<Item = (&[f32; 3], &Element)> {
        izip!(&self.coords, &self.elements)
    }
    pub fn calculate_displacement(&self) {
        // Measure the displacement vector, i.e. the vector difference, from
        // one array of atom coordinates to another array of coordinates.
        unimplemented!()
    }

    pub fn calculate_distance(&self, atoms: AtomCollection) {
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
                            bonds.push(Bond {
                                atom1: i as i32,
                                atom2: j as i32,
                                order: match_bond(bond_type),
                            });
                        }
                    }
                }
            }
        }
        // Update self.bonds
        self.bonds = Some(bonds);
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

    pub fn view<'a, 'b>(&'a self, selection: &'b Selection) -> AtomView<'a, 'b> {
        AtomView {
            collection: self,
            selection,
        }
    }
}

pub struct AtomView<'a, 'b> {
    collection: &'a AtomCollection,
    selection: &'b Selection,
}

impl<'a, 'b> IntoIterator for &'a AtomView<'a, 'b> {
    type Item = AtomRef<'a>;
    type IntoIter = AtomIterator<'a, 'b>;

    fn into_iter(self) -> Self::IntoIter {
        AtomIterator {
            view: self,
            current: 0,
        }
    }
}

pub struct AtomRef<'a> {
    pub coords: &'a [f32; 3],
    pub res_id: &'a i32,
    pub res_name: &'a String,
    pub element: &'a Element,
    // ... other fields
}

pub struct AtomIterator<'a, 'b> {
    view: &'a AtomView<'a, 'b>,
    current: usize,
}

impl<'a, 'b> Iterator for AtomIterator<'a, 'b> {
    type Item = AtomRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.view.selection.indices.len() {
            return None;
        }

        let idx = self.view.selection.indices[self.current];
        self.current += 1;

        Some(AtomRef {
            coords: &self.view.collection.coords[idx],
            res_id: &self.view.collection.res_ids[idx],
            res_name: &self.view.collection.res_names[idx],
            element: &self.view.collection.elements[idx],
        })
    }
}

impl<'a, 'b> AtomView<'a, 'b> {
    pub fn coords(&self) -> Vec<[f32; 3]> {
        self.selection
            .indices
            .iter()
            .map(|&i| self.collection.coords[i])
            .collect()
    }

    pub fn size(&self) -> usize {
        self.selection.indices.len()
    }
}

/// Bond
#[derive(Debug, PartialEq)]
pub struct Bond {
    atom1: i32,
    atom2: i32,
    order: BondOrder,
    // id
    // stereo
    // unique_id
    // has_setting
}

impl Bond {
    pub fn new(atom1: i32, atom2: i32, order: BondOrder) -> Self {
        Bond {
            atom1,
            atom2,
            order,
        }
    }
    pub fn get_atom_indices(&self) -> (i32, i32) {
        (self.atom1, self.atom2)
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq)]
/// BondOrder:
/// https://www.biotite-python.org/latest/apidoc/biotite.structure.BondType.html#biotite.structure.BondType
/// see also: http://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/Bond.html
pub enum BondOrder {
    /// Used if the actual type is unknown
    Unset,
    /// Single bond
    Single,
    /// Double bond
    Double,
    /// Triple bond
    Triple,
    /// A quadruple bond
    Quadruple,
}

impl BondOrder {
    pub fn match_bond(bond_int: i32) -> BondOrder {
        match bond_int {
            0 => BondOrder::Unset,
            1 => BondOrder::Single,
            2 => BondOrder::Double,
            3 => BondOrder::Triple,
            4 | 5 | 6 => BondOrder::Quadruple,
            _ => {
                println!("Bond Order not found: {}", bond_int);
                panic!()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::atomcollection::AtomCollection;
    use ferritin_pymol::PSEData;
    use itertools::Itertools;
    use pdbtbx::{self, Element};
    use std::path::PathBuf;

    #[test]
    fn test_pdb_from() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let file_path = PathBuf::from(manifest_dir)
            .join("tests")
            .join("data")
            .join("101m.cif");

        let (pdb_data, _errors) = pdbtbx::open(file_path.to_str().unwrap()).unwrap();
        assert_eq!(pdb_data.atom_count(), 1413);

        // check Atom Collection Numbers
        let ac = AtomCollection::from(&pdb_data);
        assert_eq!(ac.coords.len(), 1413);
        assert_eq!(ac.bonds().unwrap().len(), 1095);

        // 338 Residues
        let res_ids: Vec<i32> = ac.res_ids.into_iter().unique().collect();
        let res_max = res_ids.iter().max().unwrap();
        assert_eq!(res_max, &338);

        // Check resnames
        let res_names: Vec<String> = ac.res_names.into_iter().unique().sorted().collect();
        assert_eq!(
            res_names,
            [
                "ALA", "ARG", "ASN", "ASP", "GLN", "GLU", "GLY", "HEM", "HIS", "HOH", "ILE", "LEU",
                "LYS", "MET", "NBN", "PHE", "PRO", "SER", "SO4", "THR", "TRP", "TYR", "VAL"
            ]
        );

        // Take a peek at the unique elements
        let elements: Vec<Element> = ac.elements.into_iter().unique().sorted().collect();
        assert_eq!(
            elements,
            [Element::C, Element::N, Element::O, Element::S, Element::Fe,]
        );
    }

    #[test]
    fn test_addbonds() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let file_path = PathBuf::from(manifest_dir)
            .join("tests")
            .join("data")
            .join("101m.cif");

        let (pdb, _errors) = pdbtbx::open(file_path.to_str().unwrap()).unwrap();
        assert_eq!(pdb.atom_count(), 1413);
    }
}
