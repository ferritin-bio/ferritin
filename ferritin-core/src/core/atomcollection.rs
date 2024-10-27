use super::constants::get_bonds_canonical20;
use ferritin_pymol::PSEData;
use itertools::izip;
use itertools::Itertools;
use pdbtbx::PDB;

pub struct AtomCollection {
    size: usize,
    coords: Vec<[f32; 3]>,
    res_ids: Vec<i32>,
    res_names: Vec<String>,
    is_hetero: Vec<bool>,
    elements: Vec<String>,
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
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn bonds(&self) -> Option<&Vec<Bond>> {
        self.bonds.as_ref()
    }
    pub fn coords(&self) -> &Vec<[f32; 3]> {
        self.coords.as_ref()
    }
    pub fn iter_coords_and_elements(&self) -> impl Iterator<Item = (&[f32; 3], &String)> {
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

    // ... existing methods ...
    // // Get unique chains
    // pub fn unique_chains(&self) -> HashSet<&String> {
    //     self.chain.iter().collect()
    // }
    // // Filter by chain and atom type
    // pub fn filter_by_chain_and_type<'a>(
    //     &'a self,
    //     chain: &str,
    //     atom_type: &str,
    // ) -> impl Iterator<Item = usize> + 'a {
    //     self.chain
    //         .iter()
    //         .enumerate()
    //         .filter(move |(i, &ref c)| c == chain && self.atom_type[*i] == atom_type)
    //         .map(|(i, _)| i)
    // }
    // // Get atom indices for a specific chain
    // pub fn get_chain_indices<'a>(&'a self, chain: &str) -> impl Iterator<Item = usize> + 'a {
    //     self.chain
    //         .iter()
    //         .enumerate()
    //         .filter(move |(_, c)| *c == chain)
    //         .map(|(i, _)| i)
    // }

    // // Get atom indices for a specific atom type
    // pub fn get_atom_type_indices<'a>(
    //     &'a self,
    //     atom_type: &str,
    // ) -> impl Iterator<Item = usize> + 'a {
    //     self.atom_type
    //         .iter()
    //         .enumerate()
    //         .filter(move |(_, t)| *t == atom_type)
    //         .map(|(i, _)| i)
    // }

    // // Get a view of the atom collection filtered by chain
    // pub fn view_by_chain<'a>(&'a self, chain: &str) -> AtomView<'a> {
    //     let indices: Vec<usize> = self.get_chain_indices(chain).collect();
    //     AtomView {
    //         collection: self,
    //         indices,
    //     }
    // }
}

fn match_bond(bond_int: i32) -> BondOrder {
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

impl From<&PSEData> for AtomCollection {
    fn from(pse_data: &PSEData) -> Self {
        let mols = pse_data.get_molecule_data();

        // Pymol: most of the descriptive data is there
        let atoms: Vec<&ferritin_pymol::pymolparsing::parsing::AtomInfo> =
            mols.iter().flat_map(|mol| mol.atom.iter()).collect();

        // Pymol: coord sets are maintained seperately.
        let coord_sets: Vec<&ferritin_pymol::pymolparsing::parsing::CoordSet> =
            mols.iter().flat_map(|mol| mol.coord_set.iter()).collect();

        let coords: Vec<[f32; 3]> = coord_sets
            .iter()
            .flat_map(|c| c.get_coords_as_vec())
            .collect();

        // Pymol: most of the descriptive data is there
        let pymol_bonds: Vec<&ferritin_pymol::pymolparsing::parsing::Bond> =
            mols.iter().flat_map(|mol| mol.bond.iter()).collect();

        let bonds = pymol_bonds
            .iter()
            .map(|b| Bond {
                atom1: b.index_1,
                atom2: b.index_2,
                order: match_bond(b.order),
            })
            .collect();

        // pull out specific fields
        let (res_names, res_ids, chain_ids, is_hetero, elements, atom_names): (
            Vec<String>,
            Vec<i32>,
            Vec<String>,
            Vec<bool>,
            Vec<String>,
            Vec<String>,
        ) = atoms
            .iter()
            .map(|a| {
                (
                    a.resn.to_string(),
                    a.resv,
                    a.chain.to_string(),
                    a.is_hetatm,
                    a.elem.to_string(),
                    a.name.to_string(),
                )
            })
            .multiunzip();

        AtomCollection {
            size: atoms.len(),
            coords,
            res_names,
            res_ids,
            chain_ids,
            is_hetero,
            elements,
            atom_names,
            bonds: Some(bonds),
        }
    }
}

impl From<&PDB> for AtomCollection {
    // the PDB API requires us to iterate:
    // PDB --> Chain --> Residue --> Atom if we want data from all.
    // Here we collect all the data in one go and return an AtomCollection
    fn from(pdb_data: &PDB) -> Self {
        let (coords, is_hetero, atom_names, res_ids, res_names, elements, chain_ids): (
            Vec<[f32; 3]>,
            Vec<bool>,
            Vec<String>,
            Vec<i32>,
            Vec<String>,
            Vec<String>,
            Vec<String>,
        ) = izip!(pdb_data.chains().flat_map(|chain| {
            let chain_id = chain.id().to_string();
            chain.residues().flat_map(move |residue| {
                let (res_number, _insertion_code) = residue.id();
                let res_id = res_number as i32;
                let res_name = residue.name().unwrap_or_default().to_string();
                let chain_id = chain_id.clone();
                residue.atoms().filter_map(move |atom| {
                    atom.element().map(|element| {
                        let (x, y, z) = atom.pos();
                        (
                            [x as f32, y as f32, z as f32],
                            atom.hetero(),
                            atom.name().to_string(),
                            res_id,
                            res_name.clone(),
                            element.symbol().to_string(),
                            chain_id.clone(),
                        )
                    })
                })
            })
        }))
        .multiunzip();

        let mut ac = AtomCollection {
            size: coords.len(),
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            chain_ids,
            atom_names,
            bonds: None,
        };

        // adds bonds
        ac.connect_via_residue_names();
        ac
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

#[cfg(test)]
mod tests {
    use crate::core::atomcollection::AtomCollection;
    use ferritin_pymol::PSEData;
    use itertools::Itertools;
    use pdbtbx;
    use std::path::PathBuf;

    #[test]
    fn test_pse_from() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let file_path = PathBuf::from(manifest_dir)
            .join("tests")
            .join("data")
            .join("example.pse");

        let psedata = PSEData::load(file_path.to_str().unwrap()).expect("local pse path");

        // check Atom Collection Numbers
        let ac = AtomCollection::from(&psedata);
        assert_eq!(ac.size, 1519);
        assert_eq!(ac.coords.len(), 1519);
        assert_eq!(ac.bonds.unwrap().len(), 1537); // 1537 bonds
    }

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
        let elements: Vec<String> = ac.elements.into_iter().unique().sorted().collect();
        assert_eq!(elements, ["C", "FE", "N", "O", "S"]);
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

        // // check Atom Collection Numbers
        // let ac = AtomCollection::from(&psedata);
        // assert_eq!(ac.coords.len(), 1519);
        // assert_eq!(ac.bonds.unwrap().len(), 1537); // 1537 bonds
    }
}
