use pseutils::PSEData;

pub struct AtomCollection {
    size: usize,
    coords: Vec<[f32; 3]>,
    resvs: Vec<i32>,
    chains: Vec<String>,
    bonds: Option<Vec<Bond>>,
    // atom_type: Vec<String>,
    //
    // // ... other fixed fields
    // dynamic_fields: HashMap<String, Vec<Box<dyn Any>>>,
}

impl AtomCollection {
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

impl From<&PSEData> for AtomCollection {
    fn from(pse_data: &PSEData) -> Self {
        let mols = pse_data.get_molecule_data();

        // Pymol: most of the descriptive data is there
        let atoms: Vec<&pseutils::pymolparsing::parsing::AtomInfo> =
            mols.iter().flat_map(|mol| mol.atom.iter()).collect();

        // Pymol: coord sets are maintained seperately.
        let coord_sets: Vec<&pseutils::pymolparsing::parsing::CoordSet> =
            mols.iter().flat_map(|mol| mol.coord_set.iter()).collect();

        let coords: Vec<[f32; 3]> = coord_sets
            .iter()
            .flat_map(|c| c.get_coords_as_vec())
            .collect();

        // Pymol: most of the descriptive data is there
        let pymol_bonds: Vec<&pseutils::pymolparsing::parsing::Bond> =
            mols.iter().flat_map(|mol| mol.bond.iter()).collect();

        let bonds = pymol_bonds
            .iter()
            .map(|b| Bond {
                atom1: b.index_1,
                atom2: b.index_2,
                order: match b.order {
                    0 => BondOrder::Unset,
                    1 => BondOrder::Single,
                    2 => BondOrder::Double,
                    3 => BondOrder::Triple,
                    4 => BondOrder::Quadruple,
                    _ => {
                        println!("Bond Order not found: {:?}", b.order);
                        panic!()
                    }
                },
            })
            .collect();

        // specific fields
        let resns: Vec<String> = atoms.iter().map(|a| a.resn.to_string()).collect();
        let resvs: Vec<i32> = atoms.iter().map(|a| a.resv).collect();
        let chains: Vec<String> = atoms.iter().map(|a| a.chain.to_string()).collect();

        let ac = AtomCollection {
            size: atoms.len(),
            coords,
            resvs,
            chains,
            bonds: Some(bonds),
        };
        ac
    }
}

pub struct Bond {
    atom1: i32,
    atom2: i32,
    order: BondOrder,
    // id
    // stereo
    // unique_id
    // has_setting
}

#[repr(u8)]
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

// // A view struct to represent a subset of atoms
// pub struct AtomView<'a> {
//     collection: &'a AtomCollection,
//     indices: Vec<usize>,
// }

// impl<'a> AtomView<'a> {
//     pub fn len(&self) -> usize {
//         self.indices.len()
//     }

//     pub fn get_coord(&self, index: usize) -> Option<&[f32; 3]> {
//         self.indices
//             .get(index)
//             .and_then(|&i| self.collection.coord.get(i))
//     }

//     // ... other methods to access fields for the viewed atoms
// }

#[cfg(test)]
mod tests {
    use crate::core::atomcollection::AtomCollection;

    #[test]
    fn test_PSE_from() {
        use pseutils::PSEData;
        let psedata = PSEData::load("tests/data/example.pse").expect("local pse path");

        // check Atom Collection Numbers
        let ac = AtomCollection::from(&psedata);
        assert_eq!(ac.size, 1519);
        assert_eq!(ac.coords.len(), 1519);
        assert_eq!(ac.bonds.unwrap().len(), 1537); // 1537 bonds
    }
}
