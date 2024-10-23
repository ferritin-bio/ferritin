use pseutils::PSEData;

pub struct AtomCollection {
    size: usize,
    coords: Vec<[f32; 3]>,
    resvs: Vec<i32>,
    chains: Vec<String>,
    // atom_type: Vec<String>, // Adding atom type as a fixed field
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

        let resns: Vec<String> = atoms.iter().map(|a| a.resn.to_string()).collect();
        let resvs: Vec<i32> = atoms.iter().map(|a| a.resv).collect();
        let chains: Vec<String> = atoms.iter().map(|a| a.chain.to_string()).collect();

        let ac = AtomCollection {
            size: atoms.len(),
            coords,
            resvs,
            chains,
        };
        ac
    }
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
        assert!(psedata.version == 3000000);
        let names = psedata.get_session_names();
        assert_eq!(names.len(), 2);

        let ac = AtomCollection::from(&psedata);
        assert_eq!(ac.size, 1519);
        assert_eq!(ac.coords.len(), 1519);
    }
}
