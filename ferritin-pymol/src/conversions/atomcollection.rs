use ferritin_core::{AtomCollection, Bond, BondOrder};
use crate::{PSEData,pymolparsing};
use itertools::Itertools;
use pdbtbx::Element;

impl From<&PSEData> for AtomCollection {
    fn from(pse_data: &PSEData) -> Self {
        let mols = pse_data.get_molecule_data();

        // Pymol: most of the descriptive data is there
        let atoms: Vec<&pymolparsing::parsing::AtomInfo> =
            mols.iter().flat_map(|mol| mol.atom.iter()).collect();

        // Pymol: coord sets are maintained seperately.
        let coord_sets: Vec<&pymolparsing::parsing::CoordSet> =
            mols.iter().flat_map(|mol| mol.coord_set.iter()).collect();

        let coords: Vec<[f32; 3]> = coord_sets
            .iter()
            .flat_map(|c| c.get_coords_as_vec())
            .collect();

        // Pymol: most of the descriptive data is there
        let pymol_bonds: Vec<&pymolparsing::parsing::Bond> =
            mols.iter().flat_map(|mol| mol.bond.iter()).collect();

        let bonds = pymol_bonds
            .iter()
            .map(|b| Bond::new(b.index_1, b.index_2, BondOrder::match_bond(b.order)))
            .collect();

        // pull out specific fields
        let (res_names, res_ids, chain_ids, is_hetero, elements, atom_names): (
            Vec<String>,
            Vec<i32>,
            Vec<String>,
            Vec<bool>,
            Vec<Element>,
            Vec<String>,
        ) = atoms
            .iter()
            .map(|a| {
                (
                    a.resn.to_string(),
                    a.resv,
                    a.chain.to_string(),
                    a.is_hetatm,
                    a.elem,
                    a.name.to_string(),
                )
            })
            .multiunzip();

        AtomCollection::new(
            atoms.len(),
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            chain_ids,
            atom_names,
            Some(bonds), //bonds
        )
    }
}

#[cfg(test)]
mod tests {
    use ferritin_core::AtomCollection;
    use crate::PSEData;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_pse_from() {
        let (pymol_file, _temp) = TestFile::pymol_01().create_temp().unwrap();
        let psedata = PSEData::load(&pymol_file).expect("local pse path");

        // check Atom Collection Numbers
        let ac = AtomCollection::from(&psedata);
        assert_eq!(ac.get_size(), 1519);
        assert_eq!(ac.get_coords().len(), 1519);
        assert_eq!(ac.get_bonds().unwrap().len(), 1537); // 1537 bonds
    }
}
