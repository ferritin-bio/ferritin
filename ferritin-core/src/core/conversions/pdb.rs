use crate::core::AtomCollection;
use itertools::Itertools;
use pdbtbx::{Element, PDB};

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
            Vec<Element>,
            Vec<String>,
        ) = pdb_data
            .chains()
            .flat_map(|chain| {
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
                                element,
                                chain_id.clone(),
                            )
                        })
                    })
                })
            })
            .multiunzip();

        let mut ac = AtomCollection::new(
            coords.len(),
            coords,
            res_ids,
            res_names,
            is_hetero,
            elements,
            atom_names,
            chain_ids,
            None,
        );

        ac.connect_via_residue_names();
        ac
    }
}

#[cfg(test)]
mod tests {
    use crate::core::AtomCollection;
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
        assert_eq!(ac.get_coords().len(), 1413);
        assert_eq!(ac.get_bonds().unwrap().len(), 1095);

        // 338 Residues
        let res_ids: Vec<i32> = ac.get_resids().into_iter().cloned().unique().collect();
        let res_max = res_ids.iter().max().unwrap();
        assert_eq!(res_max, &338);

        // Check resnames
        let res_names: Vec<String> = ac
            .get_resnames()
            .into_iter()
            .cloned()
            .unique()
            .sorted()
            .collect();
        assert_eq!(
            res_names,
            [
                "ALA", "ARG", "ASN", "ASP", "GLN", "GLU", "GLY", "HEM", "HIS", "HOH", "ILE", "LEU",
                "LYS", "MET", "NBN", "PHE", "PRO", "SER", "SO4", "THR", "TRP", "TYR", "VAL"
            ]
        );

        // Take a peek at the unique elements
        let elements: Vec<Element> = ac
            .get_elements()
            .into_iter()
            .cloned()
            .unique()
            .sorted()
            .collect();
        assert_eq!(
            elements,
            [Element::C, Element::N, Element::O, Element::S, Element::Fe,]
        );
    }
}
