use super::cif;
use super::pdb;
use anyhow::{Context, Result};
use ferritin_core::AtomCollection;
use std::path::Path;

//
pub fn load_structure<P: AsRef<Path>>(file_path: P) -> Result<AtomCollection> {
    let path = file_path.as_ref();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or_else(|| anyhow::anyhow!("File has no extension"))?
        .to_lowercase();

    match extension.as_str() {
        "pdb" => Ok(pdb::PDBFile::read(path)
            .context("Failed to read PDB file")?
            .parse_to_atom_collection()
            .context("Failed to parse PDB file to atom collection")?),
        "cif" => Ok(cif::CIFFile::read(path)?.parse_to_atom_collection()?),
        _ => Err(anyhow::anyhow!("Unsupported file extension: {}", extension)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferritin_test_data::TestFile;

    #[test]
    fn test_load_structure_cif() {
        let (file_path, _handle) = TestFile::protein_01().create_temp().unwrap();
        let result = load_structure(&file_path);
        assert!(
            result.is_ok(),
            "Failed to load CIF file: {:?}",
            result.err()
        );
        let atom_collection = result.unwrap();
        assert_eq!(atom_collection.get_size(), 1413);
        // This includes Water Molecules
        let max_resid = atom_collection.get_resids().iter().max().unwrap_or(&0);
        assert_eq!(*max_resid, 338);
    }

    #[test]
    fn test_load_structure_pdb() {
        let (file_path, _handle) = TestFile::protein_02().create_temp().unwrap();
        let result = load_structure(&file_path);
        assert!(
            result.is_ok(),
            "Failed to load PDB file: {:?}",
            result.err()
        );
        let atom_collection = result.unwrap();
        println!("atom_collection: {:?}", atom_collection.get_coords());
        assert_eq!(atom_collection.get_size(), 1356);
        // This includes Water Molecules
        let max_resid = atom_collection.get_resids().iter().max().unwrap_or(&0);
        assert_eq!(*max_resid, 176);
    }
}
