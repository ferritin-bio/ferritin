use ferritin_core::{load_structure}
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
    assert_eq!(atom_collection.get_size(), 1356);
    let max_resid = atom_collection.get_resids().iter().max().unwrap_or(&0);
    assert_eq!(*max_resid, 176);
}
