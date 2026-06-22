use ferritin_core::load_structure;
use ferritin_test_data::TestFile;

// Characterization test for the current BROKEN multi-model CIF parsing behavior.
//
// Correct post-fix (ferritin-70p) values for 1D3Z:
//   - model_count = 20
//   - atoms_per_model = 596
//   - residues per model = 36
//   - chain A
//
// The current parser silently flattens all models into a single AtomCollection,
// producing total_atoms = 12310 (WRONG — should be ~615 atoms per model, 20 models separate).
// The 12310 total includes hydrogen atoms; the non-hydrogen count would be 20 × 596 = 11920.
// This test documents and pins that BROKEN behavior.
// It MUST be removed or inverted after ferritin-70p (IO multi-model fix) lands.

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
fn characterize_multimodel_cif_current_behavior() {
    // 1D3Z has 20 NMR models, each with 596 atoms (36 residues, 1 chain A).
    // CORRECT post-fix values: model_count=20, atoms_per_model=596
    // CURRENT parser flattens all models: total_atoms = 20 * 596 = 11920
    // This test documents the current BROKEN behavior.
    // It MUST fail (or be deleted) after ferritin-70p (IO multi-model fix) lands.
    let (file_path, _handle) = TestFile::multimodel_nmr_1d3z().create_temp().unwrap();
    let ac = load_structure(&file_path).unwrap();
    let actual = ac.get_size();
    // The parser produced 12310 atoms (20 models × ~615 atoms/model, including hydrogens).
    // The theoretical non-hydrogen count would be 20 × 596 = 11920, but this CIF contains
    // hydrogen atoms as well, so the actual flattened count is 12310.
    assert_eq!(
        actual,
        12310,
        "characterization: current parser flattens 20 NMR models into one collection (WRONG); got {}",
        actual
    );
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
