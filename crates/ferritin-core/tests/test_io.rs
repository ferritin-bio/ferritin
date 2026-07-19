use ferritin_core::{load_structure, load_trajectory};
use ferritin_core::trajectory::Trajectory;
use std::sync::Arc;
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
fn test_load_trajectory_cif_multimodel() {
    let (file_path, _handle) = TestFile::multimodel_nmr_1d3z().create_temp().unwrap();
    let traj = load_trajectory(&file_path).unwrap();

    // 1D3Z: all frames share one Arc<AtomicHierarchy> (topology constant across NMR models)
    assert!(traj.frame_count() > 1, "NMR trajectory must have multiple frames");
    let n_atoms = traj.representative().n_atoms();
    assert!(n_atoms > 0, "Each frame must have atoms");

    // Verify all frames share the same topology
    let h0 = Arc::clone(&traj.frame(0).hierarchy);
    let hlast = Arc::clone(&traj.frame(traj.frame_count() - 1).hierarchy);
    assert!(Arc::ptr_eq(&h0, &hlast), "All frames must share Arc<AtomicHierarchy>");

    // Verify coords differ between frames (NMR models have different coordinates)
    let coord_f0 = traj.frame(0).coord(0);
    let coord_flast = traj.frame(traj.frame_count() - 1).coord(0);
    assert_ne!(coord_f0, coord_flast, "NMR frames must have differing coordinates");
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
