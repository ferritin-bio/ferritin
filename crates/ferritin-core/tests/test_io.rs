use ferritin_core::trajectory::Trajectory;
use ferritin_core::{AtomCollection, load_structure, load_structure_from_string, load_trajectory};
use ferritin_test_data::TestFile;
use std::sync::Arc;

#[test]
fn test_cif_ingest_preserves_long_names_and_split_rows() {
    // Exceed inline string capacity, and split a row across physical lines.
    // Exercise each loop completion path without changing the parsed atoms.
    let chain = "author_chain_identifier_longer_than_twenty_four_bytes";
    let atom = "atom_identifier_longer_than_twenty_four_bytes";
    for ending in [
        "",
        "#\n",
        "loop_\n_audit.id\n_audit.details\n1 'after atoms'\n",
    ] {
        let content = format!(
            "data_long_names\n\
             loop_\n_entry.id\n_entry.details\n1 'a quoted value longer than twenty four bytes'\n#\n\
             loop_\n\
             _atom_site.id\n_atom_site.type_symbol\n_atom_site.label_atom_id\n\
             _atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n\
             _atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n\
             1 C '{atom}' AABCD '{chain}' 7\n1.25 -2.5 3.75\n\
             2 O OXT HOH B 19 -4.0 5.5 6.25\n{ending}"
        );
        let ac = load_structure_from_string(&content, "cif").unwrap();
        assert_eq!(ac.get_size(), 2);
        assert_eq!(ac.get_atom_name(0), atom);
        assert_eq!(ac.get_res_name(0), "AABCD");
        assert_eq!(ac.get_chain_id(0), chain);
        assert_eq!(*ac.get_res_id(0), 7);
        assert_eq!(ac.get_atom_name(1), "OXT");
        assert_eq!(ac.get_res_name(1), "HOH");
        assert_eq!(ac.get_chain_id(1), "B");
        assert_eq!(*ac.get_res_id(1), 19);
        assert_eq!(ac.get_coords(), &[[1.25, -2.5, 3.75], [-4.0, 5.5, 6.25]]);

        // The optimized Model conversion must preserve heap-backed and inline names.
        let converted = AtomCollection::from(&ac.to_model());
        for i in 0..2 {
            assert_eq!(converted.get_atom_name(i), ac.get_atom_name(i));
            assert_eq!(converted.get_res_name(i), ac.get_res_name(i));
            assert_eq!(converted.get_chain_id(i), ac.get_chain_id(i));
            assert_eq!(converted.get_res_id(i), ac.get_res_id(i));
        }
        assert_eq!(converted.get_coords(), ac.get_coords());
    }
}

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
    assert!(
        traj.frame_count() > 1,
        "NMR trajectory must have multiple frames"
    );
    let n_atoms = traj.representative().n_atoms();
    assert!(n_atoms > 0, "Each frame must have atoms");

    // Verify all frames share the same topology
    let h0 = Arc::clone(&traj.frame(0).hierarchy);
    let hlast = Arc::clone(&traj.frame(traj.frame_count() - 1).hierarchy);
    assert!(
        Arc::ptr_eq(&h0, &hlast),
        "All frames must share Arc<AtomicHierarchy>"
    );

    // Verify coords differ between frames (NMR models have different coordinates)
    let coord_f0 = traj.frame(0).coord(0);
    let coord_flast = traj.frame(traj.frame_count() - 1).coord(0);
    assert_ne!(
        coord_f0, coord_flast,
        "NMR frames must have differing coordinates"
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
