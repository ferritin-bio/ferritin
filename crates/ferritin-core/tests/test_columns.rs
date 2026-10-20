use ferritin_core::columns::{
    AtomHandle, BoundaryErrorKind, GpuBuffer, PositionLayout, ResidueHandle, TopologyColumns,
};
use ferritin_core::trajectory::Trajectory;
use ferritin_core::{load_model, load_trajectory};
use ferritin_test_data::TestFile;

#[test]
fn test_columns_roundtrip_real_structure() {
    let (file_path, _handle) = TestFile::protein_01().create_temp().unwrap();
    let model = load_model(&file_path).unwrap();
    let topology = TopologyColumns::from_model(&model).unwrap();
    let hierarchy = &model.hierarchy;

    assert_eq!(topology.n_atoms() as usize, model.n_atoms());
    assert_eq!(topology.n_residues() as usize, model.n_residues());
    assert_eq!(topology.n_chains() as usize, model.n_chains());

    // Dictionaries decode back to the original strings, with far fewer unique values.
    for (i, name) in hierarchy.atoms.atom_name.iter().enumerate() {
        assert_eq!(topology.atom_name().get(i), Some(name.as_str()));
    }
    for (i, comp) in hierarchy.residues.comp_id.iter().enumerate() {
        assert_eq!(topology.comp_id().get(i), Some(comp.as_str()));
    }
    assert!(topology.atom_name().n_values() < 100);

    for (i, element) in hierarchy.atoms.element.iter().enumerate() {
        assert_eq!(topology.element()[i] as usize, element.atomic_number());
    }

    // Handle navigation agrees with the hierarchy.
    let last = AtomHandle(topology.n_atoms() - 1);
    let residue = topology.residue_of(last).unwrap();
    assert_eq!(residue.index(), hierarchy.residue_of_atom(last.index()));
    assert!(topology.atoms_of(residue).unwrap().contains(&last.0));
    assert_eq!(
        topology
            .chain_of(ResidueHandle(topology.n_residues()))
            .unwrap_err()
            .kind(),
        BoundaryErrorKind::IndexOutOfRange
    );

    // Every column converts to a GPU buffer whose bytes match its layout.
    let frame = topology.frame(&model).unwrap();
    for column in topology.columns().into_iter().chain(frame.columns()) {
        let buffer = GpuBuffer::from_column(column).unwrap();
        assert_eq!(
            buffer.bytes.len() as u64,
            buffer.layout.byte_len(),
            "{}",
            column.name
        );
    }
    let positions = frame.positions(PositionLayout::Padded4);
    assert_eq!(positions.bytes.len(), model.n_atoms() * 16);
}

#[test]
fn test_one_snapshot_serves_every_trajectory_frame() {
    let (file_path, _handle) = TestFile::multimodel_nmr_1d3z().create_temp().unwrap();
    let trajectory = load_trajectory(&file_path).unwrap();
    let topology = TopologyColumns::from_model(trajectory.representative()).unwrap();

    for i in 0..trajectory.frame_count() {
        let model = trajectory.frame(i);
        let frame = topology.frame(&model).unwrap();
        assert_eq!(frame.x(), model.x());
    }
}
