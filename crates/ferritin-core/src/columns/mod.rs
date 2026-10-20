//! Stable columnar boundary for WASM, FFI, and GPU consumers.
//!
//! The [`Model`](crate::Model) layer is already struct-of-arrays, but its
//! public fields are Rust types (`Vec<String>`, `Option<char>`, `Element`,
//! `Arc`, borrowed views) that cannot cross a wasm-bindgen or C ABI boundary,
//! or be uploaded to a GPU, as they are. This module is the contract for doing
//! so. It adds no dependencies: bindings crates (wasm-bindgen, PyO3, a C
//! header) and renderers (bevy, wgpu) build on it.
//!
//! # What crosses the boundary
//!
//! * **Numeric columns** — contiguous `u8`/`i8`/`u32`/`i32`/`f32` slices
//!   ([`ColumnData`]), each with a stable name ([`Column`]). Bytes are
//!   little-endian ([`ColumnData::le_bytes`]).
//! * **String columns** — dictionary-encoded ([`StringDictionary`]): one `u32`
//!   code per row plus a UTF-8 value buffer and offsets.
//! * **Optional values** — encoded in place with sentinels
//!   ([`NO_ALT_LOC`], [`NO_INS_CODE`], [`NO_FORMAL_CHARGE`],
//!   [`MISSING_SEQ_ID`](crate::model::MISSING_SEQ_ID)), never as a separate
//!   validity bitmap.
//! * **Handles** — [`AtomHandle`], [`ResidueHandle`], [`ChainHandle`]: `u32`
//!   row indices into a topology snapshot, navigated with checked lookups.
//! * **Errors** — [`BoundaryError`], with a stable numeric
//!   [`code`](BoundaryError::code) instead of panics.
//! * **GPU uploads** — [`GpuBuffer`]: bytes plus a [`BufferLayout`]
//!   (scalar type, components, stride, count).
//!
//! # Ownership, borrowing, and invalidation
//!
//! | Data | Source | Copy or borrow | Valid until |
//! |---|---|---|---|
//! | `x`, `y`, `z`, `occupancy`, `b_iso`, `confidence` | [`FrameColumns`] | borrow of the model's conformation | the model is dropped or its conformation mutated |
//! | sequence ids, segment offsets, bonds | [`TopologyColumns`] | borrow of the pinned hierarchy | the snapshot is dropped |
//! | element, alt_loc, charge, ins_code, group, dictionaries | [`TopologyColumns`] | owned, encoded once per hierarchy | the snapshot is dropped |
//! | interleaved positions, bond line indices | [`GpuBuffer`] | owned copy | never (independent of the model) |
//!
//! Inside Rust the borrow checker enforces the "valid until" column. Across
//! FFI it is the caller's obligation: a raw pointer from
//! [`ColumnData::as_ptr`] is valid exactly as long as the Rust borrow it came
//! from.
//!
//! A [`TopologyColumns`] snapshot keeps its `Arc<AtomicHierarchy>` alive and
//! therefore immutable (`Arc::make_mut` will clone rather than mutate). A
//! topology edit thus produces a *new* hierarchy, and pairing a frame with the
//! old snapshot fails with [`StaleTopology`](BoundaryErrorKind::StaleTopology)
//! rather than exporting mismatched columns. All frames of a trajectory share
//! one hierarchy, so one snapshot serves every frame.
//!
//! ## wasm-bindgen
//!
//! A zero-copy JS typed-array view over a borrowed column (e.g.
//! `js_sys::Float32Array::view`) points into wasm linear memory and is
//! **detached by any allocation that grows that memory**, including
//! allocations made later by ferritin itself. Use views only for an immediate
//! copy or upload within the same call; copy (`Float32Array::from`) anything
//! JS keeps.
//!
//! # Example
//!
//! ```text
//! let topology = TopologyColumns::from_model(&model)?;   // once per hierarchy
//! let frame = topology.frame(&model)?;                   // per frame, zero-copy
//! upload(frame.positions(PositionLayout::Padded4));      // owned, 16-byte stride
//! upload(GpuBuffer::from_column(column)?);               // borrowed scalar column
//! upload(topology.bond_line_indices());                  // owned [a, b] pairs
//! ```

mod buffer;
mod dictionary;
mod error;
mod frame;
mod gpu;
mod handles;
mod topology;

pub use buffer::{Column, ColumnData, ScalarType};
pub use dictionary::StringDictionary;
pub use error::{BoundaryError, BoundaryErrorKind};
pub use frame::FrameColumns;
pub use gpu::{BufferLayout, GpuBuffer, PositionLayout};
pub use handles::{AtomHandle, ChainHandle, ResidueHandle};
pub use topology::{
    NO_ALT_LOC, NO_FORMAL_CHARGE, NO_INS_CODE, RESIDUE_GROUP_NON_POLYMER, RESIDUE_GROUP_POLYMER,
    TopologyColumns,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Segmentation;
    use crate::info::elements::Element;
    use crate::model::{
        AtomicConformation, AtomicHierarchy, AtomsTable, Bonds, ChainsTable, MISSING_SEQ_ID, Model,
        ResidueGroup, ResiduesTable,
    };
    use std::borrow::Cow;
    use std::sync::Arc;

    /// Two chains: chain A has ALA (N, CA, C) + GLY (N, CA); chain B has one HOH (O).
    fn make_model() -> Model {
        let atoms = AtomsTable {
            atom_name: ["N", "CA", "C", "N", "CA", "O"].map(String::from).to_vec(),
            auth_atom_name: ["N", "CA", "C", "N", "CA", "O"].map(String::from).to_vec(),
            element: vec![
                Element::N,
                Element::C,
                Element::C,
                Element::N,
                Element::C,
                Element::O,
            ],
            alt_loc: vec![None, Some('A'), None, None, None, None],
            formal_charge: vec![None, None, None, Some(1), None, Some(-2)],
        };
        let residues = ResiduesTable {
            comp_id: ["ALA", "GLY", "HOH"].map(String::from).to_vec(),
            auth_comp_id: ["ALA", "GLY", "HOH"].map(String::from).to_vec(),
            label_seq_id: vec![1, 2, MISSING_SEQ_ID],
            auth_seq_id: vec![10, 11, 101],
            ins_code: vec![None, Some('B'), None],
            group: vec![
                ResidueGroup::Polymer,
                ResidueGroup::Polymer,
                ResidueGroup::NonPolymer,
            ],
        };
        let chains = ChainsTable {
            label_asym_id: ["A", "B"].map(String::from).to_vec(),
            auth_asym_id: ["A", "B"].map(String::from).to_vec(),
            entity_id: ["1", "2"].map(String::from).to_vec(),
        };
        let hierarchy = AtomicHierarchy {
            atoms,
            residues,
            chains,
            atom_to_residue: Segmentation::from_offsets(vec![0, 3, 5, 6]),
            residue_to_chain: Segmentation::from_offsets(vec![0, 2, 3]),
            bonds: Bonds::from_unsorted(vec![1, 0, 2], vec![2, 1, 3], vec![1, 1, 1], 6),
        };
        let conformation = AtomicConformation {
            x: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            y: vec![0.5; 6],
            z: vec![-1.0; 6],
            occupancy: Some(vec![1.0; 6]),
            b_iso: None,
            confidence: None,
        };
        Model::new(Arc::new(hierarchy), conformation)
    }

    #[test]
    fn test_topology_encodes_scalars_and_sentinels() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();

        assert_eq!(topology.n_atoms(), 6);
        assert_eq!(topology.n_residues(), 3);
        assert_eq!(topology.n_chains(), 2);
        assert_eq!(topology.n_bonds(), 3);
        assert_eq!(topology.element(), &[7, 6, 6, 7, 6, 8]);
        assert_eq!(topology.alt_loc(), &[NO_ALT_LOC, 'A' as u32, 0, 0, 0, 0]);
        assert_eq!(
            topology.formal_charge(),
            &[
                NO_FORMAL_CHARGE,
                NO_FORMAL_CHARGE,
                NO_FORMAL_CHARGE,
                1,
                NO_FORMAL_CHARGE,
                -2
            ]
        );
        assert_eq!(topology.ins_code(), &[NO_INS_CODE, 'B' as u32, NO_INS_CODE]);
        assert_eq!(
            topology.residue_group(),
            &[
                RESIDUE_GROUP_POLYMER,
                RESIDUE_GROUP_POLYMER,
                RESIDUE_GROUP_NON_POLYMER
            ]
        );
        assert_eq!(topology.label_seq_id(), &[1, 2, MISSING_SEQ_ID]);
        assert_eq!(topology.residue_atom_offsets(), &[0, 3, 5, 6]);
        assert_eq!(topology.chain_residue_offsets(), &[0, 2, 3]);
    }

    #[test]
    fn test_topology_dictionaries_roundtrip() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        let atoms = &model.hierarchy.atoms;

        assert_eq!(topology.atom_name().n_values(), 4); // N, CA, C, O
        for (i, name) in atoms.atom_name.iter().enumerate() {
            assert_eq!(topology.atom_name().get(i), Some(name.as_str()));
        }
        assert_eq!(topology.comp_id().codes(), &[0, 1, 2]);
        assert_eq!(topology.entity_id().get(1), Some("2"));
    }

    #[test]
    fn test_topology_borrows_hierarchy_columns() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        assert!(std::ptr::eq(
            topology.auth_seq_id(),
            model.hierarchy.residues.auth_seq_id.as_slice()
        ));
        assert!(std::ptr::eq(
            topology.bond_atom_b(),
            model.hierarchy.bonds.atom_b.as_slice()
        ));
    }

    #[test]
    fn test_topology_columns_lengths_and_names_are_unique() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        let columns = topology.columns();

        let mut names: Vec<&str> = columns.iter().map(|c| c.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), columns.len());

        for column in &columns {
            // CSR offsets carry one sentinel entry beyond their parent table.
            let expected = match column.name {
                "residue.atom_offsets" => topology.n_residues() + 1,
                "chain.residue_offsets" => topology.n_chains() + 1,
                "atom.bond_offsets" => topology.n_atoms() + 1,
                name => match name.split('.').next().unwrap() {
                    "atom" => topology.n_atoms(),
                    "residue" => topology.n_residues(),
                    "chain" => topology.n_chains(),
                    "bond" => topology.n_bonds(),
                    other => panic!("unexpected column prefix {other}"),
                },
            };
            assert_eq!(column.data.len() as u32, expected, "{}", column.name);
        }
    }

    #[test]
    fn test_handle_navigation_is_checked() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();

        assert_eq!(
            topology.residue_of(AtomHandle(4)).unwrap(),
            ResidueHandle(1)
        );
        assert_eq!(topology.chain_of(ResidueHandle(2)).unwrap(), ChainHandle(1));
        assert_eq!(topology.atoms_of(ResidueHandle(1)).unwrap(), 3..5);
        assert_eq!(topology.residues_of(ChainHandle(0)).unwrap(), 0..2);

        let err = topology.residue_of(AtomHandle(6)).unwrap_err();
        assert_eq!(err.kind(), BoundaryErrorKind::IndexOutOfRange);
        assert_eq!(err.code(), 2);
        assert_eq!(err.message(), "atom 6 is out of range (len=6)");
        assert!(topology.residues_of(ChainHandle(2)).is_err());
    }

    #[test]
    fn test_frame_columns_are_zero_copy() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        let frame = topology.frame(&model).unwrap();

        assert!(std::ptr::eq(frame.x(), model.conformation.x.as_slice()));
        assert_eq!(frame.occupancy().map(<[f32]>::len), Some(6));
        assert!(frame.b_iso().is_none());

        let names: Vec<&str> = frame.columns().iter().map(|c| c.name).collect();
        assert_eq!(names, vec!["x", "y", "z", "occupancy"]);

        let buffer = GpuBuffer::from_column(frame.columns()[0]).unwrap();
        assert_eq!(buffer.layout, BufferLayout::new(ScalarType::F32, 1, 4, 6));
        if cfg!(target_endian = "little") {
            assert!(matches!(buffer.bytes, Cow::Borrowed(_)));
        }
    }

    #[test]
    fn test_frames_sharing_a_hierarchy_reuse_one_snapshot() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        let mut conformation = model.conformation.clone();
        conformation.x[0] = 42.0;
        let other_frame = Model::new(Arc::clone(&model.hierarchy), conformation);

        let frame = topology.frame(&other_frame).unwrap();
        assert_eq!(frame.x()[0], 42.0);
    }

    #[test]
    fn test_topology_edit_makes_snapshot_stale() {
        let mut model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        assert!(topology.is_current_for(&model));

        // The snapshot pins the hierarchy, so a topology edit must copy-on-write.
        Arc::make_mut(&mut model.hierarchy).atoms.atom_name[0] = "N1".to_string();
        assert!(!topology.is_current_for(&model));
        assert_eq!(topology.atom_name().get(0), Some("N"));

        let err = topology.frame(&model).unwrap_err();
        assert_eq!(err.kind(), BoundaryErrorKind::StaleTopology);
        assert_eq!(err.code(), 3);
    }

    #[test]
    fn test_reserved_sentinel_values_are_rejected() {
        let model = make_model();
        let mut hierarchy = (*model.hierarchy).clone();
        hierarchy.atoms.formal_charge[0] = Some(NO_FORMAL_CHARGE);
        let err = TopologyColumns::encode(&Arc::new(hierarchy)).unwrap_err();
        assert_eq!(err.kind(), BoundaryErrorKind::InvalidModel);

        let mut hierarchy = (*model.hierarchy).clone();
        hierarchy.residues.ins_code[0] = Some('\0');
        let err = TopologyColumns::encode(&Arc::new(hierarchy)).unwrap_err();
        assert_eq!(err.kind(), BoundaryErrorKind::InvalidModel);
    }

    #[test]
    fn test_invalid_hierarchy_maps_to_invalid_model() {
        let model = make_model();
        let mut hierarchy = (*model.hierarchy).clone();
        hierarchy.atoms.element.pop();
        let err = TopologyColumns::encode(&Arc::new(hierarchy)).unwrap_err();
        assert_eq!(err.kind(), BoundaryErrorKind::InvalidModel);
        assert_eq!(err.code(), 1);
        assert_eq!(err.message(), "atoms.element has length 5, expected 6");
    }

    #[test]
    fn test_positions_layouts() {
        let model = make_model();
        let frame = FrameColumns::new(&model).unwrap();

        let packed = frame.positions(PositionLayout::Packed3);
        assert_eq!(packed.layout.stride_bytes, 12);
        assert_eq!(packed.bytes.len() as u64, packed.layout.byte_len());
        assert_eq!(&packed.bytes[12..16], &1.0f32.to_le_bytes()); // atom 1 x

        let padded = frame.positions(PositionLayout::Padded4);
        assert_eq!(padded.layout, BufferLayout::new(ScalarType::F32, 4, 16, 6));
        assert_eq!(padded.bytes.len(), 6 * 16);
        assert_eq!(&padded.bytes[12..16], &1.0f32.to_le_bytes()); // atom 0 w
        assert_eq!(&padded.bytes[16..20], &1.0f32.to_le_bytes()); // atom 1 x
        assert_eq!(&padded.bytes[20..24], &0.5f32.to_le_bytes()); // atom 1 y
    }

    #[test]
    fn test_bond_line_indices() {
        let model = make_model();
        let topology = TopologyColumns::from_model(&model).unwrap();
        let lines = topology.bond_line_indices();

        assert_eq!(lines.layout, BufferLayout::new(ScalarType::U32, 2, 8, 3));
        let pairs: Vec<u32> = lines
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| u32::from_le_bytes(b))
            .collect();
        // Bonds are sorted by atom_a.
        assert_eq!(pairs, vec![0, 1, 1, 2, 2, 3]);
    }
}
