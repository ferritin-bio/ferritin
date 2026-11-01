#![cfg(target_arch = "wasm32")]

use ferritin_core::columns::{AtomHandle, FrameColumns, TopologyColumns};
use ferritin_wasm::{boundary_error_to_js, copy_frame, copy_topology, load_columns};
use js_sys::{Float32Array, Int8Array, Int32Array, Object, Reflect, Uint8Array, Uint32Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

const CIF: &str = "data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.auth_atom_id
_atom_site.label_comp_id
_atom_site.auth_comp_id
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_model_num
ATOM 1 N N NX ALA ALX label auth 7 1 42 B 1.5 2 3 0.5 12 1
ATOM 2 C CA CAX ALA ALX label auth 7 1 42 B 4.5 5 6 1 15 1
#
";

fn get(object: &JsValue, key: &str) -> JsValue {
    Reflect::get(object, &key.into()).unwrap()
}

fn decode(topology: &JsValue, column: &str, row: u32) -> String {
    let codes: Uint32Array = get(&get(topology, "columns"), column).unchecked_into();
    let dict = get(&get(topology, "dictionaries"), column);
    let offsets: Uint32Array = get(&dict, "value_offsets").unchecked_into();
    let bytes: Uint8Array = get(&dict, "value_bytes").unchecked_into();
    let code = codes.get_index(row);
    String::from_utf8(
        bytes
            .slice(offsets.get_index(code), offsets.get_index(code + 1))
            .to_vec(),
    )
    .unwrap()
}

#[wasm_bindgen_test]
fn test_public_export_types_metadata_and_identifiers() {
    let result = load_columns(CIF).unwrap();
    let topology = get(&result, "topology");
    let columns = get(&topology, "columns");
    assert_eq!(get(&topology, "n_atoms").as_f64(), Some(2.0));
    assert_eq!(get(&topology, "n_residues").as_f64(), Some(1.0));
    assert_eq!(get(&topology, "n_chains").as_f64(), Some(1.0));
    assert_eq!(
        Object::keys(&columns.clone().unchecked_into::<Object>()).length(),
        20
    );
    assert!(get(&columns, "atom.element").is_instance_of::<Uint8Array>());
    assert!(get(&columns, "atom.formal_charge").is_instance_of::<Int8Array>());
    assert!(get(&columns, "atom.name").is_instance_of::<Uint32Array>());
    assert!(get(&columns, "residue.auth_seq_id").is_instance_of::<Int32Array>());
    assert_eq!(
        Int8Array::new(&get(&columns, "atom.formal_charge")).get_index(0),
        -128
    );
    assert_eq!(
        Int32Array::new(&get(&columns, "residue.auth_seq_id")).get_index(0),
        42
    );
    assert_eq!(
        Uint32Array::new(&get(&columns, "residue.atom_offsets")).to_vec(),
        [0, 2]
    );
    for (column, expected) in [
        ("atom.name", "N"),
        ("atom.auth_name", "NX"),
        ("residue.comp_id", "ALA"),
        ("residue.auth_comp_id", "ALX"),
        ("chain.label_asym_id", "label"),
        ("chain.auth_asym_id", "auth"),
        ("chain.entity_id", "7"),
    ] {
        assert_eq!(decode(&topology, column, 0), expected);
    }
    let frame = get(&get(&result, "frame"), "columns");
    assert!(get(&frame, "x").is_instance_of::<Float32Array>());
    assert_eq!(Float32Array::new(&get(&frame, "x")).to_vec(), [1.5, 4.5]);
    assert_eq!(
        Float32Array::new(&get(&frame, "occupancy")).to_vec(),
        [0.5, 1.0]
    );
    assert_eq!(
        Float32Array::new(&get(&frame, "b_iso")).to_vec(),
        [12.0, 15.0]
    );
    assert!(get(&frame, "confidence").is_undefined());
}

#[wasm_bindgen_test]
fn test_copies_survive_mutation_drop_and_memory_growth() {
    let mut model = ferritin_core::load_model_from_string(CIF, "cif").unwrap();
    let topology = TopologyColumns::from_model(&model).unwrap();
    let first = copy_frame(topology.frame(&model).unwrap()).unwrap();
    let second = copy_frame(topology.frame(&model).unwrap()).unwrap();
    let x: Float32Array = get(&get(&first, "columns"), "x").unchecked_into();
    let other_x: Float32Array = get(&get(&second, "columns"), "x").unchecked_into();
    x.set_index(0, 99.0);
    assert_eq!(model.conformation.x[0], 1.5);
    assert_eq!(other_x.get_index(0), 1.5);
    model.conformation.x[0] = -7.0;
    assert_eq!(other_x.get_index(0), 1.5);
    let copied = copy_topology(&topology).unwrap();
    let names: Uint32Array = get(&get(&copied, "columns"), "atom.name").unchecked_into();
    names.set_index(0, 1);
    assert_eq!(topology.atom_name().codes()[0], 0);
    let dict = get(&get(&copied, "dictionaries"), "atom.name");
    let bytes: Uint8Array = get(&dict, "value_bytes").unchecked_into();
    bytes.set_index(0, b'Z');
    assert_eq!(topology.atom_name().value_bytes()[0], b'N');
    let offsets: Uint32Array = get(&dict, "value_offsets").unchecked_into();
    offsets.set_index(0, 1);
    assert_eq!(topology.atom_name().value_offsets()[0], 0);
    drop(topology);
    drop(model);
    let memory: js_sys::WebAssembly::Memory = wasm_bindgen::memory().unchecked_into();
    assert!(!Object::is(&x.buffer(), &memory.buffer()));
    memory.grow(16);
    assert_eq!(x.get_index(0), 99.0);
    assert_eq!(other_x.to_vec(), [1.5, 4.5]);
    assert_eq!(bytes.get_index(0), b'Z');
    assert_eq!(offsets.get_index(0), 1);
    assert_eq!(names.get_index(0), 1);
}

#[wasm_bindgen_test]
fn test_errors_are_js_errors_with_numeric_codes() {
    let model = ferritin_core::load_model_from_string(CIF, "cif").unwrap();
    let topology = TopologyColumns::from_model(&model).unwrap();
    let boundary = topology.residue_of(AtomHandle(2)).unwrap_err();
    let error: JsValue = boundary_error_to_js(boundary).into();
    assert!(error.is_instance_of::<js_sys::Error>());
    assert_eq!(get(&error, "code").as_f64(), Some(2.0));
    assert_eq!(
        get(&error, "message").as_string().unwrap(),
        "atom 2 is out of range (len=2)"
    );
    let mut invalid = model.clone();
    invalid.conformation.x.clear();
    let error: JsValue = boundary_error_to_js(FrameColumns::new(&invalid).unwrap_err()).into();
    assert_eq!(get(&error, "code").as_f64(), Some(1.0));
    let other = ferritin_core::load_model_from_string(CIF, "cif").unwrap();
    let error: JsValue = boundary_error_to_js(topology.frame(&other).unwrap_err()).into();
    assert_eq!(get(&error, "code").as_f64(), Some(3.0));
    let error: JsValue = load_columns("not a CIF").unwrap_err().into();
    assert!(error.is_instance_of::<js_sys::Error>());
    assert_eq!(get(&error, "code").as_f64(), Some(1.0));
}
