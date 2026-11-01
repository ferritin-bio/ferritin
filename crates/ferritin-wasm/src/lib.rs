//! Owned JavaScript snapshots of the core columns contract; see the crate README.

use ferritin_core::columns::{BoundaryError, Column, ColumnData, FrameColumns, TopologyColumns};
use js_sys::{Float32Array, Int8Array, Int32Array, Object, Reflect, Uint8Array, Uint32Array};
// js_sys exposes the JavaScript Error as an inspectable object, allowing `code`.
use js_sys::Error as JsError;
use wasm_bindgen::prelude::*;

fn set(object: &Object, key: &str, value: impl Into<JsValue>) -> Result<(), JsError> {
    Reflect::set(object, &key.into(), &value.into())
        .map(|_| ())
        .map_err(|_| JsError::new("Unable to set snapshot property"))
}

fn coded_error(message: &str, code: u32) -> JsError {
    let error = JsError::new(message);
    // A newly created Error is extensible and has no pre-existing `code` property.
    Reflect::set(error.as_ref(), &"code".into(), &code.into())
        .expect("new JavaScript Error must accept code");
    error
}

/// Convert a core boundary error to a JavaScript Error with numeric `code`.
pub fn boundary_error_to_js(error: BoundaryError) -> JsError {
    coded_error(error.message(), error.code())
}

fn copy_columns(columns: Vec<Column<'_>>) -> Result<Object, JsError> {
    let output = Object::new();
    for column in columns {
        // `from(slice)` copies into a JS-owned ArrayBuffer, never a wasm view.
        let array: JsValue = match column.data {
            ColumnData::U8(v) => Uint8Array::from(v).into(),
            ColumnData::I8(v) => Int8Array::from(v).into(),
            ColumnData::U32(v) => Uint32Array::from(v).into(),
            ColumnData::I32(v) => Int32Array::from(v).into(),
            ColumnData::F32(v) => Float32Array::from(v).into(),
        };
        set(&output, column.name, array)?;
    }
    Ok(output)
}

/// Copy all present frame columns to independently owned JS typed arrays.
pub fn copy_frame(frame: FrameColumns<'_>) -> Result<Object, JsError> {
    let output = Object::new();
    set(&output, "n_atoms", frame.n_atoms())?;
    set(&output, "columns", copy_columns(frame.columns())?)?;
    Ok(output)
}

/// Copy topology scalars, counts, and every string dictionary to JavaScript.
pub fn copy_topology(topology: &TopologyColumns) -> Result<Object, JsError> {
    let output = Object::new();
    set(&output, "n_atoms", topology.n_atoms())?;
    set(&output, "n_residues", topology.n_residues())?;
    set(&output, "n_chains", topology.n_chains())?;
    set(&output, "n_bonds", topology.n_bonds())?;
    set(&output, "columns", copy_columns(topology.columns())?)?;
    let dictionaries = Object::new();
    for (name, dictionary) in [
        ("atom.name", topology.atom_name()),
        ("atom.auth_name", topology.auth_atom_name()),
        ("residue.comp_id", topology.comp_id()),
        ("residue.auth_comp_id", topology.auth_comp_id()),
        ("chain.label_asym_id", topology.label_asym_id()),
        ("chain.auth_asym_id", topology.auth_asym_id()),
        ("chain.entity_id", topology.entity_id()),
    ] {
        let values = Object::new();
        set(
            &values,
            "value_bytes",
            Uint8Array::from(dictionary.value_bytes()),
        )?;
        set(
            &values,
            "value_offsets",
            Uint32Array::from(dictionary.value_offsets()),
        )?;
        set(&dictionaries, name, values)?;
    }
    set(&output, "dictionaries", dictionaries)?;
    Ok(output)
}

/// Parse the first mmCIF model and return `{ topology, frame }` owned snapshots.
/// Parse failures have code 1; core boundary errors retain their numeric codes.
#[wasm_bindgen(js_name = loadColumns)]
pub fn load_columns(cif: &str) -> Result<Object, JsError> {
    let model = ferritin_core::load_model_from_string(cif, "cif")
        .map_err(|error| coded_error(&format!("{error:#}"), 1))?;
    let topology = TopologyColumns::from_model(&model).map_err(boundary_error_to_js)?;
    let frame = topology.frame(&model).map_err(boundary_error_to_js)?;
    let output = Object::new();
    set(&output, "topology", copy_topology(&topology)?)?;
    set(&output, "frame", copy_frame(frame)?)?;
    Ok(output)
}
