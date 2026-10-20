//! Zero-copy views of the per-frame columns of a [`Model`].

use super::buffer::{Column, ColumnData};
use super::error::{BoundaryError, to_u32};
use super::gpu::{GpuBuffer, PositionLayout};
use crate::model::Model;

/// Borrowed per-frame columns: coordinates and per-atom scalars.
///
/// Every accessor returns a slice of the model's own
/// [`AtomicConformation`](crate::AtomicConformation) storage; nothing is copied.
/// The borrow checker keeps the model alive and unmodified for `'a`. Pointers
/// handed across FFI from these slices carry the same rule, unenforced: they
/// are valid until the model is dropped or its conformation is mutated.
#[derive(Clone, Copy)]
pub struct FrameColumns<'a> {
    model: &'a Model,
    n_atoms: u32,
}

impl std::fmt::Debug for FrameColumns<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameColumns")
            .field("n_atoms", &self.n_atoms)
            .finish_non_exhaustive()
    }
}

impl<'a> FrameColumns<'a> {
    /// Borrow the per-frame columns of `model`, validating their lengths.
    ///
    /// Prefer [`TopologyColumns::frame`](super::TopologyColumns::frame), which
    /// also checks that the frame belongs to a given topology snapshot.
    pub fn new(model: &'a Model) -> Result<Self, BoundaryError> {
        let n_atoms = model.n_atoms();
        model.conformation.validate(n_atoms)?;
        Ok(Self {
            model,
            n_atoms: to_u32("atom count", n_atoms)?,
        })
    }

    /// Number of atoms.
    pub fn n_atoms(&self) -> u32 {
        self.n_atoms
    }

    /// X coordinates (Å).
    pub fn x(&self) -> &'a [f32] {
        &self.model.conformation.x
    }

    /// Y coordinates (Å).
    pub fn y(&self) -> &'a [f32] {
        &self.model.conformation.y
    }

    /// Z coordinates (Å).
    pub fn z(&self) -> &'a [f32] {
        &self.model.conformation.z
    }

    /// Occupancy, if recorded.
    pub fn occupancy(&self) -> Option<&'a [f32]> {
        self.model.conformation.occupancy.as_deref()
    }

    /// Isotropic B-factor, if recorded.
    pub fn b_iso(&self) -> Option<&'a [f32]> {
        self.model.conformation.b_iso.as_deref()
    }

    /// Per-atom confidence (e.g. pLDDT), if recorded.
    pub fn confidence(&self) -> Option<&'a [f32]> {
        self.model.conformation.confidence.as_deref()
    }

    /// Every present per-frame column by stable name. Absent optional columns
    /// are omitted.
    pub fn columns(&self) -> Vec<Column<'a>> {
        [
            ("x", Some(self.x())),
            ("y", Some(self.y())),
            ("z", Some(self.z())),
            ("occupancy", self.occupancy()),
            ("b_iso", self.b_iso()),
            ("confidence", self.confidence()),
        ]
        .into_iter()
        .filter_map(|(name, values)| {
            values.map(|values| Column {
                name,
                data: ColumnData::F32(values),
            })
        })
        .collect()
    }

    /// Interleave x/y/z into one position buffer. Always an owned copy, since
    /// the model stores coordinates as separate SoA columns.
    pub fn positions(&self, layout: PositionLayout) -> GpuBuffer<'static> {
        let components = layout.components();
        let mut bytes = Vec::with_capacity(self.n_atoms as usize * layout.stride_bytes() as usize);
        for ((&x, &y), &z) in self.x().iter().zip(self.y()).zip(self.z()) {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
            bytes.extend_from_slice(&z.to_le_bytes());
            if components == 4 {
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
            }
        }
        GpuBuffer {
            label: "positions",
            layout: layout.buffer_layout(self.n_atoms),
            bytes: bytes.into(),
        }
    }
}
