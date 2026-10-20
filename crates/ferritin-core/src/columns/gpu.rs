//! Framework-agnostic GPU upload descriptors.
//!
//! These types describe *what* to upload — element type, component count,
//! stride, row count, and the bytes — without depending on wgpu, bevy, or any
//! other GPU framework. A renderer maps a [`BufferLayout`] onto its own vertex
//! format or storage-buffer binding and copies [`GpuBuffer::bytes`] verbatim.

use super::buffer::{Column, ScalarType};
use super::error::{BoundaryError, to_u32};
use std::borrow::Cow;

/// Memory layout of one GPU buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferLayout {
    /// Scalar type of each component.
    pub scalar: ScalarType,
    /// Components per row (1 for scalar columns, 3 or 4 for positions, 2 for line indices).
    pub components: u32,
    /// Bytes between the start of consecutive rows.
    pub stride_bytes: u32,
    /// Number of rows.
    pub count: u32,
}

impl BufferLayout {
    /// Describe `count` rows of `components` scalars spaced `stride_bytes` apart.
    pub const fn new(scalar: ScalarType, components: u32, stride_bytes: u32, count: u32) -> Self {
        Self {
            scalar,
            components,
            stride_bytes,
            count,
        }
    }

    /// Total buffer size in bytes.
    pub const fn byte_len(&self) -> u64 {
        self.stride_bytes as u64 * self.count as u64
    }
}

/// How to interleave atom positions for upload.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PositionLayout {
    /// Tightly packed `[x, y, z]`, 12-byte stride. Suits vertex buffers
    /// (`Float32x3`).
    Packed3 = 1,
    /// `[x, y, z, 1.0]`, 16-byte stride. Required for storage/uniform buffers,
    /// where WGSL `array<vec3<f32>>` and std140/std430 `vec3` arrays use a
    /// 16-byte stride; uploading `Packed3` there silently misreads every atom
    /// after the first.
    Padded4 = 2,
}

impl PositionLayout {
    /// Components per position.
    pub const fn components(self) -> u32 {
        match self {
            PositionLayout::Packed3 => 3,
            PositionLayout::Padded4 => 4,
        }
    }

    /// Bytes per position.
    pub const fn stride_bytes(self) -> u32 {
        self.components() * 4
    }

    /// Layout of a buffer holding `count` positions.
    pub const fn buffer_layout(self, count: u32) -> BufferLayout {
        BufferLayout::new(
            ScalarType::F32,
            self.components(),
            self.stride_bytes(),
            count,
        )
    }
}

/// Bytes ready to upload, with their layout.
///
/// `bytes` is little-endian, matching every GPU and wasm32. It borrows when
/// the source column is already laid out correctly (see
/// [`GpuBuffer::from_column`]) and owns an interleaved copy otherwise.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuBuffer<'a> {
    /// Debug label, suitable for a GPU buffer label.
    pub label: &'static str,
    /// Memory layout of `bytes`.
    pub layout: BufferLayout,
    /// Buffer contents; `bytes.len() == layout.byte_len()`.
    pub bytes: Cow<'a, [u8]>,
}

impl<'a> GpuBuffer<'a> {
    /// Wrap a scalar column as a one-component buffer; zero-copy on
    /// little-endian targets.
    ///
    /// Columns obtained from [`TopologyColumns`](super::TopologyColumns) or
    /// [`FrameColumns`](super::FrameColumns) are already bounded, so this only
    /// fails for hand-built columns longer than `u32::MAX` rows.
    pub fn from_column(column: Column<'a>) -> Result<Self, BoundaryError> {
        let scalar = column.data.scalar_type();
        Ok(GpuBuffer {
            label: column.name,
            layout: BufferLayout::new(
                scalar,
                1,
                scalar.size_bytes() as u32,
                to_u32("column row count", column.data.len())?,
            ),
            bytes: column.data.le_bytes(),
        })
    }
}
