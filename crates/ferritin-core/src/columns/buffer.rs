//! Typed numeric column views and their little-endian byte representation.

use std::borrow::Cow;

/// Scalar element type of an exported column.
///
/// Discriminants are part of the boundary contract and will not be renumbered.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarType {
    U8 = 1,
    I8 = 2,
    U32 = 3,
    I32 = 4,
    F32 = 5,
}

impl ScalarType {
    /// Size of one scalar in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            ScalarType::U8 | ScalarType::I8 => 1,
            ScalarType::U32 | ScalarType::I32 | ScalarType::F32 => 4,
        }
    }

    /// Stable numeric code for this type.
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Plain-old-data scalars that may be reinterpreted as bytes.
///
/// Sealed: implemented only for padding-free primitive types.
pub(crate) trait Scalar: Copy + 'static {
    fn extend_le(self, out: &mut Vec<u8>);
}

macro_rules! impl_scalar {
    ($($t:ty),*) => {$(
        impl Scalar for $t {
            fn extend_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
impl_scalar!(u8, i8, u32, i32, f32);

/// Little-endian bytes of `values`: borrowed on little-endian targets (wasm32,
/// x86_64, aarch64), copied on big-endian ones.
pub(crate) fn le_bytes<T: Scalar>(values: &[T]) -> Cow<'_, [u8]> {
    if cfg!(target_endian = "little") {
        // SAFETY: `Scalar` is only implemented for primitive integer/float types,
        // which have no padding and no invalid byte patterns when read as `u8`.
        // `u8` has alignment 1 and the length is the exact byte size of the slice.
        Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        })
    } else {
        let mut out = Vec::with_capacity(std::mem::size_of_val(values));
        for &value in values {
            value.extend_le(&mut out);
        }
        Cow::Owned(out)
    }
}

/// A borrowed, contiguous, typed numeric column.
///
/// Every variant borrows storage owned elsewhere — a [`Model`](crate::Model)'s
/// conformation, its [`AtomicHierarchy`](crate::AtomicHierarchy), or a
/// [`TopologyColumns`](super::TopologyColumns) snapshot — so creating one never
/// copies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColumnData<'a> {
    U8(&'a [u8]),
    I8(&'a [i8]),
    U32(&'a [u32]),
    I32(&'a [i32]),
    F32(&'a [f32]),
}

impl<'a> ColumnData<'a> {
    /// Element type of the column.
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            ColumnData::U8(_) => ScalarType::U8,
            ColumnData::I8(_) => ScalarType::I8,
            ColumnData::U32(_) => ScalarType::U32,
            ColumnData::I32(_) => ScalarType::I32,
            ColumnData::F32(_) => ScalarType::F32,
        }
    }

    /// Number of scalars (rows).
    pub fn len(&self) -> usize {
        match self {
            ColumnData::U8(v) => v.len(),
            ColumnData::I8(v) => v.len(),
            ColumnData::U32(v) => v.len(),
            ColumnData::I32(v) => v.len(),
            ColumnData::F32(v) => v.len(),
        }
    }

    /// Returns `true` if the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of the column in bytes.
    pub fn byte_len(&self) -> usize {
        self.len() * self.scalar_type().size_bytes()
    }

    /// Pointer to the first scalar, for FFI hand-off together with [`Self::len`].
    ///
    /// The pointer is valid exactly as long as the borrow `'a`.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            ColumnData::U8(v) => v.as_ptr(),
            ColumnData::I8(v) => v.as_ptr().cast(),
            ColumnData::U32(v) => v.as_ptr().cast(),
            ColumnData::I32(v) => v.as_ptr().cast(),
            ColumnData::F32(v) => v.as_ptr().cast(),
        }
    }

    /// Little-endian bytes of the column; zero-copy on little-endian targets.
    pub fn le_bytes(&self) -> Cow<'a, [u8]> {
        match *self {
            ColumnData::U8(v) => Cow::Borrowed(v),
            ColumnData::I8(v) => le_bytes(v),
            ColumnData::U32(v) => le_bytes(v),
            ColumnData::I32(v) => le_bytes(v),
            ColumnData::F32(v) => le_bytes(v),
        }
    }
}

/// A named column, as listed by [`TopologyColumns::columns`](super::TopologyColumns::columns)
/// and [`FrameColumns::columns`](super::FrameColumns::columns).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Column<'a> {
    /// Stable column name, e.g. `"x"`, `"atom.element"`, `"bond.atom_a"`.
    pub name: &'static str,
    /// The column's values.
    pub data: ColumnData<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_le_bytes_matches_to_le_bytes() {
        let values = [1.5f32, -2.0, 3.25];
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&*ColumnData::F32(&values).le_bytes(), expected.as_slice());

        let ints = [-1i32, 7, i32::MIN];
        let expected: Vec<u8> = ints.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(&*ColumnData::I32(&ints).le_bytes(), expected.as_slice());
    }

    #[test]
    fn test_le_bytes_is_zero_copy_on_little_endian() {
        let values = [1u32, 2, 3];
        let bytes = ColumnData::U32(&values).le_bytes();
        if cfg!(target_endian = "little") {
            assert!(matches!(bytes, Cow::Borrowed(_)));
            assert_eq!(bytes.as_ptr(), values.as_ptr().cast());
        }
        assert_eq!(bytes.len(), 12);
    }

    #[test]
    fn test_column_data_metadata() {
        let values = [0i8, -1, 2, 3];
        let column = ColumnData::I8(&values);
        assert_eq!(column.scalar_type(), ScalarType::I8);
        assert_eq!(column.len(), 4);
        assert_eq!(column.byte_len(), 4);
        assert_eq!(column.as_ptr(), values.as_ptr().cast());
        assert!(ColumnData::F32(&[]).is_empty());
    }
}
