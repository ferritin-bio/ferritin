//! Dictionary encoding for string columns.

use super::error::{BoundaryError, to_u32};
use std::collections::HashMap;

/// A dictionary-encoded string column.
///
/// Each row holds a `u32` code into a table of unique values. The value table
/// is stored Arrow-style as one UTF-8 byte buffer plus `n_values() + 1`
/// offsets, so both halves cross an FFI or wasm-bindgen boundary as plain
/// numeric buffers (`Uint32Array` codes, `Uint8Array` bytes, `Uint32Array`
/// offsets) and the consumer decodes each unique value once.
///
/// Codes are assigned in first-occurrence order, so encoding the same input
/// always yields the same codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StringDictionary {
    codes: Vec<u32>,
    value_bytes: Vec<u8>,
    value_offsets: Vec<u32>,
}

impl StringDictionary {
    /// Encode a sequence of strings.
    ///
    /// Fails with [`CapacityExceeded`](super::BoundaryErrorKind::CapacityExceeded)
    /// if the row count or total value bytes do not fit in `u32`.
    pub fn encode<'s>(values: impl IntoIterator<Item = &'s str>) -> Result<Self, BoundaryError> {
        let mut lookup: HashMap<&'s str, u32> = HashMap::new();
        let mut codes = Vec::new();
        let mut value_bytes = Vec::new();
        let mut value_offsets = vec![0u32];

        for value in values {
            let code = match lookup.get(value) {
                Some(&code) => code,
                None => {
                    let code = to_u32("dictionary value count", lookup.len())?;
                    value_bytes.extend_from_slice(value.as_bytes());
                    value_offsets.push(to_u32("dictionary value bytes", value_bytes.len())?);
                    lookup.insert(value, code);
                    code
                }
            };
            codes.push(code);
        }
        to_u32("dictionary row count", codes.len())?;

        Ok(Self {
            codes,
            value_bytes,
            value_offsets,
        })
    }

    /// Per-row codes into the value table.
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// Concatenated UTF-8 bytes of all unique values.
    pub fn value_bytes(&self) -> &[u8] {
        &self.value_bytes
    }

    /// Offsets into [`Self::value_bytes`]; value `c` spans
    /// `value_offsets[c]..value_offsets[c + 1]`.
    pub fn value_offsets(&self) -> &[u32] {
        &self.value_offsets
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns `true` if there are no rows.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Number of unique values.
    pub fn n_values(&self) -> usize {
        self.value_offsets.len() - 1
    }

    /// The unique value for `code`, or `None` if the code is out of range.
    pub fn value(&self, code: u32) -> Option<&str> {
        let code = code as usize;
        let start = *self.value_offsets.get(code)? as usize;
        let end = *self.value_offsets.get(code + 1)? as usize;
        // Bytes were copied from `&str` values and split on their boundaries.
        std::str::from_utf8(&self.value_bytes[start..end]).ok()
    }

    /// The decoded value of row `row`, or `None` if the row is out of range.
    pub fn get(&self, row: usize) -> Option<&str> {
        self.value(*self.codes.get(row)?)
    }

    /// Iterate over the unique values in code order.
    pub fn values(&self) -> impl Iterator<Item = &str> + '_ {
        (0..self.n_values() as u32).filter_map(|code| self.value(code))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_first_occurrence_codes() {
        let dict = StringDictionary::encode(["CA", "N", "CA", "C", "N"]).unwrap();
        assert_eq!(dict.codes(), &[0, 1, 0, 2, 1]);
        assert_eq!(dict.n_values(), 3);
        assert_eq!(dict.value_bytes(), b"CANC");
        assert_eq!(dict.value_offsets(), &[0, 2, 3, 4]);
        assert_eq!(dict.values().collect::<Vec<_>>(), vec!["CA", "N", "C"]);
    }

    #[test]
    fn test_dictionary_roundtrip_and_bounds() {
        let input = ["ALA", "", "HOH", "ALA", "Ω"];
        let dict = StringDictionary::encode(input).unwrap();
        let decoded: Vec<&str> = (0..dict.len()).map(|i| dict.get(i).unwrap()).collect();
        assert_eq!(decoded, input);
        assert_eq!(dict.get(input.len()), None);
        assert_eq!(dict.value(dict.n_values() as u32), None);
    }

    #[test]
    fn test_dictionary_empty() {
        let dict = StringDictionary::encode(std::iter::empty()).unwrap();
        assert!(dict.is_empty());
        assert_eq!(dict.n_values(), 0);
        assert_eq!(dict.value_offsets(), &[0]);
    }
}
