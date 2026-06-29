//! Amino acid conversion utilities.
//!
//! Single-letter ↔ three-letter ↔ integer mappings for the 20 standard amino
//! acids plus the catch-all `UNK` / `'X'` / index 20.

/// One-letter code alphabet for the 20 standard amino acids plus UNK.
pub const ALPHABET: [char; 21] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X',
];

/// Convert a 3-letter amino acid code to a 1-letter code.
#[rustfmt::skip]
pub fn aa3to1(aa: &str) -> char {
    match aa {
        "ALA" => 'A', "CYS" => 'C', "ASP" => 'D',
        "GLU" => 'E', "PHE" => 'F', "GLY" => 'G',
        "HIS" => 'H', "ILE" => 'I', "LYS" => 'K',
        "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
        "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R',
        "SER" => 'S', "THR" => 'T', "VAL" => 'V',
        "TRP" => 'W', "TYR" => 'Y', _     => 'X',
    }
}

/// Convert a 1-letter amino acid code to an integer index (0–20, UNK→20).
#[rustfmt::skip]
pub fn aa1to_int(aa: char) -> u32 {
    match aa {
        'A' => 0,  'C' => 1,  'D' => 2,
        'E' => 3,  'F' => 4,  'G' => 5,
        'H' => 6,  'I' => 7,  'K' => 8,
        'L' => 9,  'M' => 10, 'N' => 11,
        'P' => 12, 'Q' => 13, 'R' => 14,
        'S' => 15, 'T' => 16, 'V' => 17,
        'W' => 18, 'Y' => 19, _   => 20,
    }
}

/// Convert an integer index (0–20) to a 1-letter amino acid code.
#[rustfmt::skip]
pub fn int_to_aa1(aa_int: u32) -> char {
    match aa_int {
        0  => 'A', 1  => 'C', 2  => 'D',
        3  => 'E', 4  => 'F', 5  => 'G',
        6  => 'H', 7  => 'I', 8  => 'K',
        9  => 'L', 10 => 'M', 11 => 'N',
        12 => 'P', 13 => 'Q', 14 => 'R',
        15 => 'S', 16 => 'T', 17 => 'V',
        18 => 'W', 19 => 'Y', 20 => 'X',
        _  => 'X',
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aa3to1_roundtrip() {
        assert_eq!(aa3to1("ALA"), 'A');
        assert_eq!(aa3to1("TRP"), 'W');
        assert_eq!(aa3to1("UNK"), 'X');
        assert_eq!(aa3to1("???"), 'X');
    }

    #[test]
    fn test_aa1to_int_roundtrip() {
        for (i, &ch) in ALPHABET.iter().enumerate() {
            assert_eq!(aa1to_int(ch) as usize, i, "ALPHABET[{i}] = {ch}");
        }
        assert_eq!(aa1to_int('?'), 20);
    }

    #[test]
    fn test_int_to_aa1_roundtrip() {
        for i in 0u32..=20 {
            let ch = int_to_aa1(i);
            assert_eq!(aa1to_int(ch), i, "round-trip failed at index {i}");
        }
        assert_eq!(int_to_aa1(99), 'X');
    }
}
