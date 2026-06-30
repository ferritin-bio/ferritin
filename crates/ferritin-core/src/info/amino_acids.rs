//! Amino acid conversion utilities.
//!
//! Single-letter ↔ three-letter ↔ integer mappings for the 20 standard amino
//! acids plus the catch-all `UNK` / `'X'` / index 20.

use strum::EnumIter;

/// One-letter code alphabet for the 20 standard amino acids plus UNK.
pub const ALPHABET: [char; 21] = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X',
];

/// The 20 standard amino acids plus Unknown, with integer discriminants
/// matching the conventional protein-ML encoding (Ala=0 … Unknown=20).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum AminoAcid {
    Ala = 0,  Cys = 1,  Asp = 2,  Glu = 3,  Phe = 4,
    Gly = 5,  His = 6,  Ile = 7,  Lys = 8,  Leu = 9,
    Met = 10, Asn = 11, Pro = 12, Gln = 13, Arg = 14,
    Ser = 15, Thr = 16, Val = 17, Trp = 18, Tyr = 19,
    Unknown = 20,
}

impl AminoAcid {
    /// Parse a three-letter code (e.g. `"ALA"`). Unknown/missing → `Unknown`.
    #[rustfmt::skip]
    pub fn from_three_letter(aa: &str) -> Self {
        match aa {
            "ALA" => Self::Ala, "CYS" => Self::Cys, "ASP" => Self::Asp,
            "GLU" => Self::Glu, "PHE" => Self::Phe, "GLY" => Self::Gly,
            "HIS" => Self::His, "ILE" => Self::Ile, "LYS" => Self::Lys,
            "LEU" => Self::Leu, "MET" => Self::Met, "ASN" => Self::Asn,
            "PRO" => Self::Pro, "GLN" => Self::Gln, "ARG" => Self::Arg,
            "SER" => Self::Ser, "THR" => Self::Thr, "VAL" => Self::Val,
            "TRP" => Self::Trp, "TYR" => Self::Tyr, _    => Self::Unknown,
        }
    }

    /// Parse a one-letter code (e.g. `'A'`). Unknown/missing → `Unknown`.
    #[rustfmt::skip]
    pub fn from_one_letter(aa: char) -> Self {
        match aa {
            'A' => Self::Ala, 'C' => Self::Cys, 'D' => Self::Asp,
            'E' => Self::Glu, 'F' => Self::Phe, 'G' => Self::Gly,
            'H' => Self::His, 'I' => Self::Ile, 'K' => Self::Lys,
            'L' => Self::Leu, 'M' => Self::Met, 'N' => Self::Asn,
            'P' => Self::Pro, 'Q' => Self::Gln, 'R' => Self::Arg,
            'S' => Self::Ser, 'T' => Self::Thr, 'V' => Self::Val,
            'W' => Self::Trp, 'Y' => Self::Tyr, _   => Self::Unknown,
        }
    }

    /// Parse from integer index (0–20). Out-of-range → `Unknown`.
    pub fn from_index(idx: u32) -> Self {
        match idx {
            0  => Self::Ala, 1  => Self::Cys, 2  => Self::Asp,
            3  => Self::Glu, 4  => Self::Phe, 5  => Self::Gly,
            6  => Self::His, 7  => Self::Ile, 8  => Self::Lys,
            9  => Self::Leu, 10 => Self::Met, 11 => Self::Asn,
            12 => Self::Pro, 13 => Self::Gln, 14 => Self::Arg,
            15 => Self::Ser, 16 => Self::Thr, 17 => Self::Val,
            18 => Self::Trp, 19 => Self::Tyr, _  => Self::Unknown,
        }
    }

    /// One-letter code (e.g. `'A'` for Ala, `'X'` for Unknown).
    #[rustfmt::skip]
    pub fn one_letter(self) -> char {
        match self {
            Self::Ala => 'A', Self::Cys => 'C', Self::Asp => 'D',
            Self::Glu => 'E', Self::Phe => 'F', Self::Gly => 'G',
            Self::His => 'H', Self::Ile => 'I', Self::Lys => 'K',
            Self::Leu => 'L', Self::Met => 'M', Self::Asn => 'N',
            Self::Pro => 'P', Self::Gln => 'Q', Self::Arg => 'R',
            Self::Ser => 'S', Self::Thr => 'T', Self::Val => 'V',
            Self::Trp => 'W', Self::Tyr => 'Y', Self::Unknown => 'X',
        }
    }

    /// Three-letter code (e.g. `"ALA"` for Ala, `"UNK"` for Unknown).
    #[rustfmt::skip]
    pub fn three_letter(self) -> &'static str {
        match self {
            Self::Ala => "ALA", Self::Cys => "CYS", Self::Asp => "ASP",
            Self::Glu => "GLU", Self::Phe => "PHE", Self::Gly => "GLY",
            Self::His => "HIS", Self::Ile => "ILE", Self::Lys => "LYS",
            Self::Leu => "LEU", Self::Met => "MET", Self::Asn => "ASN",
            Self::Pro => "PRO", Self::Gln => "GLN", Self::Arg => "ARG",
            Self::Ser => "SER", Self::Thr => "THR", Self::Val => "VAL",
            Self::Trp => "TRP", Self::Tyr => "TYR", Self::Unknown => "UNK",
        }
    }

    /// Integer index (discriminant), 0–20.
    pub fn index(self) -> usize {
        self as usize
    }
}

// ── Backward-compatible free functions ───────────────────────────────────────

/// Convert a 3-letter amino acid code to a 1-letter code.
pub fn aa3to1(aa: &str) -> char {
    AminoAcid::from_three_letter(aa).one_letter()
}

/// Convert a 1-letter amino acid code to an integer index (0–20, UNK→20).
pub fn aa1to_int(aa: char) -> u32 {
    AminoAcid::from_one_letter(aa).index() as u32
}

/// Convert an integer index (0–20) to a 1-letter amino acid code.
pub fn int_to_aa1(aa_int: u32) -> char {
    AminoAcid::from_index(aa_int).one_letter()
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

    #[test]
    fn test_amino_acid_enum_roundtrip() {
        use strum::IntoEnumIterator;
        for aa in AminoAcid::iter() {
            let idx = aa.index() as u32;
            assert_eq!(AminoAcid::from_index(idx), aa);
            assert_eq!(AminoAcid::from_one_letter(aa.one_letter()), aa);
            if aa != AminoAcid::Unknown {
                assert_eq!(AminoAcid::from_three_letter(aa.three_letter()), aa);
            }
        }
    }
}
