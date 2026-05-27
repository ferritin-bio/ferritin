//! Input builder types for ESMFold2 structure prediction.
//!
//! This module provides typed builder structs for constructing the
//! [`StructurePredictionInput`] required by the ESMFold2 API. It mirrors
//! the Python `esm.models.esmfold2` SDK, supporting protein chains,
//! DNA chains with optional chemical modifications, and small-molecule
//! ligands specified by CCD (Chemical Component Dictionary) codes.
//!
//! # Example
//! ```rust
//! use ferritin_plms::esmfold2::input_types::{
//!     DNAInput, LigandInput, ProteinInput, StructurePredictionInput,
//! };
//!
//! let protein = ProteinInput::new("A", "MKTAYIAK").unwrap();
//! let ligand  = LigandInput::from_ccd("L", "ATP");
//! let input   = StructurePredictionInput::new()
//!     .add_protein(protein)
//!     .add_ligand(ligand);
//!
//! assert_eq!(input.num_chains(), 2);
//! ```

use std::fmt;

/// Standard 20 amino acids plus common ambiguous / non-standard codes.
///
/// Uppercase and lowercase are both accepted.  Includes:
/// - Standard: `ACDEFGHIKLMNPQRSTVWY`
/// - Ambiguous: `BJOUXZ`  (B=Asp/Asn, J=Leu/Ile, O=Pyrrolysine, U=Selenocysteine,
///   X=unknown, Z=Glu/Gln)
/// - Gap `-` and stop `*`
const PROTEIN_ALPHABET: &str = "ACDEFGHIKLMNPQRSTVWYBJOUXZacdefghiklmnpqrstvwybjouxz-*";

/// IUPAC DNA nucleotide alphabet including ambiguous base codes.
const DNA_ALPHABET: &str = "ACGTNRYSWKMBDHVacgtnryswkmbdhv";

// ── ProteinInput ─────────────────────────────────────────────────────────────

/// A single protein chain for structure prediction.
#[derive(Debug, Clone)]
pub struct ProteinInput {
    /// Chain identifier (e.g. `"A"`).
    pub id: String,
    /// Amino acid sequence in single-letter code.
    pub sequence: String,
}

impl ProteinInput {
    /// Construct a new `ProteinInput`, validating the sequence alphabet.
    ///
    /// Returns `Err` if the sequence is empty or contains an unrecognised character.
    pub fn new(id: impl Into<String>, sequence: impl Into<String>) -> Result<Self, String> {
        let id = id.into();
        let sequence = sequence.into();
        validate_protein_sequence(&sequence)?;
        Ok(Self { id, sequence })
    }

    /// Length of the amino acid sequence.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Returns `true` if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

impl fmt::Display for ProteinInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Protein[{}](len={})", self.id, self.sequence.len())
    }
}

// ── Modification ─────────────────────────────────────────────────────────────

/// A chemical modification at a specific position within a nucleic-acid chain.
#[derive(Debug, Clone)]
pub struct Modification {
    /// 1-based position in the sequence.
    pub position: usize,
    /// CCD code identifying the modification (e.g. `"5MC"` for 5-methylcytosine).
    pub ccd: String,
}

impl Modification {
    /// Construct a new `Modification`.
    pub fn new(position: usize, ccd: impl Into<String>) -> Self {
        Self {
            position,
            ccd: ccd.into(),
        }
    }
}

impl fmt::Display for Modification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mod[{}@pos{}]", self.ccd, self.position)
    }
}

// ── DNAInput ─────────────────────────────────────────────────────────────────

/// A single DNA chain, optionally carrying chemical modifications.
#[derive(Debug, Clone)]
pub struct DNAInput {
    /// Chain identifier.
    pub id: String,
    /// DNA nucleotide sequence (IUPAC single-letter codes).
    pub sequence: String,
    /// Chemical modifications along the chain (may be empty).
    pub modifications: Vec<Modification>,
}

impl DNAInput {
    /// Construct a new `DNAInput` without modifications.
    ///
    /// Returns `Err` if the sequence is empty or contains an invalid nucleotide.
    pub fn new(id: impl Into<String>, sequence: impl Into<String>) -> Result<Self, String> {
        let id = id.into();
        let sequence = sequence.into();
        validate_dna_sequence(&sequence)?;
        Ok(Self {
            id,
            sequence,
            modifications: Vec::new(),
        })
    }

    /// Construct a new `DNAInput` with a pre-built list of modifications.
    pub fn with_modifications(
        id: impl Into<String>,
        sequence: impl Into<String>,
        modifications: Vec<Modification>,
    ) -> Result<Self, String> {
        let id = id.into();
        let sequence = sequence.into();
        validate_dna_sequence(&sequence)?;
        Ok(Self {
            id,
            sequence,
            modifications,
        })
    }

    /// Append a modification and return `self` (builder-style).
    pub fn add_modification(mut self, modification: Modification) -> Self {
        self.modifications.push(modification);
        self
    }

    /// Length of the nucleotide sequence.
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Returns `true` if the sequence is empty.
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

impl fmt::Display for DNAInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DNA[{}](len={}, mods={})",
            self.id,
            self.sequence.len(),
            self.modifications.len()
        )
    }
}

// ── LigandInput ──────────────────────────────────────────────────────────────

/// A small-molecule ligand described by one or more CCD codes.
#[derive(Debug, Clone)]
pub struct LigandInput {
    /// Ligand identifier.
    pub id: String,
    /// Ordered list of CCD codes composing the ligand.
    pub ccd: Vec<String>,
}

impl LigandInput {
    /// Construct a `LigandInput` from an explicit list of CCD codes.
    pub fn new(id: impl Into<String>, ccd: Vec<String>) -> Self {
        Self { id: id.into(), ccd }
    }

    /// Convenience constructor for a single-component ligand.
    pub fn from_ccd(id: impl Into<String>, ccd: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ccd: vec![ccd.into()],
        }
    }

    /// Number of CCD components in this ligand.
    pub fn num_components(&self) -> usize {
        self.ccd.len()
    }
}

impl fmt::Display for LigandInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ligand[{}](ccd={:?})", self.id, self.ccd)
    }
}

// ── ChainInput ───────────────────────────────────────────────────────────────

/// One chain in a multi-entity structure prediction request.
#[derive(Debug, Clone)]
pub enum ChainInput {
    Protein(ProteinInput),
    DNA(DNAInput),
    Ligand(LigandInput),
}

impl ChainInput {
    /// Returns the chain identifier regardless of variant.
    pub fn id(&self) -> &str {
        match self {
            ChainInput::Protein(p) => &p.id,
            ChainInput::DNA(d) => &d.id,
            ChainInput::Ligand(l) => &l.id,
        }
    }
}

impl fmt::Display for ChainInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainInput::Protein(p) => write!(f, "{p}"),
            ChainInput::DNA(d) => write!(f, "{d}"),
            ChainInput::Ligand(l) => write!(f, "{l}"),
        }
    }
}

impl From<ProteinInput> for ChainInput {
    fn from(p: ProteinInput) -> Self {
        ChainInput::Protein(p)
    }
}

impl From<DNAInput> for ChainInput {
    fn from(d: DNAInput) -> Self {
        ChainInput::DNA(d)
    }
}

impl From<LigandInput> for ChainInput {
    fn from(l: LigandInput) -> Self {
        ChainInput::Ligand(l)
    }
}

// ── StructurePredictionInput ─────────────────────────────────────────────────

/// The top-level input for an ESMFold2 structure prediction request.
///
/// Chains are added one at a time via the builder methods.
#[derive(Debug, Clone, Default)]
pub struct StructurePredictionInput {
    /// Ordered list of chains included in the prediction.
    pub sequences: Vec<ChainInput>,
}

impl StructurePredictionInput {
    /// Create an empty prediction input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append any `ChainInput` variant (builder-style, consumes and returns `self`).
    pub fn add_chain(mut self, chain: impl Into<ChainInput>) -> Self {
        self.sequences.push(chain.into());
        self
    }

    /// Append a protein chain.
    pub fn add_protein(self, protein: ProteinInput) -> Self {
        self.add_chain(protein)
    }

    /// Append a DNA chain.
    pub fn add_dna(self, dna: DNAInput) -> Self {
        self.add_chain(dna)
    }

    /// Append a ligand.
    pub fn add_ligand(self, ligand: LigandInput) -> Self {
        self.add_chain(ligand)
    }

    /// Total number of chains (protein + DNA + ligand).
    pub fn num_chains(&self) -> usize {
        self.sequences.len()
    }
}

impl fmt::Display for StructurePredictionInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StructurePredictionInput(chains=[")?;
        for (i, chain) in self.sequences.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{chain}")?;
        }
        write!(f, "])")
    }
}

// ── Validation ───────────────────────────────────────────────────────────────

/// Validate that every character in a protein sequence belongs to the accepted alphabet.
pub fn validate_protein_sequence(sequence: &str) -> Result<(), String> {
    if sequence.is_empty() {
        return Err("Protein sequence must not be empty".to_string());
    }
    for (i, ch) in sequence.char_indices() {
        if !PROTEIN_ALPHABET.contains(ch) {
            return Err(format!(
                "Invalid amino acid '{}' at position {} in protein sequence",
                ch,
                i + 1
            ));
        }
    }
    Ok(())
}

/// Validate that every character in a DNA sequence belongs to the accepted alphabet.
pub fn validate_dna_sequence(sequence: &str) -> Result<(), String> {
    if sequence.is_empty() {
        return Err("DNA sequence must not be empty".to_string());
    }
    for (i, ch) in sequence.char_indices() {
        if !DNA_ALPHABET.contains(ch) {
            return Err(format!(
                "Invalid nucleotide '{}' at position {} in DNA sequence",
                ch,
                i + 1
            ));
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protein_input_valid() {
        let p = ProteinInput::new("A", "ACDEFGHIKLMNPQRSTVWY").unwrap();
        assert_eq!(p.id, "A");
        assert_eq!(p.len(), 20);
        assert!(!p.is_empty());
    }

    #[test]
    fn test_protein_input_ambiguous_codes() {
        // B, J, O, U, X, Z are all acceptable
        ProteinInput::new("A", "BJOUXZ").expect("ambiguous codes should be valid");
    }

    #[test]
    fn test_protein_input_invalid_char() {
        let err = ProteinInput::new("A", "ACDE1FGHIK").unwrap_err();
        assert!(err.contains("'1'"));
    }

    #[test]
    fn test_protein_input_empty() {
        assert!(ProteinInput::new("A", "").is_err());
    }

    #[test]
    fn test_dna_input_valid() {
        let d = DNAInput::new("B", "ACGTNRYSW").unwrap();
        assert_eq!(d.id, "B");
        assert_eq!(d.len(), 9);
    }

    #[test]
    fn test_dna_input_invalid_char() {
        let err = DNAInput::new("B", "ACGT1").unwrap_err();
        assert!(err.contains("'1'"));
    }

    #[test]
    fn test_dna_input_with_modification() {
        let d = DNAInput::new("B", "ACGT")
            .unwrap()
            .add_modification(Modification::new(2, "5MC"));
        assert_eq!(d.modifications.len(), 1);
        assert_eq!(d.modifications[0].ccd, "5MC");
    }

    #[test]
    fn test_ligand_from_ccd() {
        let l = LigandInput::from_ccd("L", "ATP");
        assert_eq!(l.id, "L");
        assert_eq!(l.ccd, vec!["ATP".to_string()]);
        assert_eq!(l.num_components(), 1);
    }

    #[test]
    fn test_structure_prediction_input_builder() {
        let protein = ProteinInput::new("A", "MKTAYIAK").unwrap();
        let dna = DNAInput::new("B", "ACGTACGT").unwrap();
        let ligand = LigandInput::from_ccd("L", "ATP");

        let input = StructurePredictionInput::new()
            .add_protein(protein)
            .add_dna(dna)
            .add_ligand(ligand);

        assert_eq!(input.num_chains(), 3);
        assert_eq!(input.sequences[0].id(), "A");
        assert_eq!(input.sequences[1].id(), "B");
        assert_eq!(input.sequences[2].id(), "L");
    }

    #[test]
    fn test_display_protein() {
        let p = ProteinInput::new("A", "MKTAYIAK").unwrap();
        assert_eq!(format!("{p}"), "Protein[A](len=8)");
    }

    #[test]
    fn test_display_dna() {
        let d = DNAInput::new("B", "ACGT").unwrap();
        assert_eq!(format!("{d}"), "DNA[B](len=4, mods=0)");
    }

    #[test]
    fn test_display_ligand() {
        let l = LigandInput::from_ccd("L", "ATP");
        assert_eq!(format!("{l}"), "Ligand[L](ccd=[\"ATP\"])");
    }

    #[test]
    fn test_display_structure_prediction_input() {
        let input = StructurePredictionInput::new()
            .add_protein(ProteinInput::new("A", "MKTAYIAK").unwrap());
        let s = format!("{input}");
        assert!(s.contains("StructurePredictionInput"));
        assert!(s.contains("Protein[A]"));
    }

    #[test]
    fn test_chain_input_id() {
        let chain: ChainInput = ProteinInput::new("X", "ACDE").unwrap().into();
        assert_eq!(chain.id(), "X");
    }
}
