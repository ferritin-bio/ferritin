//! Types for Standardizing Outputs for Viz

// Struct for Handling soft-maxxed logits
pub struct PseudoProbability {
    position: i32,
    pseudo_prob: f32,
    amino_acid: char,
}

// Struct for Contact Maps
pub struct ContactMap {
    position1: i32,
    position2: i32,
    pseudo_prob: f32,
    amino_acid: char,
}
