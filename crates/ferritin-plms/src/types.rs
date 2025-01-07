//! Types for Standardizing Outputs for Viz

use serde::{Deserialize, Serialize};

// Struct for Handling soft-maxxed logits
#[derive(Debug, Serialize, Deserialize)]
pub struct PseudoProbability {
    pub position: usize,
    pub pseudo_prob: f32,
    pub amino_acid: char,
}

// Struct for Contact Maps
#[derive(Debug, Serialize, Deserialize)]
pub struct ContactMap {
    pub position_1: usize,
    pub position_2: usize,
    pub amino_acid_1: char,
    pub amino_acid_2: char,
    pub layer: usize,
    pub contact_estimate: f32,
}
