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
    pub position1: usize,
    pub position2: usize,
    pub pseudo_prob: f32,
    pub amino_acid: char,
}
