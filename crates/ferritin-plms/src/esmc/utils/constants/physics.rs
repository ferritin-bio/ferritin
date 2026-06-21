//! Physics constants for backbone geometry.

/// Backbone heavy-atom positions in the canonical local frame.
///
/// Cα is at the origin, Cα→C is along +x, and N lies in the x–y plane.
/// Rows: [N, CA, C].  Shape: `[3, 3]`.  Units: Ångström.
///
/// Source: ideal peptide geometry (Engh & Huber, 1991).
pub const BB_COORDINATES: [[f32; 3]; 3] = [
    [-0.525, 1.363, 0.000], // N
    [0.000, 0.000, 0.000],  // CA (origin)
    [1.526, 0.000, 0.000],  // C
];

/// Ideal Cα–Cβ bond length (Å).
pub const CA_CB_BOND_LENGTH: f32 = 1.522;

/// Ideal Cα–C bond length (Å).
pub const CA_C_BOND_LENGTH: f32 = 1.526;

/// Ideal N–Cα bond length (Å).
pub const N_CA_BOND_LENGTH: f32 = 1.458;

/// Ideal peptide N–C bond length (Å).
pub const PEPTIDE_N_C_BOND_LENGTH: f32 = 1.329;
