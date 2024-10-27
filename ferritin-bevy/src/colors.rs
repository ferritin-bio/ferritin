//! Colors
//!
//! This module defines the color mapping used for rendering.
use bevy::prelude::Color;
use pdbtbx::Atom;

/// Represents different color schemes for rendering atoms.
#[derive(Clone)]
pub enum ColorScheme {
    /// A solid, single color for all atoms.
    Solid(Color),
    /// Colors atoms based on their element type.
    ByAtomType,
    // /// Colors atoms based on the chain they belong to.
    // ByChain(Box<dyn Fn(&Chain) -> Color>),
    // /// Colors atoms based on the secondary structure of their residue.
    // BySecondaryStructure(Box<dyn Fn(&Residue) -> Color>),
    // /// Colors atoms based on their residue type.
    // ByResidueType(Box<dyn Fn(&Residue) -> Color>),
    // /// Custom coloring function that takes atom, residue, and chain information.
    // Custom(Box<dyn Fn(&Atom, &Residue, &Chain) -> Color>),
}

// ColorScheme::ByChain(func) => func(chain),
// ColorScheme::BySecondaryStructure(func) => func(residue),
// ColorScheme::ByResidueType(func) => func(residue),
// ColorScheme::Custom(func) => func(atom, residue, chain),
impl ColorScheme {
    pub fn get_color(&self, atom: &str) -> Color {
        match &self {
            ColorScheme::Solid(color) => *color,
            ColorScheme::ByAtomType => {
                match atom {
                    "C" => Color::srgb(0.5, 0.5, 0.5), // Carbon: Gray
                    "N" => Color::srgb(0.0, 0.0, 1.0), // Nitrogen: Blue
                    "O" => Color::srgb(1.0, 0.0, 0.0), // Oxygen: Red
                    "S" => Color::srgb(1.0, 1.0, 0.0), // Sulfur: Yellow
                    _ => Color::srgb(1.0, 1.0, 1.0),   // Other: White
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_color() {
        let by_atom_scheme = ColorScheme::ByAtomType;
        assert_eq!(by_atom_scheme.get_color("C"), Color::srgb(0.5, 0.5, 0.5));
        assert_eq!(by_atom_scheme.get_color("N"), Color::srgb(0.0, 0.0, 1.0));
        assert_eq!(by_atom_scheme.get_color("O"), Color::srgb(1.0, 0.0, 0.0));
        assert_eq!(by_atom_scheme.get_color("S"), Color::srgb(1.0, 1.0, 0.0));
    }
}
