//! Colors
//!
//! This module defines the color mapping used for rendering.
use bevy::prelude::Color;
use pdbtbx::Atom;

/// Represents different color schemes for rendering atoms.
#[derive(Clone)]
pub enum ColorScheme {
    /// A solid, sinel color for all atoms.
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
    pub fn get_color(&self, atom: &Atom) -> Color {
        match &self {
            ColorScheme::Solid(color) => *color,
            ColorScheme::ByAtomType => {
                match atom.element().expect("expect atom").symbol() {
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
    use pdbtbx::Atom;

    #[test]
    fn test_get_color() {
        let by_atom_scheme = ColorScheme::ByAtomType;
        let create_atom =
            |element: &str| Atom::new(true, 1, "", 0.0, 0.0, 0.0, 0.0, 0.0, element, 1).unwrap();
        let carbon_atom = create_atom("C");
        let nitrogen_atom = create_atom("N");
        let oxygen_atom = create_atom("O");
        let sulfur_atom = create_atom("S");
        // let other_atom = create_atom("X");
        // Test ByAtomType color scheme
        assert_eq!(
            by_atom_scheme.get_color(&carbon_atom),
            Color::srgb(0.5, 0.5, 0.5)
        );
        assert_eq!(
            by_atom_scheme.get_color(&nitrogen_atom),
            Color::srgb(0.0, 0.0, 1.0)
        );
        assert_eq!(
            by_atom_scheme.get_color(&oxygen_atom),
            Color::srgb(1.0, 0.0, 0.0)
        );
        assert_eq!(
            by_atom_scheme.get_color(&sulfur_atom),
            Color::srgb(1.0, 1.0, 0.0)
        );
        // assert_eq!(
        //     by_atom_scheme.get_color(&other_atom),
        //     Color::srgb(1.0, 1.0, 1.0)
        // );
    }
}
