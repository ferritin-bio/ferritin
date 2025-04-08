//! Colors
//!
//! This module defines the color mapping used for rendering.
use bevy::prelude::Color;
use ferritin_core::info::elements::Element;

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
    pub fn get_color(&self, atom: &Element) -> Color {
        match &self {
            ColorScheme::Solid(color) => *color,
            ColorScheme::ByAtomType => {
                match atom {
                    Element::H => Color::WHITE,                   // Hydrogen: White
                    Element::C => Color::srgb(0.2, 0.2, 0.2),     // Carbon: Dark Gray / Black
                    Element::N => Color::srgb(0.0, 0.0, 1.0),     // Nitrogen: Blue
                    Element::O => Color::srgb(1.0, 0.0, 0.0),     // Oxygen: Red
                    Element::F => Color::srgb(0.0, 1.0, 0.0),     // Fluorine: Green
                    Element::Cl => Color::srgb(0.0, 1.0, 0.0),    // Chlorine: Green
                    Element::Br => Color::srgb(0.6, 0.16, 0.0),   // Bromine: Dark Red / Brown
                    Element::I => Color::srgb(0.4, 0.0, 0.75),    // Iodine: Dark Violet
                    Element::He => Color::srgb(0.0, 1.0, 1.0),    // Helium: Cyan
                    Element::Ne => Color::srgb(0.0, 1.0, 1.0),    // Neon: Cyan
                    Element::Ar => Color::srgb(0.0, 1.0, 1.0),    // Argon: Cyan
                    Element::P => Color::srgb(1.0, 0.5, 0.0),     // Phosphorus: Orange
                    Element::S => Color::srgb(1.0, 1.0, 0.0),     // Sulfur: Yellow
                    Element::B => Color::srgb(1.0, 0.7, 0.5),     // Boron: Peach / Tan
                    Element::Li => Color::srgb(0.5, 0.0, 1.0),    // Lithium: Violet
                    Element::Na => Color::srgb(0.5, 0.0, 1.0),    // Sodium: Violet
                    Element::K => Color::srgb(0.5, 0.0, 1.0),     // Potassium: Violet
                    Element::Mg => Color::srgb(0.1, 0.5, 0.0),    // Magnesium: Dark Green
                    Element::Ca => Color::srgb(0.25, 0.25, 0.25), // Calcium: Dark Gray
                    Element::Fe => Color::srgb(0.8, 0.5, 0.0),    // Iron: Dark Orange
                    Element::Au => Color::srgb(1.0, 0.8, 0.0),    // Gold: Gold
                    Element::Ag => Color::srgb(0.75, 0.75, 0.75), // Silver: Light Gray
                    //Element::Ti => Color::GREY50,                 // Titanium: Gray
                    Element::Zn => Color::srgb(0.5, 0.5, 0.7), // Zinc: Bluish Gray
                    // Add more elements as needed
                    _ => Color::srgb(1.0, 0.0, 1.0), // Unknown/Other: Magenta (makes it stand out)
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

        let carbon_color = by_atom_scheme.get_color(&Element::C);
        let nitrogen_color = by_atom_scheme.get_color(&Element::N);
        let oxygen_color = by_atom_scheme.get_color(&Element::O);
        let sulfur_color = by_atom_scheme.get_color(&Element::S);

        {
            assert_eq!(carbon_color, Color::srgb(0.2, 0.2, 0.2));
            assert_eq!(nitrogen_color, Color::srgb(0.0, 0.0, 1.0));
            assert_eq!(oxygen_color, Color::srgb(1.0, 0.0, 0.0));
            assert_eq!(sulfur_color, Color::srgb(1.0, 1.0, 0.0));
        }
    }
}
