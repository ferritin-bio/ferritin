//! Colors
//!
//! This module defines the color mapping used for rendering.

use bevy::color::Srgba;
use bevy::prelude::Color;
use ferritin_core::info::elements::Element;

/// Represents different color schemes for rendering atoms.
#[derive(Clone)]
pub enum ColorScheme {
    /// A solid, single color for all atoms.
    Solid(Color),
    /// Colors atoms based on their element type.
    ByAtomType,
}

/// Get default colors for elements based on the CPK color scheme
#[rustfmt::skip]
pub fn element_color(element: &Element) -> Srgba {
    match element {
        Element::H => Srgba::rgb(1.0, 1.0, 1.0),  // White
        Element::C => Srgba::rgb(0.5, 0.5, 0.5),  // Grey
        Element::N => Srgba::rgb(0.0, 0.0, 1.0),  // Blue
        Element::O => Srgba::rgb(1.0, 0.0, 0.0),  // Red
        Element::P => Srgba::rgb(1.0, 0.5, 0.0),  // Orange
        Element::S => Srgba::rgb(1.0, 1.0, 0.0),  // Yellow
        Element::Cl => Srgba::rgb(0.0, 1.0, 0.0), // Green
        Element::Fe => Srgba::rgb(0.6, 0.0, 0.0), // Dark Red
        Element::Ca => Srgba::rgb(0.5, 0.5, 0.5), // Grey
        Element::Mg => Srgba::rgb(0.5, 1.0, 0.0), // Yellow-Green
        Element::Na => Srgba::rgb(0.0, 0.0, 1.0), // Blue
        Element::K => Srgba::rgb(0.8, 0.6, 1.0),  // Purple
        Element::Zn => Srgba::rgb(0.6, 0.6, 0.6), // Grey
        Element::Cu => Srgba::rgb(0.8, 0.4, 0.0), // Brown
        Element::F => Srgba::rgb(0.7, 1.0, 1.0),  // Light Blue
        Element::Br => Srgba::rgb(0.6, 0.1, 0.1), // Brown
        Element::I => Srgba::rgb(0.4, 0.0, 0.7),  // Purple
        Element::B => Srgba::rgb(1.0, 0.7, 0.7),  // Light Pink
        Element::Se => Srgba::rgb(1.0, 0.5, 0.0), // Orange
        _ => Srgba::rgb(0.5, 0.5, 0.5), // Default Grey
         // Element::Other => Color::srgb(0.5, 0.5, 0.5), // Default Grey
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_colors() {
        let color = element_color(&Element::H);
        assert!(color.red > 0.0);
    }
}
