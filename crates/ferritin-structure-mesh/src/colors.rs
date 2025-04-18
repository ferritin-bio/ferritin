//! Colors
//!
//! This module defines the color mapping used for rendering.

use ferritin_core::info::elements::Element;

#[cfg(feature = "bevy")]
use bevy::prelude::Color;

/// A simple color type for use when bevy is not available
#[cfg(not(feature = "bevy"))]
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

#[cfg(not(feature = "bevy"))]
impl Color {
    pub const WHITE: Color = Color {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const RED: Color = Color {
        r: 1.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const GREEN: Color = Color {
        r: 0.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };
    pub const BLUE: Color = Color {
        r: 0.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
}

/// Represents different color schemes for rendering atoms.
#[derive(Clone)]
pub enum ColorScheme {
    /// A solid, single color for all atoms.
    Solid(Color),
    /// Colors atoms based on their element type.
    ByAtomType,
}

/// Get default colors for elements based on the CPK color scheme
#[cfg(feature = "bevy")]
#[rustfmt::skip]
pub fn element_color(element: &Element) -> Color {
    match element {
        Element::H => Color::rgb(1.0, 1.0, 1.0),  // White
        Element::C => Color::rgb(0.5, 0.5, 0.5),  // Grey
        Element::N => Color::rgb(0.0, 0.0, 1.0),  // Blue
        Element::O => Color::rgb(1.0, 0.0, 0.0),  // Red
        Element::P => Color::rgb(1.0, 0.5, 0.0),  // Orange
        Element::S => Color::rgb(1.0, 1.0, 0.0),  // Yellow
        Element::Cl => Color::rgb(0.0, 1.0, 0.0), // Green
        Element::Fe => Color::rgb(0.6, 0.0, 0.0), // Dark Red
        Element::Ca => Color::rgb(0.5, 0.5, 0.5), // Grey
        Element::Mg => Color::rgb(0.5, 1.0, 0.0), // Yellow-Green
        Element::Na => Color::rgb(0.0, 0.0, 1.0), // Blue
        Element::K => Color::rgb(0.8, 0.6, 1.0),  // Purple
        Element::Zn => Color::rgb(0.6, 0.6, 0.6), // Grey
        Element::Cu => Color::rgb(0.8, 0.4, 0.0), // Brown
        Element::F => Color::rgb(0.7, 1.0, 1.0),  // Light Blue
        Element::Br => Color::rgb(0.6, 0.1, 0.1), // Brown
        Element::I => Color::rgb(0.4, 0.0, 0.7),  // Purple
        Element::B => Color::rgb(1.0, 0.7, 0.7),  // Light Pink
        Element::Se => Color::rgb(1.0, 0.5, 0.0), // Orange
        Element::Other => Color::rgb(0.5, 0.5, 0.5), // Default Grey
    }
}

/// Get default colors for elements based on the CPK color scheme
#[cfg(not(feature = "bevy"))]
#[rustfmt::skip]
pub fn element_color(element: &Element) -> Color {
    match element {
        Element::H => Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },  // White
        Element::C => Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },  // Grey
        Element::N => Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 },  // Blue
        Element::O => Color { r: 1.0, g: 0.0, b: 0.0, a: 1.0 },  // Red
        Element::P => Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 },  // Orange
        Element::S => Color { r: 1.0, g: 1.0, b: 0.0, a: 1.0 },  // Yellow
        Element::Cl => Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 }, // Green
        Element::Fe => Color { r: 0.6, g: 0.0, b: 0.0, a: 1.0 }, // Dark Red
        Element::Ca => Color { r: 0.5, g: 0.5, b: 0.5, a: 1.0 }, // Grey
        Element::Mg => Color { r: 0.5, g: 1.0, b: 0.0, a: 1.0 }, // Yellow-Green
        Element::Na => Color { r: 0.0, g: 0.0, b: 1.0, a: 1.0 }, // Blue
        Element::K => Color { r: 0.8, g: 0.6, b: 1.0, a: 1.0 },  // Purple
        Element::Zn => Color { r: 0.6, g: 0.6, b: 0.6, a: 1.0 }, // Grey
        Element::Cu => Color { r: 0.8, g: 0.4, b: 0.0, a: 1.0 }, // Brown
        Element::F => Color { r: 0.7, g: 1.0, b: 1.0, a: 1.0 },  // Light Blue
        Element::Br => Color { r: 0.6, g: 0.1, b: 0.1, a: 1.0 }, // Brown
        Element::I => Color { r: 0.4, g: 0.0, b: 0.7, a: 1.0 },  // Purple
        Element::B => Color { r: 1.0, g: 0.7, b: 0.7, a: 1.0 },  // Light Pink
        Element::Se => Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 }, // Orange
        _ => Color { r: 1.0, g: 0.5, b: 0.0, a: 1.0 }, // Orange

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_colors() {
        let color = element_color(&Element::H);
        assert!(color.r > 0.0);
    }
}
