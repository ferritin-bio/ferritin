//! Color schemes for protein structure rendering.

use bevy::color::Srgba;
use bevy::prelude::Color;
use ferritin_core::info::elements::Element;

/// Represents different color schemes for rendering atoms.
#[derive(Clone)]
pub enum ColorScheme {
    /// A solid, single color for all atoms.
    Solid(Color),
    /// Colors atoms based on their element type (CPK scheme).
    ByAtomType,
}

impl ColorScheme {
    pub fn get_color(&self, element: &Element) -> Color {
        match self {
            ColorScheme::Solid(color) => *color,
            ColorScheme::ByAtomType => Color::Srgba(element_color(element)),
        }
    }
}

/// CPK element colors.
#[rustfmt::skip]
pub fn element_color(element: &Element) -> Srgba {
    match element {
        Element::H  => Srgba::rgb(1.0, 1.0, 1.0),
        Element::C  => Srgba::rgb(0.5, 0.5, 0.5),
        Element::N  => Srgba::rgb(0.0, 0.0, 1.0),
        Element::O  => Srgba::rgb(1.0, 0.0, 0.0),
        Element::P  => Srgba::rgb(1.0, 0.5, 0.0),
        Element::S  => Srgba::rgb(1.0, 1.0, 0.0),
        Element::Cl => Srgba::rgb(0.0, 1.0, 0.0),
        Element::Fe => Srgba::rgb(0.6, 0.0, 0.0),
        Element::Ca => Srgba::rgb(0.5, 0.5, 0.5),
        Element::Mg => Srgba::rgb(0.5, 1.0, 0.0),
        Element::Na => Srgba::rgb(0.0, 0.0, 1.0),
        Element::K  => Srgba::rgb(0.8, 0.6, 1.0),
        Element::Zn => Srgba::rgb(0.6, 0.6, 0.6),
        Element::Cu => Srgba::rgb(0.8, 0.4, 0.0),
        Element::F  => Srgba::rgb(0.7, 1.0, 1.0),
        Element::Br => Srgba::rgb(0.6, 0.1, 0.1),
        Element::I  => Srgba::rgb(0.4, 0.0, 0.7),
        Element::B  => Srgba::rgb(1.0, 0.7, 0.7),
        Element::Se => Srgba::rgb(1.0, 0.5, 0.0),
        _           => Srgba::rgb(0.5, 0.5, 0.5),
    }
}
