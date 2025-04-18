//! Structure.
//!
//! Struct for rendering protein structures
//!

use super::ColorScheme;
use bon::Builder;
use ferritin_core::AtomCollection;

#[cfg(feature = "bevy")]
use bevy::prelude::*;
#[cfg(feature = "bevy")]
use bevy::render::mesh::{Indices, PrimitiveTopology};
#[cfg(feature = "bevy")]
use bevy::render::render_asset::RenderAssetUsages;

/// Rendering options for protein structures
#[derive(Clone)]
pub enum RenderOptions {
    Wireframe,
    Cartoon,
    BallAndStick,
    Solid,
    Putty,
}

/// Structure represents a molecular structure that can be rendered
#[derive(Builder, Clone)]
pub struct Structure {
    pdb: AtomCollection,
    #[builder(default = RenderOptions::Solid)]
    rendertype: RenderOptions,
    #[cfg_attr(feature = "bevy", builder(default = ColorScheme::Solid(Color::WHITE)))]
    #[cfg_attr(not(feature = "bevy"), builder(default = ColorScheme::ByAtomType))]
    color_scheme: ColorScheme,
    #[cfg(feature = "bevy")]
    #[builder(default = StandardMaterial::default())]
    material: StandardMaterial,
}

// Basic implementation without feature gates
impl Structure {
    // Basic methods that don't depend on bevy or rerun
}

#[cfg(feature = "bevy")]
impl Structure {
    /// Convert the structure to a mesh using the specified render type
    pub fn to_mesh(&self) -> Mesh {
        match self.rendertype {
            RenderOptions::Wireframe => self.render_wireframe(),
            RenderOptions::Cartoon => self.render_cartoon(),
            RenderOptions::BallAndStick => self.render_ballandstick(),
            RenderOptions::Solid => self.render_spheres(),
            RenderOptions::Putty => self.render_putty(),
        }
    }
    
    /// Get the material used for rendering
    pub fn get_material(&self) -> StandardMaterial {
        self.material.clone()
    }
    
    // Placeholder implementations of rendering methods
    fn render_wireframe(&self) -> Mesh {
        Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default())
    }
    
    fn render_cartoon(&self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
    }
    
    fn render_ballandstick(&self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
    }
    
    fn render_spheres(&self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
    }
    
    fn render_putty(&self) -> Mesh {
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
    }
}