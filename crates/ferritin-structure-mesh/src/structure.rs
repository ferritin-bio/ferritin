//! Structure.
//!
//! Struct for rendering protein structures
//!

use super::ColorScheme;
use bevy::prelude::*;
use bon::Builder;
use ferritin_core::AtomCollection;

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
    #[builder(default = ColorScheme::ByAtomType)]
    color_scheme: ColorScheme,
    #[builder(default = StandardMaterial::default())]
    material: StandardMaterial,
}

// Basic implementation without feature gates
impl Structure {
    // Basic methods that don't depend on bevy or rerun
}

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

    // Rendering method implementations
    fn render_wireframe(&self) -> Mesh {
        self.create_sphere_mesh(0.5)
    }

    fn render_cartoon(&self) -> Mesh {
        self.create_sphere_mesh(1.0)
    }

    fn render_ballandstick(&self) -> Mesh {
        self.create_sphere_mesh(0.3)
    }

    fn render_spheres(&self) -> Mesh {
        self.create_sphere_mesh(1.5)
    }

    fn render_putty(&self) -> Mesh {
        self.create_sphere_mesh(2.0)
    }

    /// Create a mesh with spheres at each atom position
    fn create_sphere_mesh(&self, radius: f32) -> Mesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();

        let subdivisions = 8;

        for idx in 0..self.pdb.get_size() {
            let coord = self.pdb.get_coord(idx);
            let center = Vec3::new(coord[0], coord[1], coord[2]);
            let base_index = positions.len() as u32;

            // Generate a sphere using UV sphere algorithm
            for lat in 0..=subdivisions {
                let theta = lat as f32 * std::f32::consts::PI / subdivisions as f32;
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();

                for lon in 0..=subdivisions {
                    let phi = lon as f32 * 2.0 * std::f32::consts::PI / subdivisions as f32;
                    let sin_phi = phi.sin();
                    let cos_phi = phi.cos();

                    let x = sin_theta * cos_phi;
                    let y = cos_theta;
                    let z = sin_theta * sin_phi;

                    let normal = Vec3::new(x, y, z);
                    let pos = center + normal * radius;

                    positions.push([pos.x, pos.y, pos.z]);
                    normals.push([normal.x, normal.y, normal.z]);
                    uvs.push([
                        lon as f32 / subdivisions as f32,
                        lat as f32 / subdivisions as f32,
                    ]);
                }
            }

            // Generate indices for the sphere
            for lat in 0..subdivisions {
                for lon in 0..subdivisions {
                    let first = base_index + lat * (subdivisions + 1) + lon;
                    let second = first + subdivisions + 1;

                    indices.push(first);
                    indices.push(second);
                    indices.push(first + 1);

                    indices.push(second);
                    indices.push(second + 1);
                    indices.push(first + 1);
                }
            }
        }

        // Create mesh with proper vertex attributes
        let mut mesh = Mesh::new(
            bevy::mesh::PrimitiveTopology::TriangleList,
            bevy::asset::RenderAssetUsages::default(),
        );

        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(bevy::mesh::Indices::U32(indices));

        mesh
    }
}
