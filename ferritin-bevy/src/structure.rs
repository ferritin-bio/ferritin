//! Structure.
//!
//! Struct for rendering with Bevy
//!
//!

// use bevy::prelude::*;
use super::ColorScheme;
use bevy::asset::Assets;
use bevy::math::Vec4;
use bevy::prelude::{
    default, Color, Component, Cylinder, Mesh, MeshBuilder, Meshable, PbrBundle, Quat, Sphere,
    StandardMaterial, Transform, Vec3,
};
use bon::Builder;
use ferritin_core::AtomCollection;
use pdbtbx::Element;

/// Enum representing various rendering options.
///
/// Each of these enums represents a rendering path that can be used by a `Structure`
///
/// Donw the Line: allow passing an arbitrary function that maps PDB to mesh.
///
#[derive(Clone)]
pub enum RenderOptions {
    Wireframe,
    Cartoon,
    BallAndStick,
    Solid,
}

/// Define Everything Needed to render
#[derive(Builder, Component)]
pub struct Structure {
    pdb: AtomCollection,
    #[builder(default = RenderOptions::Solid)]
    rendertype: RenderOptions,
    #[builder(default = ColorScheme::Solid(Color::WHITE))]
    color_scheme: ColorScheme,
    #[builder(default = StandardMaterial::default())]
    material: StandardMaterial,
}

impl Structure {
    pub fn to_mesh(&self) -> Mesh {
        match &self.rendertype {
            RenderOptions::Wireframe => self.render_wireframe(),
            RenderOptions::Cartoon => self.render_cartoon(),
            RenderOptions::BallAndStick => self.render_ballandstick(),
            RenderOptions::Solid => self.render_spheres(),
        }
    }
    // this is the onw we probably want
    pub fn to_pbr(
        &self,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
    ) -> PbrBundle {
        let mesh = self.to_mesh();
        let material = self.material.clone();
        PbrBundle {
            mesh: meshes.add(mesh),
            material: materials.add(material),
            // transform: Transform::from_xyz(x, y, z),
            ..default()
        }
    }
    fn render_wireframe(&self) -> Mesh {
        todo!()
    }
    fn render_cartoon(&self) -> Mesh {
        todo!()
    }
    fn render_ballandstick(&self) -> Mesh {
        let radius = 0.5;
        let mut combined_mesh = self
            .pdb
            .iter_coords_and_elements()
            .map(|(coord, element_str)| {
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                // let element = Element::from_symbol(element_str).expect("Element not recognized");
                let color = self.color_scheme.get_color(element_str).to_srgba();
                let color_array =
                    vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vertex_count];
                sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
                sphere_mesh = sphere_mesh.translated_by(center);
                sphere_mesh.compute_smooth_normals();
                sphere_mesh
            })
            .reduce(|mut acc, mesh| {
                acc.merge(&mesh);
                acc
            })
            .unwrap();

        // Add bond cylinders using iterators
        if let Some(bonds) = self.pdb.bonds() {
            let coords = self.pdb.coords();
            bonds
                .iter()
                .filter_map(|bond| {
                    let (atom1, atom2) = bond.get_atom_indices();
                    let pos1 = Vec3::from_array(*coords.get(atom1 as usize)?);
                    let pos2 = Vec3::from_array(*coords.get(atom2 as usize)?);

                    // Calculate cylinder properties
                    let center = (pos1 + pos2) / 2.0;
                    let direction = pos2 - pos1;
                    let height = direction.length();
                    let rotation = Quat::from_rotation_arc(Vec3::Y, direction.normalize());

                    // Create and transform cylinder mesh
                    let mut cylinder_mesh = Cylinder {
                        radius: 0.5,
                        half_height: height / 2.0, // Note: we divide height by 2 since it expects half_height
                    }
                    .mesh()
                    .build();

                    // Apply transformation
                    cylinder_mesh = cylinder_mesh.transformed_by(Transform {
                        translation: center,
                        rotation,
                        ..default()
                    });

                    // Add colors
                    let cylinder_vertex_count = cylinder_mesh.count_vertices();
                    let cylinder_colors =
                        vec![Vec4::new(0.5, 0.5, 0.5, 0.5); cylinder_vertex_count];
                    cylinder_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, cylinder_colors);
                    Some(cylinder_mesh)
                })
                .for_each(|cylinder_mesh| {
                    combined_mesh.merge(&cylinder_mesh);
                });
        } else {
            println!("No-Bonds found!!")
        }

        combined_mesh
    }
    /// Internal fn for rendering spheres.
    fn render_spheres(&self) -> Mesh {
        self.pdb
            .iter_coords_and_elements()
            .map(|(coord, element_str)| {
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let element = Element::from_symbol(element_str).expect("Element not recognized");
                let radius = element
                    .atomic_radius()
                    .van_der_waals
                    .expect("Van der waals not defined") as f32;
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                let color = self.color_scheme.get_color(element_str).to_srgba();
                let color_array =
                    vec![Vec4::new(color.red, color.green, color.blue, color.alpha); vertex_count];
                sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
                sphere_mesh = sphere_mesh.translated_by(center);
                sphere_mesh.compute_smooth_normals();
                sphere_mesh
            })
            .reduce(|mut acc, mesh| {
                acc.merge(&mesh);
                acc
            })
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pdb_to_mesh() {
        let (pdb, _errors) = pdbtbx::open("examples/1fap.cif").unwrap();
        let structure = Structure::builder().pdb(AtomCollection::from(&pdb)).build();
        assert_eq!(structure.pdb.size(), 2154);
        let mesh = structure.to_mesh();
        assert_eq!(mesh.count_vertices(), 779748);
    }
}
