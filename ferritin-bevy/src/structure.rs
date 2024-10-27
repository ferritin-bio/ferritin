//! Renderable
//!
//! Trait and Implementations for Generating Representations.
//!
//!

// use bevy::prelude::*;
use super::ColorScheme;
use bevy::asset::Assets;
use bevy::math::Vec4;
use bevy::prelude::{
    default, Color, ColorToComponents, Component, Mesh, MeshBuilder, Meshable, PbrBundle, Sphere,
    StandardMaterial, Vec3,
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
        todo!()
    }
    /// Internal fn for rendering spheres.
    // fn render_spheres(&self) -> Mesh {
    //     let mut meshes = Vec::new();
    //     // atoms: radius

    //     for atom in self.pdb.atoms() {
    //         let (x, y, z) = atom.pos();
    //         let center = Vec3::new(x as f32, y as f32, z as f32);
    //         let radius = atom
    //             .element()
    //             .expect("Atom Element not Defined")
    //             .atomic_radius()
    //             .van_der_waals
    //             .expect("Van der waals not defined") as f32;
    //         let color = self.color_scheme.get_color(atom).to_srgba();
    //         let mut sphere_mesh = Sphere::new(radius).mesh().build();
    //         let vertex_count = sphere_mesh.count_vertices();
    //         let color_array = vec![color.to_vec4(); vertex_count];
    //         sphere_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color_array);
    //         sphere_mesh = sphere_mesh.translated_by(center);
    //         sphere_mesh.compute_smooth_normals();
    //         meshes.push(sphere_mesh);
    //     }
    //     // combine all the meshes together
    //     meshes
    //         .into_iter()
    //         .reduce(|mut acc, mesh| {
    //             acc.merge(&mesh);
    //             acc
    //         })
    //         .unwrap()
    // }

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
                let color = Color::srgb(0.3, 0.5, 0.8).to_srgba();

                // let color_array: Vec<Vec4> = vec![color.into(); vertex_count];
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
        let structure = Structure::builder().pdb(pdb).build();
        assert_eq!(structure.pdb.atom_count(), 2154);
        let mesh = structure.to_mesh();
        assert_eq!(mesh.count_vertices(), 779748);
    }
}
