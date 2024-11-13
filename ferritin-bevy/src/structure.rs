//! Structure.
//!
//! Struct for rendering with Bevy
//!
//!

use super::ColorScheme;
use bevy::log::tracing_subscriber::reload::Error;
use bevy::math::Vec4;
use bevy::prelude::{
    default, Color, Component, Cylinder, Mesh, MeshBuilder, Meshable, Quat, Sphere,
    StandardMaterial, Transform, Vec3,
};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bon::Builder;
use ferritin_core::AtomCollection;

/// Enum representing various rendering options.
///
/// Each of these enums represents a rendering path that can be used by a `Structure`
///
/// Down the Line: allow passing an arbitrary function that maps PDB to mesh.
///
#[derive(Clone)]
pub enum RenderOptions {
    Wireframe,
    Cartoon,
    BallAndStick,
    Solid,
    Putty,
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
            RenderOptions::Putty => self.render_putty().unwrap(),
        }
    }
    pub fn get_material(&self) -> StandardMaterial {
        self.material.clone()
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
        if let Some(bonds) = self.pdb.get_bonds() {
            let coords = self.pdb.get_coords();
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
            .map(|(coord, element)| {
                let center = Vec3::new(coord[0], coord[1], coord[2]);
                let radius = element
                    .atomic_radius()
                    .van_der_waals
                    .expect("Van der waals not defined") as f32;
                let mut sphere_mesh = Sphere::new(radius).mesh().build();
                let vertex_count = sphere_mesh.count_vertices();
                let color = self.color_scheme.get_color(element).to_srgba();
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
    fn render_putty(&self) -> Result<Mesh, Error> {
        fn create_smooth_curve(points: &[Vec3], segments: usize) -> Vec<Vec3> {
            let mut curve_points = Vec::new();

            for i in 0..points.len() - 1 {
                let p0 = if i == 0 { points[0] } else { points[i - 1] };
                let p1 = points[i];
                let p2 = points[i + 1];
                let p3 = if i + 2 >= points.len() {
                    points[points.len() - 1]
                } else {
                    points[i + 2]
                };

                for t in 0..segments {
                    let t = t as f32 / segments as f32;
                    let pos = catmull_rom(p0, p1, p2, p3, t);
                    curve_points.push(pos);
                }
            }

            curve_points
        }
        /// Catmull-Rom spline interpolation
        fn catmull_rom(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
            let t2 = t * t;
            let t3 = t2 * t;

            let v0 = (p2 - p0) * 0.5;
            let v1 = (p3 - p1) * 0.5;

            (2.0 * p1 - 2.0 * p2 + v0 + v1) * t3
                + (-3.0 * p1 + 3.0 * p2 - 2.0 * v0 - v1) * t2
                + v0 * t
                + p1
        }
        /// Generate a mesh around the curve
        fn generate_tube_mesh(curve: &[Vec3], radius: f32, segments: usize) -> Mesh {
            let mut positions = Vec::new();
            let mut normals = Vec::new();
            let mut uvs = Vec::new();
            let mut indices = Vec::new();
            // Generate circles around each point
            for (i, &center) in curve.iter().enumerate() {
                let forward = if i < curve.len() - 1 {
                    (curve[i + 1] - center).normalize()
                } else {
                    (center - curve[i - 1]).normalize()
                };
                let right = if forward.abs_diff_eq(Vec3::Y, 0.01) {
                    Vec3::X
                } else {
                    forward.cross(Vec3::Y).normalize()
                };
                let up = forward.cross(right);
                // Create vertices around the circle
                for j in 0..segments {
                    let angle = (j as f32 / segments as f32) * std::f32::consts::TAU;
                    let x = angle.cos();
                    let y = angle.sin();
                    let pos = center + (right * x + up * y) * radius;
                    let normal = (pos - center).normalize();
                    positions.push([pos.x, pos.y, pos.z]);
                    normals.push([normal.x, normal.y, normal.z]);
                    uvs.push([
                        i as f32 / (curve.len() - 1) as f32,
                        j as f32 / segments as f32,
                    ]);
                }
            }
            // Generate indices for triangles
            for i in 0..curve.len() - 1 {
                for j in 0..segments {
                    let next_j = (j + 1) % segments;
                    let current_ring = i * segments;
                    let next_ring = (i + 1) * segments;
                    indices.push(current_ring + j);
                    indices.push(next_ring + j);
                    indices.push(current_ring + next_j);
                    indices.push(current_ring + next_j);
                    indices.push(next_ring + j);
                    indices.push(next_ring + next_j);
                }
            }

            let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::all());
            mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
            mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
            mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
            mesh.insert_indices(Indices::U32(indices.iter().map(|&i| i as u32).collect()));
            mesh
        }

        // retain the ca alphas
        let c_alphas: Vec<Vec3> = self
            .pdb
            .iter_residues_aminoacid()
            .map(|residue| {
                let ca = residue.find_atom_by_name("CA").expect("CA in all residues");
                Vec3::from_array(ca.coords.clone())
            })
            .collect();
        let curve = create_smooth_curve(&c_alphas, 3);
        let tube_mesh = generate_tube_mesh(&curve, 0.3, 16);
        Ok(tube_mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pdb_to_mesh() {
        let (pdb, _errors) = pdbtbx::open("examples/1fap.cif").unwrap();
        let structure = Structure::builder().pdb(AtomCollection::from(&pdb)).build();
        assert_eq!(structure.pdb.get_size(), 2154);
        let mesh = structure.to_mesh();
        assert_eq!(mesh.count_vertices(), 779748);
    }
}
