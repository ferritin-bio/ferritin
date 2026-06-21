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

    /// Create a mesh with spheres at each atom position.
    ///
    /// Produces: ATTRIBUTE_POSITION, ATTRIBUTE_NORMAL, ATTRIBUTE_UV_0; U32 indices;
    /// TriangleList topology. No ATTRIBUTE_COLOR.
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

// Mesh attribute contract for ferritin-structure-mesh renderers
//
// All render_* methods are implemented via create_sphere_mesh and produce:
//   ATTRIBUTE_POSITION  — Float32x3, one entry per vertex
//   ATTRIBUTE_NORMAL    — Float32x3, outward sphere normals
//   ATTRIBUTE_UV_0      — Float32x2, lat/lon UV coordinates
//   Indices             — U32 format, TriangleList topology
//   ATTRIBUTE_COLOR     — NOT populated by any method in this crate
//
// Topology: TriangleList
// Index format: U32

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::mesh::Indices;
    use bevy::render::mesh::VertexAttributeValues;
    use ferritin_core::load_structure;
    use ferritin_test_data::TestFile;

    fn load_test_structure() -> anyhow::Result<Structure> {
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(molfile)?;
        Ok(Structure::builder().pdb(ac).build())
    }

    fn assert_mesh_has_required_attributes(mesh: &Mesh, label: &str) {
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_POSITION).is_some(),
            "{label}: missing ATTRIBUTE_POSITION"
        );
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL).is_some(),
            "{label}: missing ATTRIBUTE_NORMAL"
        );
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_UV_0).is_some(),
            "{label}: missing ATTRIBUTE_UV_0"
        );
        assert!(
            mesh.indices().is_some(),
            "{label}: missing indices"
        );
    }

    fn assert_indices_are_u32(mesh: &Mesh, label: &str) {
        match mesh.indices() {
            Some(Indices::U32(_)) => {}
            Some(Indices::U16(_)) => panic!("{label}: expected U32 indices, got U16"),
            None => panic!("{label}: no indices"),
        }
    }

    fn assert_no_color_attribute(mesh: &Mesh, label: &str) {
        assert!(
            mesh.attribute(Mesh::ATTRIBUTE_COLOR).is_none(),
            "{label}: ATTRIBUTE_COLOR should not be populated"
        );
    }

    fn assert_positions_are_float32x3(mesh: &Mesh, label: &str) {
        match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
            Some(VertexAttributeValues::Float32x3(_)) => {}
            other => panic!("{label}: ATTRIBUTE_POSITION is {other:?}, expected Float32x3"),
        }
    }

    fn assert_normals_are_float32x3(mesh: &Mesh, label: &str) {
        match mesh.attribute(Mesh::ATTRIBUTE_NORMAL) {
            Some(VertexAttributeValues::Float32x3(_)) => {}
            other => panic!("{label}: ATTRIBUTE_NORMAL is {other:?}, expected Float32x3"),
        }
    }

    fn assert_uvs_are_float32x2(mesh: &Mesh, label: &str) {
        match mesh.attribute(Mesh::ATTRIBUTE_UV_0) {
            Some(VertexAttributeValues::Float32x2(_)) => {}
            other => panic!("{label}: ATTRIBUTE_UV_0 is {other:?}, expected Float32x2"),
        }
    }

    #[test]
    fn test_render_solid_attributes() -> anyhow::Result<()> {
        let s = load_test_structure()?;
        let mesh = s.to_mesh(); // Solid is the default
        assert_mesh_has_required_attributes(&mesh, "render_solid");
        assert_indices_are_u32(&mesh, "render_solid");
        assert_positions_are_float32x3(&mesh, "render_solid");
        assert_normals_are_float32x3(&mesh, "render_solid");
        assert_uvs_are_float32x2(&mesh, "render_solid");
        assert_no_color_attribute(&mesh, "render_solid");
        assert!(mesh.count_vertices() > 0, "render_solid: no vertices");
        Ok(())
    }

    #[test]
    fn test_render_wireframe_attributes() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(molfile)?;
        let s = Structure::builder().pdb(ac).rendertype(RenderOptions::Wireframe).build();
        let mesh = s.to_mesh();
        assert_mesh_has_required_attributes(&mesh, "render_wireframe");
        assert_indices_are_u32(&mesh, "render_wireframe");
        assert_positions_are_float32x3(&mesh, "render_wireframe");
        assert_normals_are_float32x3(&mesh, "render_wireframe");
        assert_uvs_are_float32x2(&mesh, "render_wireframe");
        assert_no_color_attribute(&mesh, "render_wireframe");
        Ok(())
    }

    #[test]
    fn test_render_cartoon_attributes() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(molfile)?;
        let s = Structure::builder().pdb(ac).rendertype(RenderOptions::Cartoon).build();
        let mesh = s.to_mesh();
        assert_mesh_has_required_attributes(&mesh, "render_cartoon");
        assert_indices_are_u32(&mesh, "render_cartoon");
        assert_positions_are_float32x3(&mesh, "render_cartoon");
        assert_normals_are_float32x3(&mesh, "render_cartoon");
        assert_uvs_are_float32x2(&mesh, "render_cartoon");
        assert_no_color_attribute(&mesh, "render_cartoon");
        Ok(())
    }

    #[test]
    fn test_render_ballandstick_attributes() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(molfile)?;
        let s = Structure::builder().pdb(ac).rendertype(RenderOptions::BallAndStick).build();
        let mesh = s.to_mesh();
        assert_mesh_has_required_attributes(&mesh, "render_ballandstick");
        assert_indices_are_u32(&mesh, "render_ballandstick");
        assert_positions_are_float32x3(&mesh, "render_ballandstick");
        assert_normals_are_float32x3(&mesh, "render_ballandstick");
        assert_uvs_are_float32x2(&mesh, "render_ballandstick");
        assert_no_color_attribute(&mesh, "render_ballandstick");
        Ok(())
    }

    #[test]
    fn test_render_putty_attributes() -> anyhow::Result<()> {
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(molfile)?;
        let s = Structure::builder().pdb(ac).rendertype(RenderOptions::Putty).build();
        let mesh = s.to_mesh();
        assert_mesh_has_required_attributes(&mesh, "render_putty");
        assert_indices_are_u32(&mesh, "render_putty");
        assert_positions_are_float32x3(&mesh, "render_putty");
        assert_normals_are_float32x3(&mesh, "render_putty");
        assert_uvs_are_float32x2(&mesh, "render_putty");
        assert_no_color_attribute(&mesh, "render_putty");
        Ok(())
    }

    #[test]
    fn test_vertex_and_index_counts_are_consistent() -> anyhow::Result<()> {
        let s = load_test_structure()?;
        let mesh = s.to_mesh();
        let n_verts = mesh.count_vertices();
        assert!(n_verts > 0, "mesh has no vertices");
        if let Some(Indices::U32(idx)) = mesh.indices() {
            assert!(!idx.is_empty(), "mesh has no indices");
            assert!(
                idx.iter().all(|&i| (i as usize) < n_verts),
                "index out of bounds"
            );
        }
        Ok(())
    }
}
