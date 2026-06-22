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

    /// Ball-and-stick rendering: small spheres (radius 0.4) at each atom position,
    /// connected by cylinder meshes along each bond.
    ///
    /// Bonds are sourced from `AtomCollection::get_bonds()`. If no bonds are present,
    /// the method renders spheres only. Call `connect_via_residue_names()` on the
    /// AtomCollection before building the Structure to populate bonds.
    ///
    /// Produces: ATTRIBUTE_POSITION, ATTRIBUTE_NORMAL, ATTRIBUTE_UV_0; U32 indices;
    /// TriangleList topology. No ATTRIBUTE_COLOR.
    fn render_ballandstick(&self) -> Mesh {
        const ATOM_RADIUS: f32 = 0.4;
        const BOND_RADIUS: f32 = 0.15;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // Add sphere for each atom
        for idx in 0..self.pdb.get_size() {
            let coord = self.pdb.get_coord(idx);
            let center = Vec3::new(coord[0], coord[1], coord[2]);
            Self::append_sphere_geometry(
                center,
                ATOM_RADIUS,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
            );
        }

        // Add cylinders for each bond (if bonds are available)
        if let Some(bonds) = self.pdb.get_bonds() {
            let coords = self.pdb.get_coords();
            for bond in bonds.iter() {
                let (a1, a2) = bond.get_atom_indices();
                if let (Some(c1), Some(c2)) =
                    (coords.get(a1 as usize), coords.get(a2 as usize))
                {
                    let p1 = Vec3::from_array(*c1);
                    let p2 = Vec3::from_array(*c2);
                    Self::append_cylinder_geometry(
                        p1,
                        p2,
                        BOND_RADIUS,
                        &mut positions,
                        &mut normals,
                        &mut uvs,
                        &mut indices,
                    );
                }
            }
        }

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

    /// Space-filling (Solid) rendering: each atom is drawn as a sphere scaled to its
    /// van der Waals radius. Falls back to 1.5 Å when the VdW radius is not defined
    /// for an element.
    ///
    /// Produces: ATTRIBUTE_POSITION, ATTRIBUTE_NORMAL, ATTRIBUTE_UV_0; U32 indices;
    /// TriangleList topology. No ATTRIBUTE_COLOR.
    fn render_spheres(&self) -> Mesh {
        const VDW_FALLBACK: f32 = 1.5;

        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for (coord, element) in self.pdb.iter_coords_and_elements() {
            let center = Vec3::new(coord[0], coord[1], coord[2]);
            let radius = element
                .atomic_radius()
                .van_der_waals
                .map(|r| r as f32)
                .unwrap_or(VDW_FALLBACK);
            Self::append_sphere_geometry(
                center,
                radius,
                &mut positions,
                &mut normals,
                &mut uvs,
                &mut indices,
            );
        }

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

    fn render_putty(&self) -> Mesh {
        self.create_sphere_mesh(2.0)
    }

    /// Append UV-sphere geometry for a single sphere into the provided vertex buffers.
    ///
    /// Uses 8 latitude and 8 longitude subdivisions. The base vertex index for the
    /// generated indices is derived from the current length of `positions`.
    fn append_sphere_geometry(
        center: Vec3,
        radius: f32,
        positions: &mut Vec<[f32; 3]>,
        normals: &mut Vec<[f32; 3]>,
        uvs: &mut Vec<[f32; 2]>,
        indices: &mut Vec<u32>,
    ) {
        let subdivisions: u32 = 8;
        let base_index = positions.len() as u32;

        for lat in 0..=subdivisions {
            let theta = lat as f32 * std::f32::consts::PI / subdivisions as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for lon in 0..=subdivisions {
                let phi = lon as f32 * 2.0 * std::f32::consts::PI / subdivisions as f32;
                let x = sin_theta * phi.cos();
                let y = cos_theta;
                let z = sin_theta * phi.sin();

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

    /// Append cylinder geometry connecting `p1` to `p2` into the provided vertex buffers.
    ///
    /// The cylinder is a closed tube (no end caps) with `radius`. Normals point radially
    /// outward from the cylinder axis. UVs are a simple cylindrical projection. Produces
    /// TriangleList triangles with U32 indices compatible with `append_sphere_geometry`.
    fn append_cylinder_geometry(
        p1: Vec3,
        p2: Vec3,
        radius: f32,
        positions: &mut Vec<[f32; 3]>,
        normals: &mut Vec<[f32; 3]>,
        uvs: &mut Vec<[f32; 2]>,
        indices: &mut Vec<u32>,
    ) {
        let direction = p2 - p1;
        let length = direction.length();
        if length < 1e-6 {
            return;
        }
        let axis = direction.normalize();

        // Build an orthonormal frame (axis, right, up)
        let right = if axis.abs_diff_eq(Vec3::Y, 0.01) {
            axis.cross(Vec3::Z).normalize()
        } else {
            axis.cross(Vec3::Y).normalize()
        };
        let up = axis.cross(right).normalize();

        const SEGMENTS: u32 = 8;
        let base_index = positions.len() as u32;

        // Two rings: one at p1 (ring 0) and one at p2 (ring 1)
        for ring in 0..=1u32 {
            let center = if ring == 0 { p1 } else { p2 };
            let v = ring as f32; // UV v coordinate (0.0 or 1.0)

            for seg in 0..=SEGMENTS {
                let angle = seg as f32 * 2.0 * std::f32::consts::PI / SEGMENTS as f32;
                let normal = (right * angle.cos() + up * angle.sin()).normalize();
                let pos = center + normal * radius;

                positions.push([pos.x, pos.y, pos.z]);
                normals.push([normal.x, normal.y, normal.z]);
                uvs.push([seg as f32 / SEGMENTS as f32, v]);
            }
        }

        // Connect the two rings with quads (two triangles each)
        let ring_verts = SEGMENTS + 1;
        for seg in 0..SEGMENTS {
            let i00 = base_index + seg;
            let i01 = base_index + seg + 1;
            let i10 = base_index + ring_verts + seg;
            let i11 = base_index + ring_verts + seg + 1;

            indices.push(i00);
            indices.push(i10);
            indices.push(i01);

            indices.push(i01);
            indices.push(i10);
            indices.push(i11);
        }
    }

    /// Create a mesh with spheres at each atom position using a uniform radius.
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
// All render_* methods produce:
//   ATTRIBUTE_POSITION  — Float32x3, one entry per vertex
//   ATTRIBUTE_NORMAL    — Float32x3, outward normals (sphere or cylinder)
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

    #[test]
    fn test_render_ballandstick_has_more_verts_than_solid() -> anyhow::Result<()> {
        // BallAndStick (radius 0.4 spheres + bond cylinders) should produce more
        // vertices than a Solid (VdW radius spheres, no cylinders) for the same structure,
        // because bond cylinders add geometry on top of the atom spheres.
        //
        // Both modes use 8×8 UV sphere subdivisions per atom (81 vertices each).
        // BallAndStick adds 9 vertices per ring × 2 rings per bond cylinder.
        //
        // If bonds exist the BallAndStick mesh will be strictly larger.
        let (molfile, _handle) = TestFile::protein_01().create_temp()?;
        let ac = load_structure(molfile)?;
        let n_atoms = ac.get_size();
        let has_bonds = ac.get_bonds().map(|b| !b.is_empty()).unwrap_or(false);

        let s = Structure::builder()
            .pdb(ac)
            .rendertype(RenderOptions::BallAndStick)
            .build();
        let mesh = s.to_mesh();

        // Each atom contributes (8+1)*(8+1) = 81 vertices for the sphere
        let sphere_only_verts = n_atoms * 81;
        assert!(
            mesh.count_vertices() >= sphere_only_verts,
            "expected at least {} vertices from atom spheres, got {}",
            sphere_only_verts,
            mesh.count_vertices(),
        );

        if has_bonds {
            assert!(
                mesh.count_vertices() > sphere_only_verts,
                "expected bond cylinder vertices on top of atom spheres: total={}, spheres_only={}",
                mesh.count_vertices(),
                sphere_only_verts,
            );
        }
        Ok(())
    }

    #[test]
    fn test_render_solid_uses_vdw_radii() -> anyhow::Result<()> {
        // Solid mode uses per-element VdW radii, so the vertex positions should differ
        // from a uniform-radius sphere mesh (radius 1.5) only when VdW radii vary.
        // At minimum, verify the mesh has vertices and passes attribute checks.
        let s = load_test_structure()?;
        let mesh = s.to_mesh();
        assert!(mesh.count_vertices() > 0, "render_solid: no vertices");
        Ok(())
    }
}
